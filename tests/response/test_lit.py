# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.lit_workflow import (
    MolecularLITConfig,
    MoleculeLITWorkflow,
    _batched_data_chunks,
    _cyclic_batched_data_chunk,
    _lit_omega_grid,
    _local_parallel_slots,
    _parallel_worker_slots,
    _parse_parallel_remote_hosts,
    _solve_sr_direction_chunked,
)
from jaqmc.data import BatchedData
from jaqmc.response.lit import (
    broadened_from_lit,
    lit_from_poles,
)


def _hydrogen_1s_np_energies(n_max: int = 4) -> np.ndarray:
    n = np.arange(2, n_max + 1, dtype=np.float64)
    return 0.5 * (1.0 - 1.0 / n**2)


def _hydrogen_1s_np_oscillator_strengths(n_max: int = 4) -> np.ndarray:
    n = np.arange(2, n_max + 1, dtype=np.float64)
    return (
        2**8
        * n**5
        * (n - 1.0) ** (2.0 * n - 4.0)
        / (3.0 * (n + 1.0) ** (2.0 * n + 4.0))
    )


def _hydrogen_1s_np_axis_dipole_strengths(n_max: int = 4) -> np.ndarray:
    energies = _hydrogen_1s_np_energies(n_max)
    oscillator_strengths = _hydrogen_1s_np_oscillator_strengths(n_max)
    return oscillator_strengths / (2.0 * energies)


def test_hydrogen_1s_np_exact_reference_values():
    energies = _hydrogen_1s_np_energies(4)
    oscillator_strengths = _hydrogen_1s_np_oscillator_strengths(4)
    axis_strengths = _hydrogen_1s_np_axis_dipole_strengths(4)

    np.testing.assert_allclose(energies[0], 0.375, rtol=1e-14)
    np.testing.assert_allclose(oscillator_strengths[0], 8192 / 19683, rtol=1e-14)
    np.testing.assert_allclose(
        axis_strengths[0],
        oscillator_strengths[0] / (2.0 * energies[0]),
        rtol=1e-14,
    )


def test_hydrogen_bound_lit_matches_hardcoded_lorentzian_sum():
    omega = np.array([0.35, 0.375, 0.40])
    eta = 0.02
    energies = np.array([0.375, 4 / 9, 15 / 32])
    strengths = _hydrogen_1s_np_axis_dipole_strengths(4)

    expected = broadened_from_lit(lit_from_poles(omega, energies, strengths, eta), eta)
    actual = broadened_from_lit(lit_from_poles(omega, energies, strengths, eta), eta)

    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_parallel_worker_slots_oversubscribe_evenly():
    slots = _local_parallel_slots(("0", "1", "2"), procs_per_device=3)
    selected = _parallel_worker_slots(slots, 8)

    assert tuple(slot.device for slot in selected) == (
        "0",
        "1",
        "2",
        "0",
        "1",
        "2",
        "0",
        "1",
    )


def test_parallel_remote_hosts_parse_ipv6_ranges():
    slots = _parse_parallel_remote_hosts(
        ("10234@fdbd:dc03:16:340::210:0-2,7",),
        remote_root="/opt/tiger/jaqmc",
        remote_python=".venv-gpu/bin/python",
        ssh_options=("-o", "BatchMode=yes"),
        procs_per_device=1,
    )

    assert tuple(slot.device for slot in slots) == ("0", "1", "2", "7")
    assert slots[0].host == "fdbd:dc03:16:340::210"
    assert slots[0].port == 10234
    assert slots[0].root == "/opt/tiger/jaqmc"
    assert slots[0].python == ".venv-gpu/bin/python"
    assert slots[0].ssh_options == ("-o", "BatchMode=yes")


def test_lit_omega_values_override_linspace():
    config = MolecularLITConfig(
        omega_min=0.0,
        omega_max=1.0,
        omega_points=5,
        omega_values=(0.774, 0.775, 0.7765),
    )

    np.testing.assert_allclose(_lit_omega_grid(config), [0.774, 0.775, 0.7765])


def test_lit_omega_values_must_be_strictly_increasing():
    config = MolecularLITConfig(omega_values=(0.775, 0.775, 0.776))

    with pytest.raises(ValueError, match="strictly increasing"):
        _lit_omega_grid(config)


def test_parallel_worker_command_passes_exact_omega_values():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig()
    workflow.restore_path = UPath("ground")

    command = workflow._parallel_worker_command(
        UPath("base_config.yaml"),
        UPath("part"),
        np.asarray([0.774, 0.7765, 0.78]),
        run_seed=123,
        worker_index=2,
    )

    assert "lit.omega_min=0.774" in command
    assert "lit.omega_max=0.78" in command
    assert "lit.omega_points=3" in command
    assert "lit.omega_values=[0.774, 0.7765, 0.78]" in command


def test_batched_data_chunks_cover_pool_and_cycle():
    pool = BatchedData(
        data=MoleculeData(
            electrons=jnp.arange(30, dtype=jnp.float32).reshape(10, 1, 3),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.ones((1,), dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )

    cycled = _cyclic_batched_data_chunk(pool, 4, 3)
    chunks = list(_batched_data_chunks(pool, 4))

    np.testing.assert_array_equal(
        np.asarray(cycled.data.electrons[:, 0, 0]),
        [12, 15, 18, 21],
    )
    assert [chunk.batch_size for chunk in chunks] == [4, 4, 2]
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(chunk.data.electrons[:, 0, 0]) for chunk in chunks]),
        np.arange(0, 30, 3),
    )


def test_chunked_sr_solve_matches_full_metric_branch():
    score_aug = jnp.asarray(np.arange(30, dtype=np.float32).reshape(10, 3) / 17.0)
    grad = jnp.asarray([0.5, -0.25, 0.125], dtype=jnp.float32)
    damping = jnp.asarray(0.03, dtype=jnp.float32)

    full = _solve_sr_direction_chunked(
        (10,),
        lambda _: score_aug,
        grad,
        damping,
    )
    chunks = (score_aug[:4], score_aug[4:7], score_aug[7:])
    chunked = _solve_sr_direction_chunked(
        tuple(chunk.shape[0] for chunk in chunks),
        lambda index: chunks[index],
        grad,
        damping,
    )

    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=5e-5)


def test_chunked_sr_solve_matches_full_kernel_branch():
    score_aug = jnp.asarray(np.arange(24, dtype=np.float32).reshape(4, 6) / 13.0)
    grad = jnp.asarray([0.2, -0.1, 0.3, -0.4, 0.05, 0.7], dtype=jnp.float32)
    damping = jnp.asarray(0.07, dtype=jnp.float32)

    full = _solve_sr_direction_chunked(
        (4,),
        lambda _: score_aug,
        grad,
        damping,
    )
    chunks = (score_aug[:1], score_aug[1:3], score_aug[3:])
    chunked = _solve_sr_direction_chunked(
        tuple(chunk.shape[0] for chunk in chunks),
        lambda index: chunks[index],
        grad,
        damping,
    )

    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=5e-5)
