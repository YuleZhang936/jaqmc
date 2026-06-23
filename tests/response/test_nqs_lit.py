# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.data import BatchedData
from jaqmc.response.nqs_lit import (
    MolecularResponseFermiNet,
    local_action_ratio,
    nqs_lit_double_sampled_stats,
    nqs_lit_source_sampled_stats,
    restore_params_from_checkpoint,
)
from jaqmc.utils.checkpoint import NumPyCheckpointManager


def _hydrogen_1s_logpsi(params, data: MoleculeData):
    del params
    return -jnp.linalg.norm(data.electrons[0] - data.atoms[0])


def _hydrogen_2pz_logpsi(params, data: MoleculeData):
    del params
    rel = data.electrons[0] - data.atoms[0]
    sign_phase = jnp.where(rel[2] < 0.0, jnp.pi, 0.0)
    return jnp.log(jnp.abs(rel[2])) - 0.5 * jnp.linalg.norm(rel) + 1j * sign_phase


def _h_batch() -> BatchedData[MoleculeData]:
    return BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[0.2, 0.1, 0.8]],
                    [[-0.3, 0.0, 0.7]],
                    [[0.1, -0.2, 0.9]],
                    [[-0.1, 0.2, 0.6]],
                    [[0.2, -0.2, 0.4]],
                ],
                dtype=jnp.float32,
            ),
            atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=["electrons"],
    )


def test_hydrogen_2pz_response_local_action_is_exact():
    point = MoleculeData(
        electrons=jnp.asarray([[0.3, -0.2, 0.7]], dtype=jnp.float32),
        atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )

    action, response_ratio, local_energy = local_action_ratio(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        point,
        ground_energy=-0.5,
        omega=0.4,
        eta=0.02,
    )

    expected = (0.375 - 0.4 - 0.02j) * response_ratio
    np.testing.assert_allclose(np.asarray(action), np.asarray(expected), rtol=2e-6)
    np.testing.assert_allclose(float(jnp.real(local_energy)), -0.125, rtol=2e-6)


def test_full_response_source_sampled_hydrogen_stats_are_finite():
    batch = _h_batch()

    stats = nqs_lit_source_sampled_stats(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        batch,
        axis=2,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-0.5,
        omega=0.375,
        eta=0.02,
        source_floor=1e-4,
    )

    assert np.isfinite(float(stats.loss))
    assert 0.0 <= float(stats.fidelity) <= 1.0
    assert np.isfinite(float(stats.lit))
    assert float(stats.reweight_ess) > 0.0
    assert 0.0 < float(stats.reweight_ess_fraction) <= 1.0
    assert np.isfinite(float(stats.error_d))
    np.testing.assert_allclose(float(stats.source_norm), 1.0)


def test_double_sampled_hydrogen_stats_reports_direct_estimator():
    batch = _h_batch()

    stats = nqs_lit_double_sampled_stats(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        batch,
        batch,
        axis=2,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-0.5,
        omega=0.375,
        eta=0.02,
        source_floor=1e-4,
    )

    assert int(stats.estimator_mode) == 1
    assert 0.0 <= float(stats.fidelity) <= 1.0
    assert float(stats.action_norm) >= 0.0


def test_molecular_response_ferminet_returns_complex_logpsi():
    batch = _h_batch()
    data = MoleculeData(
        electrons=batch.data.electrons[0],
        atoms=batch.data.atoms,
        charges=batch.data.charges,
    )
    response = MolecularResponseFermiNet(
        nspins=(1, 0),
        ndets=2,
        hidden_dims_single=(4,),
        hidden_dims_double=(2,),
    )
    params = response.init(jax.random.PRNGKey(1), batch.unbatched_example())
    value = response.apply(params, data)

    assert jnp.iscomplexobj(value)
    assert np.isfinite(float(jnp.real(value)))
    assert np.isfinite(float(jnp.imag(value)))


def test_restore_params_from_stage_checkpoint(tmp_path):
    fallback = {"params": {"w": jnp.asarray([0.0, 0.0])}}
    restored = {"params": {"w": jnp.asarray([1.0, 2.0])}}
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    manager.save(7, {"params": restored})

    step, params = restore_params_from_checkpoint(tmp_path, fallback)

    assert step == 7
    np.testing.assert_allclose(np.asarray(params["params"]["w"]), [1.0, 2.0])
