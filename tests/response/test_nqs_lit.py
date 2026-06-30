# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.lit_workflow import (
    MoleculeLITWorkflow,
    _direct_double_mc_sr_components,
    _host_batched_data,
    _omega_grid_from_config,
    _parallel_shared_source_pool_dir,
    _reweighted_double_mc_sr_components,
)
from jaqmc.data import BatchedData
from jaqmc.response.nqs_lit import (
    MolecularResponseFermiNet,
    local_action_ratio,
    nqs_lit_source_sampled_stats,
    restore_params_from_checkpoint,
)
from jaqmc.response.symmetry import (
    SymmetryProjector,
    identity_spatial_projector,
    make_spin_projector,
    parity_spatial_projector,
    projected_log_apply,
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


def _tilted_hydrogen_2pz_logpsi(params, data: MoleculeData):
    rel = data.electrons[0] - data.atoms[0]
    return _hydrogen_2pz_logpsi(params, data) + params["tilt"] * rel[0]


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

    def source_ratio(data):
        return -(data.electrons[0, 2] - data.atoms[0, 2])

    stats = nqs_lit_source_sampled_stats(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        batch,
        source_ratio_apply=source_ratio,
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
    assert int(stats.estimator_mode) == 0
    np.testing.assert_allclose(float(stats.source_norm), 1.0)


def test_source_sampled_stats_accepts_projected_source_ratio():
    batch = _h_batch()

    def source_ratio(data):
        return -(data.electrons[0, 2] - data.atoms[0, 2])

    stats = nqs_lit_source_sampled_stats(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        batch,
        source_ratio_apply=source_ratio,
        source_norm=1.0,
        ground_energy=-0.5,
        omega=0.375,
        eta=0.02,
        source_floor=1e-4,
    )

    assert np.isfinite(float(stats.loss))
    assert 0.0 <= float(stats.fidelity) <= 1.0
    np.testing.assert_allclose(float(stats.source_norm), 1.0)


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


def test_parallel_worker_command_uses_shared_source_pool(tmp_path):
    workflow = SimpleNamespace(restore_path=str(tmp_path / "restore"))
    base_config = UPath(tmp_path / "parallel_scan" / "base_config.yaml")
    part_dir = UPath(tmp_path / "parallel_scan" / "block_000")
    shared_pool = UPath(tmp_path / "parallel_scan" / "source_pools")

    command = MoleculeLITWorkflow._parallel_worker_command(
        workflow,
        base_config,
        part_dir,
        np.asarray([0.1, 0.2]),
        run_seed=11,
        source_pool_dir=shared_pool,
    )

    assert f"lit.nqs_source_pool_dir={shared_pool}" in command
    assert f"lit.nqs_source_pool_dir={part_dir / 'source_pools'}" not in command
    assert "lit.omega_values=0.1,0.2" in command


def test_omega_values_override_linear_grid():
    lit_config = SimpleNamespace(
        omega_min=0.0,
        omega_max=1.0,
        omega_points=11,
        omega_values="0.7283574, 0.779748 0.848432",
    )

    np.testing.assert_allclose(
        _omega_grid_from_config(lit_config),
        [0.7283574, 0.779748, 0.848432],
    )


def test_omega_values_must_be_increasing():
    lit_config = SimpleNamespace(
        omega_min=0.0,
        omega_max=1.0,
        omega_points=11,
        omega_values="0.8,0.7",
    )

    with np.testing.assert_raises(ValueError):
        _omega_grid_from_config(lit_config)


def test_parallel_scan_respects_explicit_source_pool_dir(tmp_path):
    parallel_root = UPath(tmp_path / "parallel_scan")
    explicit_pool = UPath(tmp_path / "existing_source_pool")
    lit_config = SimpleNamespace(
        nqs_source_pool_dir=str(explicit_pool),
        nqs_parallel_shared_source_pool=False,
    )

    assert _parallel_shared_source_pool_dir(lit_config, parallel_root) == explicit_pool

    lit_config.nqs_source_pool_dir = ""
    lit_config.nqs_parallel_shared_source_pool = True
    assert _parallel_shared_source_pool_dir(lit_config, parallel_root) == (
        parallel_root / "source_pools"
    )


def test_projected_source_ratio_subtracts_elastic_overlap():
    workflow = SimpleNamespace(lit_config=SimpleNamespace(nqs_projection_eps=1e-12))
    projector = SymmetryProjector(
        spatial=identity_spatial_projector("c1"),
        spin=make_spin_projector((1, 0), target_s=None, enabled=False),
        label="identity",
    )
    data = MoleculeData(
        electrons=jnp.asarray([[0.2, 0.0, 0.0]], dtype=jnp.float32),
        atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )

    def ground_logpsi(params, local_data):
        del params, local_data
        return jnp.asarray(0.0 + 0.0j)

    source_ratio = MoleculeLITWorkflow._make_projected_source_ratio(
        workflow,
        ground_logpsi,
        {},
        axis=0,
        source_center=0.0,
        projector=projector,
        elastic_overlap=-0.2 + 0.0j,
    )

    np.testing.assert_allclose(complex(source_ratio(data)), 0.0 + 0.0j, atol=1e-6)


def test_projected_action_ratio_commutes_projector_with_hamiltonian():
    workflow = SimpleNamespace(
        lit_config=SimpleNamespace(
            eta=0.02,
            nqs_projection_eps=1e-12,
            nqs_projection_chunk_size=1,
        )
    )
    projector = SymmetryProjector(
        spatial=parity_spatial_projector("odd"),
        spin=make_spin_projector((1, 0), target_s=None, enabled=False),
        label="odd",
    )
    projected_apply = projected_log_apply(
        _hydrogen_2pz_logpsi,
        projector,
        eps=1e-12,
        chunk_size=1,
    )
    action_ratio_apply = MoleculeLITWorkflow._make_local_action_ratio_apply(
        workflow,
        projected_apply,
        _hydrogen_2pz_logpsi,
        projector,
        _hydrogen_1s_logpsi,
        {},
        -0.5,
    )
    point = MoleculeData(
        electrons=jnp.asarray([[0.3, -0.2, 0.7]], dtype=jnp.float32),
        atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )

    actual = action_ratio_apply({}, point, 0.4)
    expected = local_action_ratio(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        point,
        ground_energy=-0.5,
        omega=0.4,
        eta=0.02,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=3e-5,
            atol=3e-6,
        )


def test_projected_action_scores_are_finite_for_parameter_derivatives():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = SimpleNamespace(
        eta=0.02,
        nqs_projection_eps=1e-12,
        nqs_projection_chunk_size=1,
        nqs_sr_score_eps=1e-8,
        nqs_source_floor=1e-4,
    )
    projector = SymmetryProjector(
        spatial=parity_spatial_projector("odd"),
        spin=make_spin_projector((1, 0), target_s=None, enabled=False),
        label="odd",
    )
    projected_apply = projected_log_apply(
        _tilted_hydrogen_2pz_logpsi,
        projector,
        eps=1e-12,
        chunk_size=1,
    )
    action_ratio_apply = MoleculeLITWorkflow._make_local_action_ratio_apply(
        workflow,
        projected_apply,
        _tilted_hydrogen_2pz_logpsi,
        projector,
        _hydrogen_1s_logpsi,
        {},
        -0.5,
    )

    def source_ratio(data):
        return -(data.electrons[0, 2] - data.atoms[0, 2])

    score, ratio, source_weight = MoleculeLITWorkflow._source_sampled_action_scores(
        workflow,
        projected_apply,
        {"tilt": jnp.asarray(0.1, dtype=jnp.float32)},
        _hydrogen_1s_logpsi,
        {},
        _h_batch(),
        source_ratio_apply=source_ratio,
        ground_energy=-0.5,
        omega=0.4,
        action_ratio_apply=action_ratio_apply,
    )

    assert score.shape == (_h_batch().batch_size, 1)
    assert ratio.shape == (_h_batch().batch_size,)
    assert source_weight.shape == (_h_batch().batch_size,)
    assert np.all(np.isfinite(np.asarray(score)))
    assert np.all(np.isfinite(np.asarray(ratio)))
    assert np.all(np.isfinite(np.asarray(source_weight)))


def test_host_batched_data_materializes_device_arrays_on_host():
    host_batch = _host_batched_data(_h_batch())

    assert isinstance(host_batch.data.electrons, np.ndarray)
    np.testing.assert_allclose(
        host_batch.data.electrons,
        np.asarray(_h_batch().data.electrons),
    )


def test_projected_response_uses_memory_safe_score_batch():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.config = SimpleNamespace(batch_size=4096)
    workflow.lit_config = SimpleNamespace(
        nqs_symmetry_projectors=True,
        nqs_sr_score_batch_size=16,
        nqs_eval_batch_size=4096,
        nqs_projected_sr_score_batch_cap=16,
        nqs_projected_action_batch_cap=128,
    )

    assert workflow._nqs_sr_score_batch_size(4096) == 16
    assert workflow._nqs_action_batch_size(4096) == 128
    assert workflow._use_nqs_sr_score_chunks(4096)

    workflow.lit_config.nqs_projected_action_batch_cap = 512
    assert workflow._nqs_action_batch_size(4096) == 512


def test_unprojected_response_keeps_configured_score_batch():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.config = SimpleNamespace(batch_size=4096)
    workflow.lit_config = SimpleNamespace(
        nqs_symmetry_projectors=False,
        nqs_sr_score_batch_size=32,
        nqs_eval_batch_size=512,
    )

    assert workflow._nqs_sr_score_batch_size(4096) == 32
    assert workflow._nqs_action_batch_size(4096) == 512


def test_reweighted_double_mc_sr_components_match_article_formula():
    score = jnp.asarray(
        [
            [0.2 + 0.4j, -0.3 + 0.1j],
            [0.5 - 0.2j, 0.7 + 0.3j],
            [-0.1 + 0.8j, 0.4 - 0.5j],
        ],
        dtype=jnp.complex64,
    )
    ratio = jnp.asarray([0.8 + 0.2j, -0.4 + 0.5j, 1.1 - 0.3j], dtype=jnp.complex64)
    source_weight = jnp.asarray([0.7, 1.3, 0.5], dtype=jnp.float32)

    components = _reweighted_double_mc_sr_components(
        score,
        ratio,
        source_weight,
        eps=1e-8,
    )

    phi_weight = source_weight / jnp.sum(source_weight)
    amplitude = jnp.sum(phi_weight * ratio)
    ratio_norm = jnp.sum(phi_weight * jnp.abs(ratio) ** 2)
    psi_weight = phi_weight * jnp.abs(ratio) ** 2 / ratio_norm
    score_mean = jnp.sum(psi_weight[:, None] * score, axis=0, keepdims=True)
    centered_score = score - score_mean
    hloc_conj = jnp.conj(amplitude / ratio)
    expected_gradient = 2.0 * jnp.real(
        jnp.sum(psi_weight[:, None] * centered_score * hloc_conj[:, None], axis=0)
    )
    weighted_score = jnp.sqrt(psi_weight)[:, None] * centered_score
    expected_metric = jnp.real(jnp.conj(weighted_score).T @ weighted_score)
    actual_metric = components.score_aug.T @ components.score_aug

    np.testing.assert_allclose(
        np.asarray(components.gradient),
        np.asarray(expected_gradient),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_metric),
        np.asarray(expected_metric),
        rtol=5e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        float(components.fidelity),
        float(jnp.abs(amplitude) ** 2 / ratio_norm),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(components.reweight_ess),
        float(1.0 / jnp.sum(psi_weight**2)),
        rtol=2e-6,
    )


def test_direct_double_mc_sr_components_match_article_formula():
    score = jnp.asarray(
        [
            [0.2 + 0.4j, -0.3 + 0.1j],
            [0.5 - 0.2j, 0.7 + 0.3j],
            [-0.1 + 0.8j, 0.4 - 0.5j],
        ],
        dtype=jnp.complex64,
    )
    ratio = jnp.asarray([0.8 + 0.2j, -0.4 + 0.5j, 1.1 - 0.3j], dtype=jnp.complex64)
    normalization = jnp.asarray(0.6 - 0.1j, dtype=jnp.complex64)

    components = _direct_double_mc_sr_components(
        score,
        ratio,
        normalization,
        eps=1e-8,
    )

    centered_score = score - jnp.mean(score, axis=0, keepdims=True)
    hloc_conj = jnp.conj(normalization / ratio)
    expected_gradient = 2.0 * jnp.real(
        jnp.mean(centered_score * hloc_conj[:, None], axis=0)
    )
    weighted_score = centered_score / jnp.sqrt(score.shape[0])
    expected_metric = jnp.real(jnp.conj(weighted_score).T @ weighted_score)
    actual_metric = components.score_aug.T @ components.score_aug

    np.testing.assert_allclose(
        np.asarray(components.gradient),
        np.asarray(expected_gradient),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_metric),
        np.asarray(expected_metric),
        rtol=5e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        float(components.fidelity),
        float(jnp.real(jnp.mean(normalization / ratio))),
        rtol=2e-6,
    )
