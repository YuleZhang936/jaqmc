# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001, RUF100

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.response.ferminet_bfnksr import (
    OFFICIAL_CORRELATED_DIPOLE_EE_SCALES,
    OFFICIAL_DRESSING_PAIR_SCALES,
    OFFICIAL_DRESSING_RADIAL_SCALES,
    OFFICIAL_DIFFUSE_PTAIL_KAPPAS,
    OFFICIAL_PARTIAL_WAVE_CLOSURE_SCALE,
    FermiNetGround,
    append_response_block_params,
    apply_official_partial_wave_closure,
    auxiliary_source_values,
    cas_dressed_teacher_values_and_gradients_from_arrays,
    carrier_product_rule_hbar_batch,
    empty_response_basis_params,
    ground_value_single,
    init_cas_dressing_params,
    init_response_params,
    make_response_model,
    molecular_potential,
    response_basis_head_count,
    response_subspace_pretrain_loss,
    response_values,
    rescale_density_pieces_to_common_log_shift,
    sample_cas_dressed_teacher_bright_influence_distribution,
    sample_krylov_teacher_pretrain_distribution,
    select_response_heads,
    source_carrier_value_single,
)
from jaqmc.response.explicit_ritz import (
    explicit_carrier_value_gradient_blocks,
    helium_singlet_p_carrier_specs,
    hydrogen_p_carrier_specs,
    run_explicit_ritz,
    sample_explicit_ritz_bright_influence_distribution,
    scalar_rayleigh_diagnostics,
)
from jaqmc.response.qc_warmstart import (
    build_casscf_krylov_teacher_model,
    build_casscf_krylov_teacher_targets,
    casscf_krylov_teacher_coefficients,
)


class _GaussianWavefunction:
    def phase_logpsi(self, params, data):
        return jnp.asarray(1.0), -jnp.sum(data.electrons**2)


class _HydrogenicWavefunction:
    def phase_logpsi(self, params, data):
        radius = jnp.linalg.norm(data.electrons[0])
        return jnp.asarray(1.0), -radius


def _ground(
    wf,
    *,
    atoms=((0.0, 0.0, 0.0),),
    charges=(2.0,),
    electron_shape=(2, 3),
    nspins=(1, 1),
    energy=-2.8,
) -> FermiNetGround:
    return FermiNetGround(
        wf=wf,
        params={},
        atoms=jnp.asarray(atoms, dtype=jnp.float64),
        charges=jnp.asarray(charges, dtype=jnp.float64),
        electron_shape=electron_shape,
        nspins=nspins,
        energy=energy,
        checkpoint_step=0,
    )


def _response_ground(n_heads: int = 2) -> FermiNetGround:
    ground = _ground(_GaussianWavefunction())
    return FermiNetGround(
        **{
            **ground.__dict__,
            "response_model": make_response_model(
                nspins=ground.nspins,
                n_heads=n_heads,
                hidden=4,
                hidden_double=2,
                layers=1,
                determinants_per_head=1,
            ),
        }
    )


def test_diffuse_ptail_closure_matches_hydrogen_2p_far_tail():
    ground = _ground(
        _HydrogenicWavefunction(),
        charges=(1.0,),
        electron_shape=(1, 3),
        nspins=(1, 0),
        energy=-0.5,
    )
    points = jnp.asarray([[[0.0, 0.0, 40.0]], [[0.0, 0.0, 41.0]]])

    values = auxiliary_source_values(
        ground,
        points,
        jnp.asarray([]),
        dipole_radial_powers=jnp.asarray([0.5]),
        dipole_radial_scale=1.0,
    )[:, 0]
    scaled = np.asarray(values / points[:, 0, 2])

    assert scaled[1] / scaled[0] == pytest.approx(np.exp(-0.5), rel=2e-3)


def test_carrier_product_rule_hbar_matches_direct_hessian():
    ground = _ground(_GaussianWavefunction())
    points = jnp.asarray(
        [
            [[0.0, 0.1, 0.2], [0.2, 0.0, -0.3]],
            [[0.1, 0.0, 0.4], [-0.2, 0.1, -0.2]],
        ]
    )

    def value_fn(point):
        return ground_value_single(ground, point) * source_carrier_value_single(
            ground,
            point,
        )

    def direct_hbar(point):
        hessian = jax.hessian(value_fn)(point)
        laplacian = jnp.trace(jnp.reshape(hessian, (point.size, point.size)))
        return -0.5 * laplacian + (
            molecular_potential(ground, point) - ground.energy
        ) * value_fn(point)

    actual = carrier_product_rule_hbar_batch(
        ground,
        points,
        source_carrier_value_single,
    )
    expected = jax.vmap(direct_hbar)(points)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_casscf_krylov_teacher_coefficients_span_source_hamiltonian_block():
    coefficients, singular_values = casscf_krylov_teacher_coefficients(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 1.0, 0.0]),
        max_teachers=3,
        svd_rtol=1e-12,
        svd_atol=1e-14,
    )

    assert coefficients.shape == (3, 2)
    assert singular_values.shape == (2,)
    projector = coefficients @ coefficients.T
    np.testing.assert_allclose(projector[2], np.asarray([0.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(projector[:2, :2], np.eye(2), atol=1e-12)


def test_casscf_krylov_teacher_targets_have_gradients():
    pytest.importorskip("pyscf")
    ground = _ground(
        _HydrogenicWavefunction(),
        charges=(1.0,),
        electron_shape=(1, 3),
        nspins=(1, 0),
        energy=-0.5,
    )
    points = np.asarray([[[0.1, 0.2, 0.6]], [[-0.2, 0.1, 0.8]]])
    targets = build_casscf_krylov_teacher_targets(
        ground,
        basis="cc-pvdz",
        points=points,
        n_targets=1,
        ncas=0,
        n_roots=5,
        gradients=True,
        finite_difference_step=1e-3,
    )

    assert targets.target_mode == "casscf-krylov-teachers"
    assert targets.values.shape == (points.shape[0], 1)
    assert targets.gradients is not None
    assert targets.gradients.shape == (points.shape[0], 1, 1, 3)
    assert targets.krylov_singular_values is not None
    assert targets.krylov_coefficients is not None
    assert np.all(np.isfinite(targets.values))
    assert np.all(np.isfinite(targets.gradients))


def test_cas_dressed_teacher_identity_preserves_teacher_block():
    ground = _ground(
        _HydrogenicWavefunction(),
        charges=(1.0,),
        electron_shape=(1, 3),
        nspins=(1, 0),
        energy=-0.5,
    )
    params = init_cas_dressing_params(
        jax.random.PRNGKey(25),
        teacher_count=2,
        atom_count=1,
        radial_scale_count=len(OFFICIAL_DRESSING_RADIAL_SCALES),
        pair_scale_count=len(OFFICIAL_DRESSING_PAIR_SCALES),
    )
    points = jnp.asarray([[[0.2, -0.3, 0.7]], [[-0.1, 0.2, 0.5]]])
    teacher_values = jnp.asarray([[1.0, -2.0], [0.5, 0.25]])
    teacher_gradients = jnp.asarray(
        [
            [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]],
            [[[0.5, 0.0, 0.0]], [[0.0, 0.25, 0.0]]],
        ]
    )

    values, gradients = cas_dressed_teacher_values_and_gradients_from_arrays(
        params,
        ground,
        points,
        teacher_values,
        teacher_gradients,
        radial_scales=jnp.asarray(OFFICIAL_DRESSING_RADIAL_SCALES),
        pair_scales=jnp.asarray(OFFICIAL_DRESSING_PAIR_SCALES),
    )

    np.testing.assert_allclose(values, teacher_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gradients,
        teacher_gradients,
        rtol=1e-12,
        atol=1e-12,
    )


def test_explicit_ritz_carrier_blocks_have_source_plus_carriers():
    h_ground = _ground(
        _HydrogenicWavefunction(),
        charges=(1.0,),
        electron_shape=(1, 3),
        nspins=(1, 0),
        energy=-0.5,
    )
    h_points = np.asarray([[[0.2, -0.1, 0.7]], [[-0.3, 0.1, 0.5]]])
    h_specs = hydrogen_p_carrier_specs([0.5, 1.0], laguerre_orders=[0, 1])

    h_values, h_gradients = explicit_carrier_value_gradient_blocks(
        h_ground,
        [h_points],
        h_specs,
        batch_size=2,
    )

    assert h_values[0].shape == (2, 1 + len(h_specs))
    assert h_gradients[0].shape == (2, 1 + len(h_specs), 1, 3)
    assert np.all(np.isfinite(h_values[0]))
    assert np.all(np.isfinite(h_gradients[0]))

    he_ground = _ground(_GaussianWavefunction())
    he_points = np.asarray(
        [
            [[0.2, -0.1, 0.7], [-0.3, 0.1, 0.5]],
            [[-0.2, 0.1, 0.6], [0.4, -0.2, 0.8]],
        ]
    )
    he_specs = helium_singlet_p_carrier_specs(
        [0.5],
        s_decays=[1.7],
        s_laguerre_orders=[0, 1],
        d_decays=[0.8],
        d_laguerre_orders=[0],
        laguerre_orders=[0],
        geminal_gammas=[0.5],
        include_pd=True,
    )

    he_values, he_gradients = explicit_carrier_value_gradient_blocks(
        he_ground,
        [he_points],
        he_specs,
        batch_size=2,
    )

    assert len(he_specs) == 9
    assert any(spec.channel == 1 for spec in he_specs)
    assert he_values[0].shape == (2, 1 + len(he_specs))
    assert he_gradients[0].shape == (2, 1 + len(he_specs), 2, 3)
    assert np.all(np.isfinite(he_values[0]))
    assert np.all(np.isfinite(he_gradients[0]))


def test_explicit_ritz_bright_influence_sampler_returns_finite_density():
    ground = _ground(
        _HydrogenicWavefunction(),
        charges=(1.0,),
        electron_shape=(1, 3),
        nspins=(1, 0),
        energy=-0.5,
    )
    specs = hydrogen_p_carrier_specs([0.4, 0.8], laguerre_orders=[0])
    pilot = run_explicit_ritz(
        ground,
        specs=specs,
        n_samples=24,
        n_blocks=3,
        core_decay=2.0,
        diffuse_decay=0.4,
        seed=301,
        batch_size=8,
        overlap_cutoff=1e-8,
        fixed_whitening=True,
    )
    scalar = scalar_rayleigh_diagnostics(
        pilot.raw_block_overlaps,
        pilot.raw_block_hamiltonians,
        pilot.block_counts,
        pilot.ritz_carrier_coefficients[:, 0],
        full_overlap=pilot.raw_overlap,
        full_hamiltonian=pilot.raw_hamiltonian,
        bootstrap_replicates=4,
        bootstrap_seed=303,
    )
    assert scalar["full"] == pytest.approx(pilot.spectrum.excitation_energies[0])
    assert np.asarray(scalar["bootstrap_values"]).shape == (4,)

    points, density, stats = sample_explicit_ritz_bright_influence_distribution(
        ground,
        specs,
        pilot,
        n_samples=12,
        core_decay=2.0,
        diffuse_decay=0.4,
        batch_size=6,
        seed=302,
        candidate_factor=2,
        max_candidate_samples=32,
    )

    assert points.shape == (12, 1, 3)
    assert density.shape == (12,)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(density))
    assert np.all(density > 0.0)
    assert stats["sampler"] == "explicit_ritz_bright_influence_sobol_resampling"
    assert stats["proposal_samples"] == 24
    assert 0.0 < stats["proposal_ess_fraction"] <= 1.0
    assert 0.0 < stats["resampling_unique_fraction"] <= 1.0
    component_weights = stats["leverage_component_weights"]
    assert component_weights.shape == (4,)
    assert np.all(component_weights >= 0.0)
    assert np.sum(component_weights) == pytest.approx(1.0)
    assert stats["bright_weight"] == pytest.approx(component_weights[2])
    assert stats["leverage_weight"] == pytest.approx(0.0)
    assert stats["aux_weight"] == pytest.approx(component_weights[3])


def test_krylov_teacher_pretrain_sampler_returns_finite_density():
    pytest.importorskip("pyscf")
    ground = _ground(
        _HydrogenicWavefunction(),
        charges=(1.0,),
        electron_shape=(1, 3),
        nspins=(1, 0),
        energy=-0.5,
    )
    teacher_model = build_casscf_krylov_teacher_model(
        ground,
        basis="cc-pvdz",
        n_targets=1,
        ncas=0,
        n_roots=5,
        state_average=False,
    )

    points, density, stats = sample_krylov_teacher_pretrain_distribution(
        ground,
        teacher_model,
        n_samples=8,
        walkers=4,
        burn_in=1,
        steps_between=1,
        width=0.3,
        core_decay=2.0,
        diffuse_decay=0.5,
        ground_weight=1.0,
        teacher_weight=1.0,
        aux_weight=0.25,
        batch_size=8,
        seed=123,
    )

    assert points.shape == (8, 1, 3)
    assert density.shape == (8,)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)
    assert 0.0 <= stats["pmove"] <= 1.0
    assert np.isfinite(stats["ground_norm"])
    assert np.all(np.isfinite(stats["teacher_norms"]))


def test_cas_dressed_teacher_bright_influence_sampler_reports_diagnostics():
    pytest.importorskip("pyscf")
    ground = _ground(
        _HydrogenicWavefunction(),
        charges=(1.0,),
        electron_shape=(1, 3),
        nspins=(1, 0),
        energy=-0.5,
    )
    teacher_model = build_casscf_krylov_teacher_model(
        ground,
        basis="cc-pvdz",
        n_targets=1,
        ncas=0,
        n_roots=5,
        state_average=False,
    )
    params = init_cas_dressing_params(
        jax.random.PRNGKey(28),
        teacher_count=1,
        atom_count=1,
        radial_scale_count=len(OFFICIAL_DRESSING_RADIAL_SCALES),
        pair_scale_count=len(OFFICIAL_DRESSING_PAIR_SCALES),
    )

    points, density, stats = sample_cas_dressed_teacher_bright_influence_distribution(
        params,
        ground,
        teacher_model,
        head_count=1,
        n_samples=12,
        core_decay=2.0,
        diffuse_decay=0.5,
        batch_size=6,
        seed=126,
        basis="cc-pvdz",
        finite_difference_step=1e-3,
        candidate_factor=2,
        max_candidate_samples=32,
    )

    assert points.shape == (12, 1, 3)
    assert density.shape == (12,)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)
    assert stats["sampler"] == "bright_influence_mixture_sobol_resampling"
    assert stats["proposal_samples"] == 24
    assert 0.0 < stats["proposal_ess_fraction"] <= 1.0
    assert 0.0 < stats["resampling_unique_fraction"] <= 1.0
    assert np.isfinite(stats["leverage_normalizer"])
    assert stats["leverage_max"] >= stats["leverage_mean"] > 0.0
    component_weights = stats["leverage_component_weights"]
    assert component_weights.shape == (5,)
    assert np.all(component_weights >= 0.0)
    assert np.sum(component_weights) == pytest.approx(1.0)
    assert stats["ground_weight"] == pytest.approx(component_weights[0])
    assert stats["source_weight"] == pytest.approx(component_weights[1])
    assert stats["teacher_weight"] == pytest.approx(component_weights[2])
    assert stats["dressed_weight"] == pytest.approx(component_weights[3])
    assert stats["aux_weight"] == pytest.approx(component_weights[4])


def test_production_density_pieces_are_rescaled_to_common_log_shift():
    density_a = np.asarray([1.0, np.exp(-2.0)])
    density_b = np.asarray([1.0, np.exp(-3.0)])
    stats = [
        {"density_log_shift": 10.0},
        {"density_log_shift": 12.0},
    ]

    rescaled, global_shift, shifts = rescale_density_pieces_to_common_log_shift(
        [density_a, density_b],
        stats,
    )

    assert global_shift == pytest.approx(12.0)
    np.testing.assert_allclose(shifts, np.asarray([10.0, 12.0]))
    np.testing.assert_allclose(rescaled[0], density_a * np.exp(-2.0))
    np.testing.assert_allclose(rescaled[1], density_b)


def test_response_subspace_pretrain_loss_is_zero_for_in_span_targets():
    ground = _response_ground(n_heads=2)
    params = init_response_params(
        jax.random.PRNGKey(22),
        ground=ground,
        initial_decay_min=0.2,
        initial_decay_max=0.5,
    )
    points = jnp.asarray(
        [
            [[0.2, -0.3, 0.7], [1.1, 0.4, -0.2]],
            [[-0.1, 0.2, 0.5], [0.7, -0.5, -0.4]],
            [[0.3, 0.1, -0.6], [-0.8, 0.2, 0.9]],
            [[-0.4, 0.5, 0.8], [0.6, -0.1, -0.7]],
        ]
    )
    heads = response_values(params, ground, points)
    targets = heads @ jnp.asarray([[1.2, -0.3], [0.4, 0.8]])
    density = jnp.ones((points.shape[0],), dtype=points.dtype)

    loss, stats = response_subspace_pretrain_loss(
        params,
        ground,
        points,
        targets,
        density,
        head_count=2,
        ridge=1e-12,
    )

    assert float(loss) < 1e-8
    assert float(stats["residual_norm"]) < 1e-6


def test_response_block_dictionary_concatenates_independent_blocks():
    ground = _response_ground(n_heads=3)
    points = jnp.asarray(
        [
            [[0.2, -0.3, 0.7], [1.1, 0.4, -0.2]],
            [[-0.1, 0.2, 0.5], [0.7, -0.5, -0.4]],
        ]
    )
    params_a = init_response_params(
        jax.random.PRNGKey(31),
        ground=ground,
        initial_decay_min=0.2,
        initial_decay_max=0.5,
    )
    params_b = init_response_params(
        jax.random.PRNGKey(32),
        ground=ground,
        initial_decay_min=0.2,
        initial_decay_max=0.5,
    )
    block_a = select_response_heads(params_a, raw_head_count=3, head_count=2)
    block_b = select_response_heads(params_b, raw_head_count=3, head_count=1)
    basis = append_response_block_params(empty_response_basis_params(), block_a)
    basis = append_response_block_params(basis, block_b)

    expected = jnp.concatenate(
        [
            response_values(block_a, ground, points),
            response_values(block_b, ground, points),
        ],
        axis=1,
    )

    assert response_basis_head_count(basis, ground) == 3
    np.testing.assert_allclose(response_values(basis, ground, points), expected)


def test_official_closure_installs_diffuse_ptail_grid():
    args = SimpleNamespace(
        aux_source_dipole_radial_powers=[],
        aux_source_dipole_ee_scales=[],
        aux_source_dipole_radial_scale=7.0,
        residual_aux_source_weight=1.0,
    )

    apply_official_partial_wave_closure(args, electron_count=1)

    assert args.aux_source_dipole_radial_powers == list(OFFICIAL_DIFFUSE_PTAIL_KAPPAS)
    assert args.aux_source_dipole_ee_scales == []
    assert args.aux_source_dipole_radial_scale == pytest.approx(
        OFFICIAL_PARTIAL_WAVE_CLOSURE_SCALE
    )
    assert args.residual_aux_source_weight == pytest.approx(0.0)

    apply_official_partial_wave_closure(args, electron_count=2)

    assert args.aux_source_dipole_ee_scales == list(
        OFFICIAL_CORRELATED_DIPOLE_EE_SCALES
    )
