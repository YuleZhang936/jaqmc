# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.lit_workflow import (
    MolecularLITConfig,
    MoleculeLITWorkflow,
    _add_source_sums,
)
from jaqmc.data import BatchedData
from jaqmc.response.nqs_lit import (
    MolecularResponseFermiNet,
    MolecularVectorResponseFermiNet,
    NQSLITSourceSums,
    local_action_ratio,
    nqs_lit_double_sampled_stats,
    nqs_lit_source_sampled_stats,
    nqs_lit_source_sampled_sums,
    nqs_lit_stats_from_source_sums,
    odd_parity_project_log_amplitude,
    parity_log_amplitude_loss,
    parity_log_amplitude_residual,
    parity_project_log_amplitude,
    restore_params_from_checkpoint,
    source_aligned_vector_logpsi,
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


def _scaled_hydrogen_2pz_logpsi(params, data: MoleculeData):
    return params["scale"] * _hydrogen_2pz_logpsi({}, data)


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
    assert np.isfinite(float(stats.reverse_kl))
    assert float(stats.reverse_kl) >= 0.0
    assert np.isfinite(float(stats.signed_lit))
    assert np.isfinite(float(stats.lit))
    assert float(stats.reweight_ess) > 0.0
    assert 0.0 < float(stats.reweight_ess_fraction) <= 1.0
    assert np.isfinite(float(stats.error_d))
    assert np.isfinite(float(stats.equation_relative_residual))
    assert float(stats.equation_relative_residual) >= 0.0
    np.testing.assert_allclose(float(stats.invalid_sample_fraction), 0.0)
    assert np.isnan(float(stats.direct_hloc_rmse))
    assert np.isnan(float(stats.direct_hloc_std))
    assert np.isnan(float(stats.direct_hloc_sem))
    np.testing.assert_allclose(float(stats.source_norm), 1.0)


def test_fused_source_scores_preserve_source_sampled_sums():
    batch = _h_batch()
    response_params = {"scale": jnp.asarray(1.0, dtype=jnp.float32)}
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        eta=0.02,
        nqs_source_floor=1e-4,
        nqs_sr_score_eps=1e-8,
    )

    _, _, _, fused_sums = workflow._source_sampled_action_scores_and_sums(
        _scaled_hydrogen_2pz_logpsi,
        response_params,
        _hydrogen_1s_logpsi,
        {},
        batch,
        axis=2,
        source_center=0.0,
        ground_energy=-0.5,
        omega=0.375,
    )
    reference_sums = nqs_lit_source_sampled_sums(
        _scaled_hydrogen_2pz_logpsi,
        response_params,
        _hydrogen_1s_logpsi,
        {},
        batch,
        axis=2,
        source_center=0.0,
        ground_energy=-0.5,
        omega=0.375,
        eta=0.02,
        source_floor=1e-4,
    )

    for actual, expected in zip(fused_sums, reference_sums, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5)

    fused_stats = nqs_lit_stats_from_source_sums(
        fused_sums,
        source_norm=1.0,
        omega=0.375,
        eta=0.02,
    )
    reference_stats = nqs_lit_stats_from_source_sums(
        reference_sums,
        source_norm=1.0,
        omega=0.375,
        eta=0.02,
    )
    np.testing.assert_allclose(
        float(fused_stats.fidelity),
        float(reference_stats.fidelity),
    )
    np.testing.assert_allclose(
        float(fused_stats.equation_relative_residual),
        float(reference_stats.equation_relative_residual),
    )


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
    assert np.isfinite(float(stats.reverse_kl))
    assert float(stats.action_norm) >= 0.0
    assert np.isfinite(float(stats.equation_relative_residual))
    assert np.isfinite(float(stats.direct_hloc_rmse))
    assert np.isfinite(float(stats.direct_hloc_std))
    assert np.isfinite(float(stats.direct_hloc_sem))
    assert float(stats.direct_hloc_rmse) >= 0.0
    assert float(stats.direct_hloc_std) >= 0.0
    assert float(stats.direct_hloc_sem) >= 0.0


def test_zero_action_mass_is_finite_and_marked_invalid():
    real = jnp.asarray(0.0, dtype=jnp.float32)
    complex_zero = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex64)
    sums = NQSLITSourceSums(
        sample_count=jnp.asarray(3.0, dtype=jnp.float32),
        weight_sum=jnp.asarray(3.0, dtype=jnp.float32),
        valid_sample_count=jnp.asarray(3.0, dtype=jnp.float32),
        ratio_scale=jnp.asarray(1.0, dtype=jnp.float32),
        ratio_sum=complex_zero,
        ratio_abs2_sum=real,
        psi_weight_sum=real,
        psi_weight_sq_sum=real,
        psi_log_ratio_abs2_sum=real,
        response_conj_over_source_sum=complex_zero,
        response_over_source_abs2_sum=real,
        hbar_over_source_sum=complex_zero,
        hbar_over_source_abs2_sum=real,
        ground_energy_sum=real,
    )

    stats = nqs_lit_stats_from_source_sums(
        sums,
        source_norm=1.0,
        omega=0.2,
        eta=0.05,
    )

    assert np.isfinite(float(stats.fidelity))
    assert np.isfinite(float(stats.reverse_kl))
    np.testing.assert_allclose(float(stats.fidelity), 0.0)
    np.testing.assert_allclose(float(stats.reverse_kl), 0.0)
    np.testing.assert_allclose(float(stats.invalid_sample_fraction), 1.0)


def test_scale_aware_source_moments_survive_float32_dynamic_range():
    def one_sample(scale):
        real_zero = jnp.asarray(0.0, dtype=jnp.float32)
        complex_zero = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex64)
        return NQSLITSourceSums(
            sample_count=jnp.asarray(1.0, dtype=jnp.float32),
            weight_sum=jnp.asarray(1.0, dtype=jnp.float32),
            valid_sample_count=jnp.asarray(1.0, dtype=jnp.float32),
            ratio_scale=jnp.asarray(scale, dtype=jnp.float32),
            ratio_sum=jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64),
            ratio_abs2_sum=jnp.asarray(1.0, dtype=jnp.float32),
            psi_weight_sum=jnp.asarray(1.0, dtype=jnp.float32),
            psi_weight_sq_sum=jnp.asarray(1.0, dtype=jnp.float32),
            psi_log_ratio_abs2_sum=real_zero,
            response_conj_over_source_sum=complex_zero,
            response_over_source_abs2_sum=real_zero,
            hbar_over_source_sum=complex_zero,
            hbar_over_source_abs2_sum=real_zero,
            ground_energy_sum=real_zero,
        )

    sums = _add_source_sums(one_sample(1e30), one_sample(1e-30))
    stats = nqs_lit_stats_from_source_sums(
        sums,
        source_norm=1.0,
        omega=0.2,
        eta=0.05,
    )

    assert np.isfinite(float(stats.fidelity))
    assert np.isfinite(float(stats.reverse_kl))
    assert np.isfinite(float(stats.reweight_ess))
    np.testing.assert_allclose(float(stats.fidelity), 0.5, rtol=2e-6)
    np.testing.assert_allclose(float(stats.reverse_kl), np.log(2.0), rtol=2e-6)


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


def test_molecular_vector_response_ferminet_returns_three_complex_components():
    batch = _h_batch()
    data = MoleculeData(
        electrons=batch.data.electrons[0],
        atoms=batch.data.atoms,
        charges=batch.data.charges,
    )
    response = MolecularVectorResponseFermiNet(
        nspins=(1, 0),
        ndets=2,
        hidden_dims_single=(4,),
        hidden_dims_double=(2,),
    )
    params = response.init(jax.random.PRNGKey(2), batch.unbatched_example())
    value = response.apply(params, data)

    assert value.shape == (3,)
    assert jnp.iscomplexobj(value)
    assert np.all(np.isfinite(np.asarray(value)))


def test_odd_parity_projection_is_stable_under_extreme_common_log_shifts():
    base_log_psi = jnp.asarray(0.4 + 0.3j, dtype=jnp.complex64)
    base_inverted_log_psi = jnp.asarray(-0.7 - 0.2j, dtype=jnp.complex64)
    common_shifts = jnp.asarray([1.0e3, -1.0e3], dtype=jnp.float32)

    projected_logs = odd_parity_project_log_amplitude(
        base_log_psi + common_shifts,
        base_inverted_log_psi + common_shifts,
    )
    scaled_amplitudes = jnp.exp(projected_logs - common_shifts)
    expected = 0.5 * (jnp.exp(base_log_psi) - jnp.exp(base_inverted_log_psi))

    assert np.all(np.isfinite(np.asarray(projected_logs)))
    np.testing.assert_allclose(
        np.asarray(scaled_amplitudes),
        np.broadcast_to(np.asarray(expected), (2,)),
        rtol=2e-4,
        atol=2e-4,
    )


def test_even_parity_projection_is_stable_under_extreme_common_log_shifts():
    base_log_psi = jnp.asarray(0.4 + 0.3j, dtype=jnp.complex64)
    base_inverted_log_psi = jnp.asarray(-0.7 - 0.2j, dtype=jnp.complex64)
    common_shifts = jnp.asarray([1.0e3, -1.0e3], dtype=jnp.float32)

    projected_logs = parity_project_log_amplitude(
        base_log_psi + common_shifts,
        base_inverted_log_psi + common_shifts,
        1,
    )
    scaled_amplitudes = jnp.exp(projected_logs - common_shifts)
    expected = 0.5 * (jnp.exp(base_log_psi) + jnp.exp(base_inverted_log_psi))

    assert np.all(np.isfinite(np.asarray(projected_logs)))
    np.testing.assert_allclose(
        np.asarray(scaled_amplitudes),
        np.broadcast_to(np.asarray(expected), (2,)),
        rtol=2e-4,
        atol=2e-4,
    )


def test_even_parity_projection_encodes_wrapped_antiphase_nodes_exactly():
    pi = jnp.asarray(jnp.pi, dtype=jnp.float32)
    log_psi = jnp.asarray(
        [
            2.0 + 0.0j,
            -700.0 + 0.3j,
            40.0 - 0.7j,
            -40.0 + 0.2j,
            0.1 + 0.0j,
            -0.1 + 0.0j,
        ],
        dtype=jnp.complex64,
    )
    odd_windings = jnp.asarray([1, -1, 3, -3, 5, -5], dtype=jnp.float32)
    inverted_log_psi = log_psi + 1j * odd_windings * pi

    projected = jax.jit(
        lambda first, second: parity_project_log_amplitude(first, second, 1)
    )(log_psi, inverted_log_psi)

    assert np.all(np.isneginf(np.asarray(jnp.real(projected))))
    assert np.all(np.isfinite(np.asarray(jnp.imag(projected))))
    np.testing.assert_array_equal(
        np.asarray(jnp.exp(projected)),
        np.zeros((6,), dtype=np.complex64),
    )


def test_even_parity_projection_preserves_nearly_cancelled_sum():
    pi = jnp.asarray(jnp.pi, dtype=jnp.float32)
    log_psi = jnp.asarray(
        [40.0 + 0.7j, 40.0 - 0.4j],
        dtype=jnp.complex64,
    )
    inverted_log_psi = log_psi + jnp.asarray(
        [-2.0e-5 + 1j * (pi + 3.0e-6), -3.0e-5 + 1j * (-pi - 4.0e-6)],
        dtype=jnp.complex64,
    )

    projected = parity_project_log_amplitude(log_psi, inverted_log_psi, 1)
    scaled_amplitude = jnp.exp(projected - log_psi)
    actual_delta = np.asarray(inverted_log_psi - log_psi, dtype=np.complex64).astype(
        np.complex128
    )
    nearest_antiphase = np.asarray([np.pi, -np.pi], dtype=np.float32).astype(np.float64)
    shifted_delta = actual_delta.real + 1j * (actual_delta.imag - nearest_antiphase)
    expected = -0.5 * np.expm1(shifted_delta)

    np.testing.assert_allclose(
        np.asarray(scaled_amplitude),
        expected,
        rtol=3e-5,
        atol=2e-10,
    )


def test_odd_parity_wrapper_matches_generic_projection():
    log_psi = jnp.asarray([40.0 + 0.7j, -800.0 - 0.1j], dtype=jnp.complex64)
    inverted_log_psi = jnp.asarray(
        [39.99998 + 0.700003j, -801.0 + 0.4j],
        dtype=jnp.complex64,
    )

    wrapped = odd_parity_project_log_amplitude(log_psi, inverted_log_psi)
    generic = parity_project_log_amplitude(log_psi, inverted_log_psi, -1)

    np.testing.assert_array_equal(np.asarray(wrapped), np.asarray(generic))


def test_odd_parity_projection_changes_amplitude_sign_when_swapped():
    log_psi = jnp.asarray(0.2 + 0.8j, dtype=jnp.complex64)
    inverted_log_psi = jnp.asarray(-0.5 - 0.3j, dtype=jnp.complex64)

    projected = odd_parity_project_log_amplitude(log_psi, inverted_log_psi)
    swapped = odd_parity_project_log_amplitude(inverted_log_psi, log_psi)

    amplitude_ratio = jnp.exp(swapped - projected)
    np.testing.assert_allclose(
        np.asarray(amplitude_ratio),
        np.asarray(-1.0 + 0.0j),
        rtol=2e-6,
        atol=2e-6,
    )


def test_odd_parity_projection_preserves_nearly_cancelled_component():
    log_psi = jnp.asarray(40.0 + 0.7j, dtype=jnp.complex64)
    inverted_log_psi = jnp.asarray(39.99998 + 0.700003j, dtype=jnp.complex64)

    projected = odd_parity_project_log_amplitude(log_psi, inverted_log_psi)
    scaled_amplitude = jnp.exp(projected - log_psi)
    actual_delta = np.complex128(np.asarray(inverted_log_psi - log_psi))
    expected = -0.5 * np.expm1(actual_delta)

    np.testing.assert_allclose(
        np.asarray(scaled_amplitude),
        expected,
        rtol=2e-5,
        atol=1e-10,
    )


def test_odd_parity_projection_encodes_exact_node_as_zero():
    log_psi = jnp.asarray(2.5 - 0.7j, dtype=jnp.complex64)

    projected = odd_parity_project_log_amplitude(log_psi, log_psi)

    assert np.isneginf(float(jnp.real(projected)))
    assert not np.isnan(float(jnp.imag(projected)))
    np.testing.assert_array_equal(
        np.asarray(jnp.exp(projected)),
        np.asarray(0.0 + 0.0j, dtype=np.complex64),
    )


def test_parity_projection_preserves_encoded_zeros():
    encoded_zero = jnp.asarray(-jnp.inf + 0.0j, dtype=jnp.complex64)

    even = parity_project_log_amplitude(encoded_zero, encoded_zero, 1)
    odd = parity_project_log_amplitude(encoded_zero, encoded_zero, -1)

    for projected in (even, odd):
        assert np.isneginf(float(jnp.real(projected)))
        assert not np.isnan(float(jnp.imag(projected)))
        np.testing.assert_array_equal(
            np.asarray(jnp.exp(projected)),
            np.asarray(0.0 + 0.0j, dtype=np.complex64),
        )


def test_odd_parity_projection_supports_jit_and_vmap():
    log_psi = jnp.asarray(
        [0.1 + 0.2j, -0.4 + 0.6j, 0.8 - 0.3j],
        dtype=jnp.complex64,
    )
    inverted_log_psi = jnp.asarray(
        [-0.6 - 0.1j, 0.2 - 0.5j, -0.3 + 0.4j],
        dtype=jnp.complex64,
    )

    jitted = jax.jit(odd_parity_project_log_amplitude)(
        log_psi,
        inverted_log_psi,
    )
    vmapped = jax.vmap(odd_parity_project_log_amplitude)(
        log_psi,
        inverted_log_psi,
    )
    expected_amplitudes = 0.5 * (jnp.exp(log_psi) - jnp.exp(inverted_log_psi))

    np.testing.assert_allclose(
        np.asarray(jnp.exp(jitted)),
        np.asarray(expected_amplitudes),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.exp(vmapped)),
        np.asarray(expected_amplitudes),
        rtol=2e-6,
        atol=2e-6,
    )


def test_odd_parity_projection_has_finite_derivatives_away_from_node():
    def objective(coordinates):
        x, y = coordinates
        log_psi = 0.3 * x - 0.2 * y**2 + 1j * (0.4 * y + 0.1 * x**2)
        inverted_log_psi = -0.7 + 0.1 * x * y + 1j * (-0.2 * x + 0.3 * y)
        projected = odd_parity_project_log_amplitude(
            log_psi,
            inverted_log_psi,
        )
        return jnp.real(projected) + 0.25 * jnp.imag(projected)

    coordinates = jnp.asarray([0.4, -0.2], dtype=jnp.float32)
    first = jax.grad(objective)(coordinates)
    second = jax.hessian(objective)(coordinates)

    assert np.all(np.isfinite(np.asarray(first)))
    assert np.all(np.isfinite(np.asarray(second)))


def test_even_parity_projection_supports_jit_vmap_grad_and_hessian():
    log_psi = jnp.asarray(
        [0.1 + 0.2j, -0.4 + 0.6j, 0.8 - 0.3j],
        dtype=jnp.complex64,
    )
    inverted_log_psi = jnp.asarray(
        [-0.6 - 0.1j, 0.2 - 0.5j, -0.3 + 0.4j],
        dtype=jnp.complex64,
    )

    def project_even(first, second):
        return parity_project_log_amplitude(first, second, 1)

    jitted = jax.jit(project_even)(log_psi, inverted_log_psi)
    vmapped = jax.vmap(project_even)(log_psi, inverted_log_psi)
    expected_amplitudes = 0.5 * (jnp.exp(log_psi) + jnp.exp(inverted_log_psi))

    np.testing.assert_allclose(
        np.asarray(jnp.exp(jitted)),
        np.asarray(expected_amplitudes),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.exp(vmapped)),
        np.asarray(expected_amplitudes),
        rtol=2e-6,
        atol=2e-6,
    )

    def objective(coordinates):
        x, y = coordinates
        first = 0.3 * x - 0.2 * y**2 + 1j * (0.4 * y + 0.1 * x**2)
        second = -0.7 + 0.1 * x * y + 1j * (-0.2 * x + 0.3 * y)
        projected = project_even(first, second)
        return jnp.real(projected) + 0.25 * jnp.imag(projected)

    coordinates = jnp.asarray([0.4, -0.2], dtype=jnp.float32)
    first = jax.grad(objective)(coordinates)
    second = jax.hessian(objective)(coordinates)

    assert np.all(np.isfinite(np.asarray(first)))
    assert np.all(np.isfinite(np.asarray(second)))


def test_even_parity_near_cancellation_has_finite_grad_and_hessian():
    pi = jnp.asarray(jnp.pi, dtype=jnp.float32)

    def objective(coordinates):
        x, y = coordinates
        common = 30.0 + 0.1 * x + 1j * (0.2 - 0.05 * y)
        magnitude_gap = 0.02 + 0.003 * x**2
        phase_offset = 0.01 * y
        inverted = common - magnitude_gap + 1j * (pi + phase_offset)
        projected = parity_project_log_amplitude(common, inverted, 1)
        return jnp.real(projected) + 0.25 * jnp.imag(projected)

    coordinates = jnp.asarray([0.4, -0.2], dtype=jnp.float32)
    first = jax.grad(objective)(coordinates)
    second = jax.hessian(objective)(coordinates)

    assert np.all(np.isfinite(np.asarray(first)))
    assert np.all(np.isfinite(np.asarray(second)))


def test_parity_residual_is_scale_invariant_and_distinguishes_parity():
    log_psi = jnp.asarray(
        [0.2 + 0.3j, -0.7 - 0.2j, 1.1 + 0.8j],
        dtype=jnp.complex64,
    )
    common_shifts = jnp.asarray([1.0e3, -1.0e3, 500.0], dtype=jnp.float32)
    even_inverted = log_psi
    odd_inverted = log_psi + jnp.asarray(1j * jnp.pi, dtype=jnp.complex64)

    even_residual = parity_log_amplitude_residual(
        log_psi + common_shifts,
        even_inverted + common_shifts,
        1,
    )
    even_wrong_residual = parity_log_amplitude_residual(
        log_psi + common_shifts,
        even_inverted + common_shifts,
        -1,
    )
    odd_residual = parity_log_amplitude_residual(
        log_psi + common_shifts,
        odd_inverted + common_shifts,
        -1,
    )
    odd_wrong_residual = parity_log_amplitude_residual(
        log_psi + common_shifts,
        odd_inverted + common_shifts,
        1,
    )

    np.testing.assert_allclose(np.asarray(even_residual), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(even_wrong_residual), 2.0, rtol=2e-6)
    np.testing.assert_allclose(np.asarray(odd_residual), 0.0, atol=2e-11)
    np.testing.assert_allclose(np.asarray(odd_wrong_residual), 2.0, rtol=2e-6)


def test_parity_loss_handles_batches_encoded_zeros_and_jit():
    encoded_zero = jnp.asarray(-jnp.inf + 0.0j, dtype=jnp.complex64)
    log_psi = jnp.asarray(
        [0.2 + 0.3j, encoded_zero, -800.0 - 0.4j],
        dtype=jnp.complex64,
    )
    inverted_log_psi = log_psi

    residual = jax.jit(
        lambda first, second: parity_log_amplitude_residual(first, second, 1)
    )(log_psi, inverted_log_psi)
    loss = jax.jit(lambda first, second: parity_log_amplitude_loss(first, second, 1))(
        log_psi, inverted_log_psi
    )

    np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-7)


def test_parity_helpers_reject_invalid_parity_and_shape():
    log_psi = jnp.asarray([0.2 + 0.3j, -0.1 + 0.4j])

    with np.testing.assert_raises_regex(ValueError, "parity must be"):
        parity_project_log_amplitude(log_psi, log_psi, 0)
    with np.testing.assert_raises_regex(ValueError, "identical shapes"):
        parity_log_amplitude_residual(log_psi, log_psi[:1], 1)


def test_source_aligned_vector_logpsi_is_source_dominated_at_initialization():
    raw_logpsi = jnp.asarray(
        [0.2 + 0.3j, -0.4 - 0.1j, 0.1 + 0.7j],
        dtype=jnp.complex64,
    )
    ground_logpsi = jnp.asarray(-0.6 + 0.4j, dtype=jnp.complex64)
    dipole = jnp.asarray([0.5, -0.3, 0.8], dtype=jnp.float32)
    source_center = jnp.asarray([0.1, 0.05, -0.2], dtype=jnp.float32)
    coefficient = jnp.asarray(1.2 - 0.35j, dtype=jnp.complex64)
    residual_log_scale = jnp.asarray(-18.0, dtype=jnp.float32)

    result = source_aligned_vector_logpsi(
        raw_logpsi,
        ground_logpsi,
        dipole,
        source_center,
        coefficient,
        residual_log_scale,
    )
    source = coefficient * (dipole - source_center) * jnp.exp(ground_logpsi)
    expected = source + jnp.exp(raw_logpsi + residual_log_scale)

    assert result.shape == (3,)
    np.testing.assert_allclose(
        np.asarray(jnp.exp(result)),
        np.asarray(expected),
        rtol=3e-6,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.exp(result)),
        np.asarray(source),
        rtol=3e-6,
        atol=3e-7,
    )


def test_source_aligned_vector_logpsi_exact_zero_is_finite():
    result = source_aligned_vector_logpsi(
        jnp.full((3,), -jnp.inf + 0.0j, dtype=jnp.complex64),
        jnp.asarray(0.2 + 0.4j, dtype=jnp.complex64),
        jnp.asarray([0.2, -0.1, 0.7], dtype=jnp.float32),
        jnp.asarray([0.2, -0.1, 0.7], dtype=jnp.float32),
        jnp.asarray(1.0 + 0.5j, dtype=jnp.complex64),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert result.shape == (3,)
    assert np.all(np.isfinite(np.asarray(result)))


def test_source_aligned_vector_logpsi_keeps_source_hessian_at_source_zero():
    """An exact dipole zero must not detach the source term from coordinate AD."""

    def log_amplitude(coordinate):
        result = source_aligned_vector_logpsi(
            jnp.zeros(3, dtype=jnp.complex64),
            jnp.asarray(2.0 + 0.0j, dtype=jnp.complex64),
            jnp.asarray([coordinate, 0.3, -0.4], dtype=jnp.float32),
            jnp.zeros(3, dtype=jnp.float32),
            jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return jnp.real(result[0])

    coordinate = jnp.asarray(0.0, dtype=jnp.float32)
    first = jax.grad(log_amplitude)(coordinate)
    second = jax.grad(jax.grad(log_amplitude))(coordinate)

    np.testing.assert_allclose(float(first), np.exp(2.0), rtol=2e-6)
    np.testing.assert_allclose(float(second), -np.exp(4.0), rtol=2e-6)


def test_source_aligned_vector_response_has_finite_derivatives():
    batch = _h_batch()
    data = MoleculeData(
        electrons=jnp.asarray([[0.2, -0.1, 0.4]], dtype=jnp.float32),
        atoms=batch.data.atoms,
        charges=batch.data.charges,
    )
    response = MolecularVectorResponseFermiNet(
        nspins=(1, 0),
        ndets=1,
        hidden_dims_single=(2,),
        hidden_dims_double=(1,),
    )
    params = response.init(jax.random.PRNGKey(3), data)
    direction = jax.tree.map(
        lambda leaf: jnp.ones_like(leaf) / max(1, leaf.size),
        params,
    )
    flat_electrons = jnp.ravel(data.electrons)

    def scalar_response(parameter_step, electrons_flat):
        local_params = jax.tree.map(
            lambda leaf, tangent: leaf + parameter_step * tangent,
            params,
            direction,
        )
        electrons = jnp.reshape(electrons_flat, data.electrons.shape)
        local_data = data.merge({"electrons": electrons})
        raw_logpsi = response.apply(local_params, local_data)
        ground_logpsi = -0.4 * jnp.sum(electrons**2) + 0.1j * jnp.sum(electrons)
        dipole = -jnp.sum(electrons, axis=0)
        aligned = source_aligned_vector_logpsi(
            raw_logpsi,
            ground_logpsi,
            dipole,
            jnp.zeros(3, dtype=electrons.dtype),
            jnp.asarray(1.0 + 0.2j, dtype=jnp.complex64),
            jnp.asarray(-8.0, dtype=electrons.dtype),
        )
        return jnp.sum(jnp.real(aligned)) + 0.2 * jnp.sum(jnp.imag(aligned))

    def parameter_function(step):
        return scalar_response(step, flat_electrons)

    def coordinate_function(electrons):
        return scalar_response(0.0, electrons)

    parameter_first = jax.grad(parameter_function)(jnp.asarray(0.0))
    parameter_second = jax.grad(jax.grad(parameter_function))(jnp.asarray(0.0))
    coordinate_first = jax.grad(coordinate_function)(flat_electrons)
    coordinate_second = jax.hessian(coordinate_function)(flat_electrons)

    assert np.isfinite(float(parameter_first))
    assert np.isfinite(float(parameter_second))
    assert np.all(np.isfinite(np.asarray(coordinate_first)))
    assert np.all(np.isfinite(np.asarray(coordinate_second)))


def test_restore_params_from_stage_checkpoint(tmp_path):
    fallback = {"params": {"w": jnp.asarray([0.0, 0.0])}}
    restored = {"params": {"w": jnp.asarray([1.0, 2.0])}}
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    manager.save(7, {"params": restored})

    step, params = restore_params_from_checkpoint(tmp_path, fallback)

    assert step == 7
    np.testing.assert_allclose(np.asarray(params["params"]["w"]), [1.0, 2.0])
