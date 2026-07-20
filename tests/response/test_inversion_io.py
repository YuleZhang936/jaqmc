# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from jaqmc.response.inversion import forward_lit, lit_block_statistics
from jaqmc.response.inversion_io import (
    LITInversionSettings,
    aggregate_lit_npz,
    invert_lit_npz,
    lit_inversion_npz_payload,
)


def _write_workflow_npz(
    path: Path,
    *,
    omega: np.ndarray,
    eta: float,
    signed_lit: np.ndarray,
    blocks: np.ndarray | None = None,
    raw_blocks: np.ndarray | None = None,
    digests: tuple[str, ...] | str | None = ("pool-x", "pool-z"),
    covariance: np.ndarray | None = None,
    systematic_error: np.ndarray | None = None,
    error_d_valid: np.ndarray | None = None,
    axes: str = "xz",
    axis_indices: np.ndarray = np.asarray([0, 2]),
) -> None:
    payload: dict[str, object] = {
        "omega": omega,
        "eta": eta,
        "signed_lit": signed_lit,
        "axes": axes,
        "axis_indices": axis_indices,
        "error_bound_monitor": (
            np.zeros_like(signed_lit) if systematic_error is None else systematic_error
        ),
        "error_d_valid": (
            np.ones_like(signed_lit, dtype=np.bool_)
            if error_d_valid is None
            else error_d_valid
        ),
    }
    if blocks is not None:
        payload["signed_lit_jackknife_blocks"] = blocks
        payload["signed_lit_jackknife_block_count"] = blocks.shape[-1]
    if raw_blocks is not None:
        payload["signed_lit_blocks"] = raw_blocks
    if digests is not None:
        payload["eval_pool_sha256"] = np.asarray(digests)
    if covariance is not None:
        payload["signed_lit_covariance"] = covariance
    np.savez(path, **payload)


def test_aggregate_matched_blocks_preserves_cross_eta_covariance(tmp_path: Path):
    omega_a = np.asarray([0.1, 0.2])
    omega_b = np.asarray([0.15, 0.25, 0.35])
    full_a = np.asarray([[10.0, 11.0], [20.0, 21.0]])
    full_b = np.asarray([[12.0, 13.0, 14.0], [22.0, 23.0, 24.0]])
    blocks_a = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5) / 10.0
    blocks_b = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5) / 7.0 + 2.0
    systematic_a = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    systematic_b = np.asarray([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    path_a = tmp_path / "eta-a.npz"
    path_b = tmp_path / "eta-b.npz"
    _write_workflow_npz(
        path_a,
        omega=omega_a,
        eta=0.03,
        signed_lit=full_a,
        blocks=blocks_a,
        systematic_error=systematic_a,
    )
    _write_workflow_npz(
        path_b,
        omega=omega_b,
        eta=0.08,
        signed_lit=full_b,
        blocks=blocks_b,
        systematic_error=systematic_b,
    )

    result = aggregate_lit_npz([path_a, path_b])

    combined_blocks = np.concatenate([blocks_a, blocks_b], axis=1)
    expected_statistical = lit_block_statistics(combined_blocks).covariance
    expected_systematic = np.concatenate([systematic_a, systematic_b], axis=1)
    expected_covariance = np.array(expected_statistical, copy=True)
    diagonal = np.arange(expected_covariance.shape[-1])
    expected_covariance[:, diagonal, diagonal] += expected_systematic**2
    np.testing.assert_array_equal(result.omega, np.concatenate([omega_a, omega_b]))
    np.testing.assert_array_equal(
        result.eta,
        np.asarray([0.03, 0.03, 0.08, 0.08, 0.08]),
    )
    # The formal central value is the full-pool ratio-of-sums, not the mean of
    # jackknife pseudo-values.
    np.testing.assert_array_equal(
        result.signed_lit,
        np.concatenate([full_a, full_b], axis=1),
    )
    assert not np.allclose(result.signed_lit, np.mean(combined_blocks, axis=-1))
    np.testing.assert_array_equal(result.block_estimates, combined_blocks)
    np.testing.assert_allclose(result.statistical_covariance, expected_statistical)
    np.testing.assert_array_equal(result.systematic_error, expected_systematic)
    np.testing.assert_allclose(result.covariance, expected_covariance)
    assert np.any(np.abs(result.covariance[:, :2, 2:]) > 0.0)
    assert result.axes == "xz"
    np.testing.assert_array_equal(result.axis_indices, [0, 2])
    assert result.covariance_mode == "matched_blocks"
    assert result.metadata[0].observation_start == 0
    assert result.metadata[0].observation_stop == 2
    assert result.metadata[1].observation_start == 2
    assert result.metadata[1].observation_stop == 5
    assert result.metadata[1].eta_values == (0.08,)


def test_legacy_raw_blocks_are_rejected_even_with_jackknife(tmp_path: Path):
    path = tmp_path / "both-fields.npz"
    omega = np.asarray([0.1, 0.2])
    signed_lit = np.ones((2, 2))
    jackknife = np.arange(16, dtype=float).reshape(2, 2, 4)
    raw = jackknife + 1000.0
    _write_workflow_npz(
        path,
        omega=omega,
        eta=0.04,
        signed_lit=signed_lit,
        blocks=jackknife,
        raw_blocks=raw,
        digests="common-pool",
    )

    with pytest.raises(ValueError, match="unsupported legacy raw-block fields"):
        aggregate_lit_npz(path)


def test_mismatched_pools_fail_unless_independence_is_explicit(tmp_path: Path):
    omega = np.asarray([0.1, 0.2])
    signed_lit = np.ones((2, 2))
    blocks = np.arange(16, dtype=float).reshape(2, 2, 4)
    covariance_a = np.stack([np.eye(2), 2.0 * np.eye(2)])
    covariance_b = np.stack([3.0 * np.eye(2), 4.0 * np.eye(2)])
    path_a = tmp_path / "pool-a.npz"
    path_b = tmp_path / "pool-b.npz"
    _write_workflow_npz(
        path_a,
        omega=omega,
        eta=0.03,
        signed_lit=signed_lit,
        blocks=blocks,
        digests=("pool-a-x", "pool-a-z"),
        covariance=covariance_a,
    )
    _write_workflow_npz(
        path_b,
        omega=omega,
        eta=0.08,
        signed_lit=2.0 * signed_lit,
        blocks=blocks + 1.0,
        digests=("pool-b-x", "pool-b-z"),
        covariance=covariance_b,
    )

    with pytest.raises(ValueError, match="eval_pool_sha256 differs"):
        aggregate_lit_npz([path_a, path_b])

    result = aggregate_lit_npz(
        [path_a, path_b],
        assume_independent=True,
    )

    expected = np.zeros((2, 4, 4))
    expected[:, :2, :2] = covariance_a
    expected[:, 2:, 2:] = covariance_b
    np.testing.assert_array_equal(result.covariance, expected)
    np.testing.assert_array_equal(result.statistical_covariance, expected)
    assert result.block_estimates is None
    assert result.covariance_mode == "independent_files"


def test_independence_fallback_requires_stored_covariance_in_every_file(
    tmp_path: Path,
):
    omega = np.asarray([0.1, 0.2])
    signed_lit = np.ones((2, 2))
    path_a = tmp_path / "has-cov.npz"
    path_b = tmp_path / "no-cov.npz"
    _write_workflow_npz(
        path_a,
        omega=omega,
        eta=0.03,
        signed_lit=signed_lit,
        digests=None,
        covariance=np.stack([np.eye(2), np.eye(2)]),
    )
    _write_workflow_npz(
        path_b,
        omega=omega,
        eta=0.08,
        signed_lit=signed_lit,
        digests=None,
    )

    with pytest.raises(ValueError, match="requires signed_lit_covariance"):
        aggregate_lit_npz([path_a, path_b], assume_independent=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"axes": "xy"}, "inconsistent with axis_indices"),
        ({"axis_indices": np.asarray([0, 0])}, "duplicate axes"),
        ({"eta": -0.03}, "finite, positive"),
        ({"signed_lit": np.ones((1, 2))}, "must have shape"),
    ],
)
def test_workflow_metadata_and_shapes_are_validated(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
):
    path = tmp_path / "invalid.npz"
    payload: dict[str, object] = {
        "omega": np.asarray([0.1, 0.2]),
        "eta": 0.03,
        "signed_lit": np.ones((2, 2)),
        "axes": "xz",
        "axis_indices": np.asarray([0, 2]),
        "signed_lit_jackknife_blocks": np.ones((2, 2, 4)),
        "eval_pool_sha256": np.asarray(["pool-x", "pool-z"]),
        "error_bound_monitor": np.zeros((2, 2)),
        "error_d_valid": np.ones((2, 2), dtype=np.bool_),
    }
    payload.update(override)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=message):
        aggregate_lit_npz(path)


def test_files_with_different_axes_cannot_be_combined(tmp_path: Path):
    omega = np.asarray([0.1, 0.2])
    blocks = np.arange(16, dtype=float).reshape(2, 2, 4)
    path_a = tmp_path / "xz.npz"
    path_b = tmp_path / "yz.npz"
    _write_workflow_npz(
        path_a,
        omega=omega,
        eta=0.03,
        signed_lit=np.ones((2, 2)),
        blocks=blocks,
    )
    _write_workflow_npz(
        path_b,
        omega=omega,
        eta=0.08,
        signed_lit=np.ones((2, 2)),
        blocks=blocks,
        axes="yz",
        axis_indices=np.asarray([1, 2]),
    )

    with pytest.raises(ValueError, match="incompatible response axes"):
        aggregate_lit_npz([path_a, path_b])


@pytest.mark.parametrize(
    ("systematic_error", "error_d_valid", "message"),
    [
        (np.asarray([[0.1, np.nan], [0.2, 0.3]]), None, "finite, nonnegative"),
        (np.asarray([[0.1, -0.2], [0.2, 0.3]]), None, "finite, nonnegative"),
        (
            None,
            np.asarray([[True, False], [True, True]]),
            "invalid fidelity/D error bounds",
        ),
    ],
)
def test_formal_loader_rejects_invalid_systematic_error(
    tmp_path: Path,
    systematic_error: np.ndarray | None,
    error_d_valid: np.ndarray | None,
    message: str,
):
    path = tmp_path / "invalid-systematic.npz"
    _write_workflow_npz(
        path,
        omega=np.asarray([0.1, 0.2]),
        eta=0.03,
        signed_lit=np.ones((2, 2)),
        blocks=np.ones((2, 2, 4)),
        systematic_error=systematic_error,
        error_d_valid=error_d_valid,
    )

    with pytest.raises(ValueError, match=message):
        aggregate_lit_npz(path)


def test_tiny_relative_covariance_asymmetry_is_rejected_consistently(tmp_path: Path):
    path = tmp_path / "asymmetric-covariance.npz"
    covariance = np.stack([np.eye(2), np.eye(2)]).astype(np.float64) * 1e-12
    covariance[0, 0, 1] = 1e-15
    _write_workflow_npz(
        path,
        omega=np.asarray([0.1, 0.2]),
        eta=0.03,
        signed_lit=np.ones((2, 2)),
        blocks=np.ones((2, 2, 4)),
        covariance=covariance,
    )

    with pytest.raises(ValueError, match="must be symmetric"):
        aggregate_lit_npz(path)


def test_formal_npz_inversion_propagates_jackknife_and_is_pickle_free(
    tmp_path: Path,
):
    omega = np.linspace(0.75, 0.81, 41)
    exact_energy = 0.78
    signed_lit = forward_lit(
        omega,
        0.003,
        pole_energies=np.asarray([exact_energy]),
        pole_strengths=np.asarray([0.18]),
    )[np.newaxis, :]
    block_count = 8
    phases = np.arange(block_count, dtype=np.float64)[np.newaxis, np.newaxis, :]
    positions = np.linspace(0.0, 2.0 * np.pi, omega.size)[np.newaxis, :, np.newaxis]
    noise = 0.2 * np.sin(positions + phases)
    noise -= np.mean(noise, axis=-1, keepdims=True)
    blocks = signed_lit[..., np.newaxis] + noise
    path = tmp_path / "lit_spectrum.npz"
    _write_workflow_npz(
        path,
        omega=omega,
        eta=0.003,
        signed_lit=signed_lit,
        blocks=blocks,
        digests=("pool-x",),
        systematic_error=np.full_like(signed_lit, 2.0),
        axes="x",
        axis_indices=np.asarray([0]),
    )
    settings = LITInversionSettings(
        threshold=0.9,
        pole_energies=(0.779,),
        fit_pole_energies=True,
        pole_energy_bounds=((0.77, 0.79),),
        pole_fit_tolerance=1e-10,
        pole_fit_max_iterations=1000,
    )

    inversion = invert_lit_npz(path, settings)

    assert inversion.result.pole_energies[0] == pytest.approx(exact_energy, abs=1e-8)
    assert inversion.jackknife is not None
    assert inversion.jackknife.block_count == block_count
    assert inversion.jackknife.leave_one_out_pole_energies.shape == (block_count, 1)
    assert inversion.jackknife.leave_one_out_pole_strengths.shape == (
        block_count,
        1,
        1,
    )
    assert np.all(np.isfinite(inversion.jackknife.pole_energy_standard_error))

    output = tmp_path / "lit_inversion.npz"
    np.savez_compressed(output, **lit_inversion_npz_payload(inversion))
    with np.load(output, allow_pickle=False) as archive:
        assert bool(archive["solver_success"][0])
        assert bool(archive["jackknife_available"])
        assert int(archive["jackknife_block_count"]) == block_count
        assert archive["pole_energies"][0] == pytest.approx(exact_energy, abs=1e-8)
        assert archive["source_lit_paths"].tolist() == [str(path)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0.9}, "at least one discrete pole"),
        (
            {
                "threshold": 0.9,
                "pole_energies": (0.78,),
                "pole_energy_bounds": ((0.77, 0.79),),
            },
            "require fit_pole_energies",
        ),
        (
            {
                "threshold": 0.9,
                "pole_energies": (0.78,),
                "continuum_regularization": -1.0,
            },
            "finite and nonnegative",
        ),
    ],
)
def test_formal_inversion_settings_fail_before_a_scan(
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        LITInversionSettings(**kwargs)
