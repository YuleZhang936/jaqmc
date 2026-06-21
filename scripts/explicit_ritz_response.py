#!/usr/bin/env python
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Run source-adapted explicit Ritz carrier BF-NKSR for H/He atom tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jaqmc.response.explicit_ritz import (
    helium_singlet_p_carrier_specs,
    hydrogen_p_carrier_specs,
    run_explicit_ritz,
    scalar_rayleigh_diagnostics,
)
from jaqmc.response.ferminet_bfnksr import (
    final_replica_pole_diagnostics,
    load_ferminet_ground,
)
from jaqmc.response.spectrum import find_spectrum_peaks, lorentzian_spectrum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-adapted explicitly correlated Ritz carrier response"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ground-energy", type=float, default=np.nan)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--overlap-cutoff", type=float, default=1e-8)
    parser.add_argument("--fixed-whitening", action="store_true")
    parser.add_argument("--block-stable-metric", action="store_true")
    parser.add_argument("--block-mode-min", type=float, default=0.0)
    parser.add_argument("--bright-influence-sampling", action="store_true")
    parser.add_argument("--bright-influence-pilot-samples", type=int, default=0)
    parser.add_argument("--bright-influence-candidate-factor", type=int, default=2)
    parser.add_argument("--bright-influence-max-candidates", type=int, default=32768)
    parser.add_argument("--bright-influence-max-states", type=int, default=1)
    parser.add_argument("--bright-influence-gradient-weight", type=float, default=1.0)
    parser.add_argument("--bright-influence-potential-weight", type=float, default=1.0)
    parser.add_argument("--bright-influence-source-weight", type=float, default=1.0)
    parser.add_argument("--bright-influence-winsor-quantile", type=float, default=0.995)
    parser.add_argument("--bright-influence-floor-fraction", type=float, default=1e-8)
    parser.add_argument(
        "--bright-influence-direct-leverage-component",
        action="store_true",
    )
    parser.add_argument(
        "--bright-influence-pair-resampling",
        action="store_true",
    )
    parser.add_argument("--bright-min-weight", type=float, default=0.05)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--scalar-certification", action="store_true")
    parser.add_argument("--cert-samples", type=int, default=0)
    parser.add_argument("--cert-blocks", type=int, default=0)
    parser.add_argument("--cert-seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.004)
    parser.add_argument("--omega-min", type=float, default=0.2)
    parser.add_argument("--omega-max", type=float, default=1.5)
    parser.add_argument("--grid-size", type=int, default=4001)
    parser.add_argument(
        "--p-decays",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--laguerre-orders", type=int, nargs="+", default=[0])
    parser.add_argument("--s-decays", type=float, nargs="+", default=[1.6875])
    parser.add_argument("--s-laguerre-orders", type=int, nargs="+", default=[0])
    parser.add_argument("--d-decays", type=float, nargs="+", default=[0.5, 1.0, 1.8])
    parser.add_argument("--d-laguerre-orders", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--geminal-gammas",
        type=float,
        nargs="*",
        default=[0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--no-f12", action="store_true")
    parser.add_argument("--include-exp-geminals", action="store_true")
    parser.add_argument("--include-r12-geminal", action="store_true")
    parser.add_argument("--include-pd", action="store_true")
    parser.add_argument("--core-decay", type=float, default=2.0)
    parser.add_argument("--diffuse-decay", type=float, default=0.35)
    parser.add_argument("--max-roots", type=int, default=5)
    return parser.parse_args()


def _target_energy_ha(electron_count: int) -> float:
    if electron_count == 1:
        return 0.375
    if electron_count == 2:
        return 0.780
    return float("nan")


def _bright_roots(
    poles: np.ndarray,
    weights: np.ndarray,
    *,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    poles = np.asarray(poles, dtype=np.float64)
    weights = np.asarray(weights.real, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    max_weight = max(float(np.max(weights)), 1e-300)
    relative = weights / max_weight
    keep = (poles > 0.0) & (relative >= float(min_weight))
    return poles[keep], relative[keep]


def _bright_indices(
    poles: np.ndarray,
    weights: np.ndarray,
    *,
    min_weight: float,
) -> np.ndarray:
    poles_np = np.asarray(poles, dtype=np.float64)
    weights_np = np.asarray(weights.real, dtype=np.float64)
    weights_np = np.where(
        np.isfinite(weights_np) & (weights_np > 0.0),
        weights_np,
        0.0,
    )
    max_weight = max(float(np.max(weights_np)), 1e-300)
    relative = weights_np / max_weight
    return np.flatnonzero((poles_np > 0.0) & (relative >= float(min_weight)))


def main() -> None:  # noqa: C901
    args = _parse_args()
    if not np.isfinite(args.ground_energy):
        msg = "--ground-energy is required for weak-form explicit Ritz matrices"
        raise ValueError(msg)
    ground = load_ferminet_ground(args.checkpoint, ground_energy=args.ground_energy)
    electron_count = ground.electron_shape[0]
    if electron_count == 1:
        specs = hydrogen_p_carrier_specs(
            args.p_decays,
            laguerre_orders=args.laguerre_orders,
        )
    elif electron_count == 2:
        specs = helium_singlet_p_carrier_specs(
            args.p_decays,
            s_decays=args.s_decays,
            s_laguerre_orders=args.s_laguerre_orders,
            d_decays=args.d_decays,
            d_laguerre_orders=args.d_laguerre_orders,
            laguerre_orders=args.laguerre_orders,
            geminal_gammas=args.geminal_gammas,
            include_f12=not args.no_f12,
            include_exp_geminals=args.include_exp_geminals,
            include_r12_geminal=args.include_r12_geminal,
            include_pd=args.include_pd,
        )
    else:
        msg = "explicit Ritz atom smoke currently supports H/He only"
        raise ValueError(msg)
    result = run_explicit_ritz(
        ground,
        specs=specs,
        n_samples=args.samples,
        n_blocks=args.blocks,
        core_decay=args.core_decay,
        diffuse_decay=args.diffuse_decay,
        seed=args.seed,
        batch_size=args.batch_size,
        overlap_cutoff=args.overlap_cutoff,
        fixed_whitening=args.fixed_whitening,
        block_stable_metric=args.block_stable_metric,
        block_mode_min=args.block_mode_min,
        bright_influence_sampling=args.bright_influence_sampling,
        bright_influence_pilot_samples=args.bright_influence_pilot_samples,
        bright_influence_candidate_factor=args.bright_influence_candidate_factor,
        bright_influence_max_candidates=args.bright_influence_max_candidates,
        bright_influence_min_weight=args.bright_min_weight,
        bright_influence_max_states=args.bright_influence_max_states,
        bright_influence_gradient_weight=args.bright_influence_gradient_weight,
        bright_influence_potential_weight=args.bright_influence_potential_weight,
        bright_influence_source_weight=args.bright_influence_source_weight,
        bright_influence_winsor_quantile=args.bright_influence_winsor_quantile,
        bright_influence_floor_fraction=args.bright_influence_floor_fraction,
        bright_influence_direct_leverage_component=(
            args.bright_influence_direct_leverage_component
        ),
        bright_influence_pair_resampling=args.bright_influence_pair_resampling,
    )
    final_replica = final_replica_pole_diagnostics(
        result.block_overlaps,
        result.block_hamiltonians,
        result.block_sources,
        result.block_counts,
        retained_heads=len(specs),
        source_in_basis=False,
        overlap_cutoff=args.overlap_cutoff,
        root_floor=0.0,
        min_weight=args.bright_min_weight,
        max_roots=args.max_roots,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed + 1_000_003,
    )
    omega = np.linspace(args.omega_min, args.omega_max, args.grid_size)
    weights = result.spectrum.weights[:, 0, 0]
    bright_roots, bright_weights = _bright_roots(
        result.spectrum.excitation_energies,
        weights,
        min_weight=args.bright_min_weight,
    )
    intensity = lorentzian_spectrum(
        omega,
        result.spectrum.excitation_energies,
        weights,
        args.eta,
    )
    peaks = find_spectrum_peaks(
        omega,
        intensity,
        min_height_fraction=0.01,
        max_peaks=args.max_roots,
    )
    target = _target_energy_ha(electron_count)
    raw_first_pole = (
        float(result.spectrum.excitation_energies[0])
        if result.spectrum.excitation_energies.size
        else float("nan")
    )
    first_bright_pole = float(bright_roots[0]) if bright_roots.size else float("nan")
    scalar_fit = None
    scalar_cert = None
    scalar_bright_index = -1
    scalar_coefficients = np.asarray([], dtype=np.float64)
    if args.scalar_certification:
        bright_indices = _bright_indices(
            result.spectrum.excitation_energies,
            weights,
            min_weight=args.bright_min_weight,
        )
        if bright_indices.size == 0:
            msg = "scalar certification found no positive bright Ritz root"
            raise ValueError(msg)
        if result.ritz_carrier_coefficients is None:
            msg = "scalar certification requires Ritz carrier coefficients"
            raise ValueError(msg)
        if result.raw_block_overlaps is None or result.raw_block_hamiltonians is None:
            msg = "scalar certification requires raw block matrices"
            raise ValueError(msg)
        scalar_bright_index = int(bright_indices[0])
        scalar_coefficients = np.asarray(
            result.ritz_carrier_coefficients[:, scalar_bright_index],
            dtype=np.float64,
        )
        scalar_fit = scalar_rayleigh_diagnostics(
            result.raw_block_overlaps,
            result.raw_block_hamiltonians,
            result.block_counts,
            scalar_coefficients,
            full_overlap=result.raw_overlap,
            full_hamiltonian=result.raw_hamiltonian,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.seed + 2_000_003,
        )
        if int(args.cert_samples) > 0:
            cert_result = run_explicit_ritz(
                ground,
                specs=specs,
                n_samples=int(args.cert_samples),
                n_blocks=(
                    int(args.cert_blocks) if int(args.cert_blocks) > 0 else args.blocks
                ),
                core_decay=args.core_decay,
                diffuse_decay=args.diffuse_decay,
                seed=int(args.cert_seed) or args.seed + 3_000_003,
                batch_size=args.batch_size,
                overlap_cutoff=args.overlap_cutoff,
                fixed_whitening=False,
                block_stable_metric=False,
                block_mode_min=0.0,
            )
            if (
                cert_result.raw_block_overlaps is None
                or cert_result.raw_block_hamiltonians is None
            ):
                msg = "independent scalar certification produced no raw matrices"
                raise ValueError(msg)
            scalar_cert = scalar_rayleigh_diagnostics(
                cert_result.raw_block_overlaps,
                cert_result.raw_block_hamiltonians,
                cert_result.block_counts,
                scalar_coefficients,
                full_overlap=cert_result.raw_overlap,
                full_hamiltonian=cert_result.raw_hamiltonian,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.seed + 4_000_003,
            )
    print(
        "explicit_ritz_response "
        f"checkpoint={args.checkpoint} "
        f"electron_count={electron_count} "
        f"carriers={len(specs)} "
        f"retained={result.overlap.shape[0]} "
        f"fixed_whitening={args.fixed_whitening} "
        f"bright_influence_sampling={args.bright_influence_sampling} "
        f"samples={args.samples} blocks={args.blocks} "
        f"raw_first_pole_ha={raw_first_pole:.10f} "
        f"first_bright_pole_ha={first_bright_pole:.10f} "
        f"target_ha={target:.10f} "
        f"bright_error_ha={first_bright_pole - target:.3e} "
        f"bootstrap_mean_ha={float(final_replica['bootstrap_mean']):.10f} "
        f"bootstrap_se_ha={float(final_replica['bootstrap_se']):.3e}"
    )
    if result.sampling_stats:
        stats = result.sampling_stats
        print(
            "sampling "
            f"sampler={stats.get('sampler', 'unknown')} "
            f"proposal_samples={stats.get('proposal_samples', 0)} "
            "proposal_ess_fraction="
            f"{float(stats.get('proposal_ess_fraction', np.nan)):.6f} "
            "unique_fraction="
            f"{float(stats.get('resampling_unique_fraction', np.nan)):.6f} "
            f"pair_resampling={bool(stats.get('pair_resampling', False))} "
            f"ground_weight={float(stats.get('ground_weight', np.nan)):.6f} "
            f"source_weight={float(stats.get('source_component_weight', np.nan)):.6f} "
            f"bright_weight={float(stats.get('bright_weight', np.nan)):.6f} "
            f"leverage_weight={float(stats.get('leverage_weight', np.nan)):.6f} "
            f"aux_weight={float(stats.get('aux_weight', np.nan)):.6f}"
        )
    if scalar_fit is not None:
        print(
            "scalar_fit_certification "
            f"root_index={scalar_bright_index} "
            f"full_ha={float(scalar_fit['full']):.10f} "
            f"bootstrap_mean_ha={float(scalar_fit['bootstrap_mean']):.10f} "
            f"bootstrap_std_ha={float(scalar_fit['bootstrap_std']):.3e} "
            f"loo_jackknife_se_ha="
            f"{float(scalar_fit['loo_jackknife_se']):.3e}"
        )
    if scalar_cert is not None:
        print(
            "scalar_independent_certification "
            f"samples={int(args.cert_samples)} "
            f"full_ha={float(scalar_cert['full']):.10f} "
            f"bootstrap_mean_ha={float(scalar_cert['bootstrap_mean']):.10f} "
            f"bootstrap_std_ha={float(scalar_cert['bootstrap_std']):.3e} "
            f"loo_jackknife_se_ha="
            f"{float(scalar_cert['loo_jackknife_se']):.3e}"
        )
    print("projected poles and weights")
    for idx, (pole, weight) in enumerate(
        zip(
            result.spectrum.excitation_energies,
            result.spectrum.weights[:, 0, 0],
            strict=False,
        )
    ):
        print(f"root={idx:02d} pole_ha={pole:.10f} weight={float(weight.real):.10e}")
    print("peaks read from broadened explicit spectrum")
    for peak in peaks:
        print(f"peak_ha={peak.energy:.10f} intensity={peak.intensity:.10e}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampling_payload = {}
    for key, value in (result.sampling_stats or {}).items():
        sampling_payload[f"explicit_ritz_sampling_{key}"] = np.asarray(value)
    scalar_payload = {}
    if scalar_fit is not None:
        scalar_payload.update(
            {
                "scalar_fit_root_index": scalar_bright_index,
                "scalar_fit_coefficients": scalar_coefficients,
                "scalar_fit_full": scalar_fit["full"],
                "scalar_fit_block_values": scalar_fit["block_values"],
                "scalar_fit_loo_values": scalar_fit["loo_values"],
                "scalar_fit_loo_mean": scalar_fit["loo_mean"],
                "scalar_fit_loo_jackknife_se": scalar_fit["loo_jackknife_se"],
                "scalar_fit_bootstrap_values": scalar_fit["bootstrap_values"],
                "scalar_fit_bootstrap_mean": scalar_fit["bootstrap_mean"],
                "scalar_fit_bootstrap_std": scalar_fit["bootstrap_std"],
                "scalar_fit_bootstrap_min": scalar_fit["bootstrap_min"],
                "scalar_fit_bootstrap_max": scalar_fit["bootstrap_max"],
            }
        )
    if scalar_cert is not None:
        scalar_payload.update(
            {
                "scalar_cert_samples": int(args.cert_samples),
                "scalar_cert_full": scalar_cert["full"],
                "scalar_cert_block_values": scalar_cert["block_values"],
                "scalar_cert_loo_values": scalar_cert["loo_values"],
                "scalar_cert_loo_mean": scalar_cert["loo_mean"],
                "scalar_cert_loo_jackknife_se": scalar_cert["loo_jackknife_se"],
                "scalar_cert_bootstrap_values": scalar_cert["bootstrap_values"],
                "scalar_cert_bootstrap_mean": scalar_cert["bootstrap_mean"],
                "scalar_cert_bootstrap_std": scalar_cert["bootstrap_std"],
                "scalar_cert_bootstrap_min": scalar_cert["bootstrap_min"],
                "scalar_cert_bootstrap_max": scalar_cert["bootstrap_max"],
            }
        )
    np.savez(
        args.output,
        ground_energy=ground.energy,
        poles=result.spectrum.excitation_energies,
        weights=result.spectrum.weights,
        overlap=result.overlap,
        hamiltonian=result.hamiltonian,
        source=result.source,
        samples=result.samples,
        sample_density=result.density,
        carrier_labels=np.asarray(result.carrier_labels),
        explicit_ritz_carrier_count=len(specs),
        explicit_ritz_retained_count=result.overlap.shape[0],
        explicit_ritz_fixed_whitening=args.fixed_whitening,
        explicit_ritz_block_stable_metric=args.block_stable_metric,
        explicit_ritz_block_mode_min=args.block_mode_min,
        explicit_ritz_bright_influence_sampling=args.bright_influence_sampling,
        explicit_ritz_whitening_eigenvalues=(
            np.asarray([])
            if result.whitening_eigenvalues is None
            else result.whitening_eigenvalues
        ),
        explicit_ritz_whitening_retained=(
            np.asarray([], dtype=np.int64)
            if result.whitening_retained is None
            else result.whitening_retained
        ),
        explicit_ritz_p_decays=np.asarray(args.p_decays, dtype=np.float64),
        explicit_ritz_laguerre_orders=np.asarray(args.laguerre_orders, dtype=np.int64),
        explicit_ritz_s_decays=np.asarray(args.s_decays, dtype=np.float64),
        explicit_ritz_s_laguerre_orders=np.asarray(
            args.s_laguerre_orders,
            dtype=np.int64,
        ),
        explicit_ritz_d_decays=np.asarray(args.d_decays, dtype=np.float64),
        explicit_ritz_d_laguerre_orders=np.asarray(
            args.d_laguerre_orders,
            dtype=np.int64,
        ),
        explicit_ritz_geminal_gammas=np.asarray(args.geminal_gammas, dtype=np.float64),
        explicit_ritz_include_exp_geminals=args.include_exp_geminals,
        explicit_ritz_include_r12_geminal=args.include_r12_geminal,
        explicit_ritz_include_pd=args.include_pd,
        explicit_ritz_target_ha=target,
        explicit_ritz_raw_first_pole=raw_first_pole,
        explicit_ritz_first_bright_pole=first_bright_pole,
        explicit_ritz_first_bright_error_ha=first_bright_pole - target,
        explicit_ritz_bright_roots=bright_roots,
        explicit_ritz_bright_normalized_weights=bright_weights,
        omega=omega,
        intensity=intensity,
        peaks=np.asarray([peak.energy for peak in peaks], dtype=np.float64),
        final_replica_counts=final_replica["counts"],
        final_replica_block_poles=final_replica["block_poles"],
        final_replica_loo_poles=final_replica["loo_poles"],
        final_replica_loo_mean=final_replica["loo_mean"],
        final_replica_loo_jackknife_se=final_replica["loo_jackknife_se"],
        final_bootstrap_poles=final_replica["bootstrap_poles"],
        final_bootstrap_mean=final_replica["bootstrap_mean"],
        final_bootstrap_se=final_replica["bootstrap_se"],
        final_bootstrap_se_ev=final_replica["bootstrap_se_ev"],
        **sampling_payload,
        **scalar_payload,
    )


if __name__ == "__main__":
    main()
