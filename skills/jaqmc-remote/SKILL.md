---
name: jaqmc-remote
description: "Use when working on JaQMC with the ByteDance remote GPU worker: connecting through the workspace SSH host, finding active GPU workers with mlx, creating or using the remote JaQMC checkout and GPU Python environment, syncing local code, and running JAX GPU checks, JaQMC tests, or training smoke runs."
---

# JaQMC Remote

## Official BF-NKSR Response Workflow

- The public JaQMC response entry point now has one supported theory path:
  direct antisymmetric FermiNet response heads with CIS value warm-start,
  source-sector seeded candidate blocks, gauge-fixed metric-complement
  whitening, region-balanced `gauge-action` training, held-out action-oracle
  acceptance, moment/cutoff diagnostics, and
  `source-envelope-pz-sobol-antithetic` final sampling.
- Do not use obsolete enrichment branches such as weak residual candidate
  training, action-MINRES training, action-MINRES composite compression,
  source-prefactor response heads, Ritz enrichment acceptance, or
  system-specific spin symmetrization for new formal BF-NKSR runs. Historical
  notes below are retained only to explain old NPZ files and failed
  experiments; the current workflow above supersedes them.
- The response CLI records `response_official_workflow=True` and
  `response_official_workflow_name=official_warmstarted_bfnksr_projected_resolvent`
  in each new NPZ result.
- Current 2026-06-18 GPU smoke status: local and remote `tests/response`
  pass, H runs end-to-end with the first warm-start peak near the exact
  `1s->2p` value, but the new candidate block is still rejected by held-out
  strong residual/action-oracle checks. He also runs end-to-end, but the
  source-lift candidate collapses to a near-zero action complement and is
  rejected; its warm-start-only spectrum is not physically accurate yet.

## BF-NKSR Response Status, 2026-06-07

- Formal response workflow should use `--training-flow residual-enrichment`
  with held-out residual acceptance and moment/conditioning diagnostics.
  `--enrichment-ritz-accept` and `--enrichment-source-bright-gate` are
  optional diagnostics/stabilizers, not default paper-aligned acceptance.
- The residual-enrichment objective exposes the paper source-channel weights
  `w_a`: use `--residual-source-weights` for an explicit vector of length
  `1 + aux_source_count`, or the shorthand `--residual-physical-source-weight`
  / `--residual-aux-source-weight`. Defaults are still uniform for backward
  compatibility. For H2 learned-head attempts where the spectrum is read from
  the first physical dipole source and many auxiliary probes only shape the
  response subspace, start with `--residual-aux-source-weight 0.05` or `0.1`
  so candidate heads are not trained mostly on auxiliary-source residuals. The
  chosen weights are saved as `residual_source_weights` and printed by
  `scripts/analyze_bfnksr_npz.py`.
- Multi-head residual enrichment now scores the candidate block as a subspace:
  the capture term uses `R^dagger (S_CC + delta I)^-1 R` instead of averaging
  independent per-head captures, and the old-subspace redundancy/roughness
  terms use the same `S_CC` metric divided by the block's effective dimension
  `tr[(S_CC + delta I)^-1 S_CC]`. This keeps the single-head paper objective
  unchanged, prevents duplicate candidate heads from receiving duplicated
  residual credit or diluted penalties, and makes the main objective invariant
  to orthogonal rotations inside the candidate block; overlap whitening still
  handles final spectrum extraction.
- When `--enrichment-candidate-attempts > 1`, candidate selection now prefers
  the accepted candidate with the best worst-case held-out residual improvement
  (`holdout_objective_delta_min`, then `holdout_capture_ratio_min`) before
  falling back to median objective/capture. This keeps the acceptance gate
  conservative and reduces the chance that H2 learned-head runs pick a
  high-median but unstable candidate.
- Within each candidate attempt, best-epoch selection now evaluates all
  held-out residual sets and prefers the snapshot with the best worst-case
  held-out objective before median objective/capture tie-breaks. Earlier code
  used only the first held-out set for early stopping, even though final
  acceptance audited all held-outs.
- The physical dipole source in `src/jaqmc/response/ferminet_bfnksr.py` is
  centered at the nuclear charge center. Response heads for molecules should
  avoid `--source-prefactor` unless the intended state is known to share that
  source node. For H2/molecules prefer direct heads such as
  `--response-orbital-z-prefactor --opposite-spin-symmetry singlet
  --response-spatial-parity odd`.
- Auxiliary probes are part of the response subspace while the physical
  spectrum is still read from the first dipole source. The current code has
  centered `p_z exp(-alpha r^2)` probes via
  `--aux-source-gaussian-exponents`, bounded radial dipole probes
  `sum_i z_i (r_i/(s+r_i))^p Psi0` via
  `--aux-source-dipole-radial-powers` and
  `--aux-source-dipole-radial-scale`, and two-center atom-odd Gaussian probes
  via `--aux-source-atom-odd-gaussian-exponents`, plus two-center atom-odd
  Slater probes via `--aux-source-atom-odd-slater-decays`. The H2-focused
  family is two-center odd Slater orbital-ratio probes
  `(exp[-zeta r_A]-exp[-zeta r_B])/(exp[-r_A]+exp[-r_B])` via
  `--aux-source-bond-odd-slater-decays`. The correlated
  `--aux-source-bond-odd-ee-slater-decays` x
  `--aux-source-bond-odd-ee-scales` probes now multiply that odd ratio by a
  smooth two-center covalent pair feature, not just a global pair-distance
  scalar, so H2/molecular tests can represent more electron-nuclear correlated
  odd response shapes. Auxiliary probes are included in adaptive mixture `Q`,
  held-out validation, final matrices, strong residual audits, and saved
  `aux_source_count`.
- H atom direct Slater/source-envelope ladder remains the cleanest validation:
  `h_iter65_direct_decay_ladder10_frozen_524k_16rep.npz` gave n=2/n=3/n=4
  bright poles within about `1e-4 eV`, `1e-4 eV`, and `2e-5 eV`.
- H one-electron final matrix sampling now has the paper-aligned response
  proposal `--final-sampling one-electron-pz-envelope-mixture-sobol` and
  antithetic variant
  `--final-sampling one-electron-pz-envelope-mixture-sobol-antithetic`. It
  samples a normalized `s + p_z` mixture: the `p_z exp(-alpha r)` components
  cover source/response values, and spherical envelope components cover
  weak-form gradient terms at odd-response nodes. It saves proposal
  decays/weights in `final_one_electron_pz_envelope_*` and
  `final_one_electron_s_envelope_*`. The default component weights follow
  `--ground-weight`, `--source-weight`, and total `--head-weight`, matching the
  existing positive sampling Q more closely than equal component weights. Use
  this for H audits instead of pure `p_z^2` sampling when checking high bright
  poles.
- H 2026-06-07 s+pz final audit with frozen formal response params from
  `h_iter76_explicitdecay10_frozen_262k_16rep.npz`:
  `h_iter90_oneepz_spz_weightedq_ladder10_524k_32rep.npz` used
  `524288` Sobol samples / 32 replicas. At `overlap_cutoff=1e-8`, roots were
  `0.3750066492, 0.4444102530, 0.4687632669, 0.4792768518 Ha`; errors vs
  exact H n=2..5 are about `+0.000181, -0.000930, +0.000361, -0.01968 eV`.
  First three bright poles pass the `0.001 eV` target; the fourth remains near
  the ionization limit and is not converged.
- He best center from antithetic pz Sobol was near NIST:
  `he_iter102_radial2_frozen_pz_antithetic_1M_128rep.npz =
  0.7797697346 Ha`, error about `+0.000592 eV`, but uncertainty remained large
  (`bootstrap_se_ev ~ 0.058 eV`), so do not claim converged `<0.001 eV`.
- He next final-sampling iteration adds
  `--final-sampling source-envelope-spz-sobol` and
  `--final-sampling source-envelope-spz-sobol-antithetic`: this is the
  two-electron analog of the H `s+p_z` fix. It mixes the existing core/pz
  proposal with a small core/diffuse spherical floor, using saved metadata
  `source_envelope_spherical_floor=True`,
  `source_envelope_pz_weight`, and `source_envelope_spherical_weight`. The
  intent is to keep weak-form kinetic-gradient terms covered at odd-response
  nodes while preserving pz source concentration. By default the spherical
  weight is `--eps-env * (ground_weight + source_weight + head_weight)`; use
  `--source-envelope-spherical-floor-weight` for explicit scans. Do not use a
  large spherical floor for He: an early 131k smoke with comparable spherical
  and pz weights produced a low, noisy ghost root near `0.649 Ha`.
- He s+pz follow-up on worker `3888454`: the first implementation used one
  7D Sobol sequence with a component selector and was wrong for He; even
  `--source-envelope-spherical-floor-weight 0` gave a ghost root near
  `0.9086 Ha`. The sampler was fixed to stratify by component and reuse the
  validated old pz/core-diffuse Sobol samplers, with a unit test ensuring
  floor `0` matches the old pz sampler. After the fix,
  `he_iter107_radial2_frozen_spz_floor0_fixed_131k_16rep.npz` returned to the
  expected small-sample pz range (`0.7729592468 Ha`, noisy), but the default
  eps floor
  `he_iter108_radial2_frozen_spz_epsfloor_fixed_131k_16rep.npz` gave
  `0.7811241098 Ha`, high by about `0.037 eV` vs the NIST first singlet-P
  excitation. Conclusion: keep s+pz as a diagnostic/floor scan option, but do
  not treat it as a validated He improvement or default path.
- `scripts/analyze_bfnksr_npz.py` now supports saved raw-block coarsening via
  `--coarsen-blocks`, recomputing projection-aware LOO/bootstrap diagnostics
  after grouping consecutive final Sobol/MC blocks. This is for checking
  whether large uncertainty is a block-granularity artifact. It cannot repair
  old outputs that did not save raw projection blocks. In particular,
  `he_iter102_radial2_frozen_pz_antithetic_1M_128rep.npz` has
  `saved_projection_resampling=False` and no raw-block fields, so it cannot be
  coarsened offline; rerun the 1M pz-antithetic audit with current code to get
  projection-aware/coarsenable diagnostics. A newer small He control
  `he_iter106_radial2_frozen_pz_antithetic_131k_16rep_control.npz` does have
  raw blocks. Coarsening its 16 blocks by 2/4/8 still left first-root
  bootstrap/LOO errors at roughly `0.066-0.10 eV`, so the high uncertainty at
  131k is not merely a too-fine-block artifact.
- `scripts/analyze_bfnksr_npz.py` also supports automated reference-root
  audits via `--reference-roots-ha` and `--reference-tolerance-ev`. This is a
  reporting/validation layer, not a replacement for the BF-NKSR matrices: it
  reads the source-bright poles from each cutoff, saved LOO root means, and
  saved bootstrap root means, then prints `reference_comparison` lines with
  Hartree roots, eV errors, and `passed=True/False`. Use this whenever testing
  the paper flow against exact/reference excitation energies instead of
  manually reading peaks. Examples:

```bash
python scripts/analyze_bfnksr_npz.py \
  runs/h_audit.npz \
  --cutoffs 1e-10 1e-8 1e-7 1e-6 \
  --reference-roots-ha 0.375 0.4444444444 0.46875 \
  --reference-tolerance-ev 0.001

python scripts/analyze_bfnksr_npz.py \
  runs/h2_ferminet_formal/h2_bondodd_covalent_pair_screen_65k.npz \
  --cutoffs 1e-10 1e-8 1e-7 1e-6 \
  --reference-roots-ha 0.4680232388 0.5786030543 \
  --reference-tolerance-ev 0.001
```
- H2 formal ground-state test used
  `/opt/tiger/jaqmc/runs/h2_ferminet_formal/train_ckpt_019999.npz`,
  ground energy `-1.1745995283 Ha`. PySCF FCI reference at R=1.4 bohr:
  first bright around `0.4680232388 Ha`, second bright around
  `0.5786030543 Ha` (aug-cc-pVTZ rough reference; aug-cc-pVDZ first bright
  around `0.4650404362 Ha`).
- H2 response is not yet accurate. Source-only pole is about `0.57704 Ha`.
  Source-prefactor formal runs can accept candidates but bias high
  (`0.66-0.73 Ha`) because they force the dipole/source node. Direct orbital-z
  learned heads without source-prefactor currently fail held-out acceptance.
  Atom-odd auxiliary probes improve the fixed response subspace:
  `h2_iter10_atomodd_mixed_auxQ_forcedreject_65k.npz` gave
  `0.4998649834 Ha` with good moments and condition `1.7e6`, still high by
  about `0.032 Ha` (`0.87 eV`) versus the first-bright FCI reference.
  Wider 10-exponent atom-odd basis worsened to `0.5235139538 Ha`.
- H2 2026-06-07 continuation on worker `3888150`: the prolate
  `x_bond exp[-zeta(r_A+r_B)]` test
  `h2_iter12_prolate_atomslater_auxQ_forcedreject_65k.npz` worsened to
  `0.5374370070 Ha`, so do not use that family. The orbital-ratio two-center
  probes were better: `h2_iter13_bondratio_atomslater_auxQ_forcedreject_65k.npz`
  gave `0.4964157566 Ha`; ratio-only learned-head attempts
  `h2_iter15_bondratio_eejastrow_learned_4head_65k.npz` and
  `h2_iter16_bondratio_eejastrow_ritzgate_4head_65k.npz` rejected the neural
  heads by held-out/pole gates and fell back to fixed ratio poles around
  `0.497-0.510 Ha`. The remaining H2 error is still about `0.028 Ha`
  (`0.77 eV`), so the next useful theory step is a correlated/many-body
  response basis or better candidate initialization, not merely wider one-body
  decay ladders.
- 2026-06-13 H2 strong-oracle/source-aligned tangent update on worker
  `3903148`: the fixed-ground tangent path now has explicit finite-difference
  action diagnostics (`--enrichment-strong-oracle-action-mode
  compare-fd-tangent`), robust strong-oracle residual metrics (L2, winsor99,
  p95, p99, pointwise max), source-aligned readout tangents, and optional
  occupied/gauge projection
  `--response-tangent-source-orthogonalize` (default true). The source-aligned
  fit uses hidden features to approximate
  `(z-z_center) * raw_ground_orbital`, then projects that target away from the
  raw orbital span before fitting the readout vector. The robust oracle max
  metrics plus candidate projected-value/action norm diagnostics are saved in
  `enrichment_history_strong_oracle_*` fields.
- H2 diagnostic results from the same update: `h2_iter38` compared old
  AD/projected-value action columns with explicit finite-difference tangent
  action columns and found agreement at numerical precision
  (`rel_l2_max=9.6e-13`, `rel_p99_max=3.8e-12`), so the action oracle itself is
  not the main bug. Random-readout 8-head smoke `h2_iter39` gave only a weak
  validation signal (`val_ratio_w99_max=0.996`,
  `val_ratio_p99_max=0.946`), not the useful `<0.9` winsor99 target.
  Source-aligned without target projection `h2_iter40` overfit and worsened
  validation (`val_ratio_w99_max=1.207`). Source-aligned with
  occupied/gauge projection `h2_iter41_sourcealigned_orth8_oracle_smoke.npz`
  improved the training strong residual (`3.865 -> 1.379`) but still failed
  held-out robust validation (`val_ratio_w99_max=1.029`,
  `val_ratio_p99_max=0.970`, `val_ratio_pointmax_max=0.966`) and was correctly
  rejected by residual acceptance. A tiny 8-sample history smoke `h2_iter42`
  saved the new history fields and showed `val_ratio_w99_max=0.324`, but this
  is too small/noisy for a physics conclusion. A later 8-sample prefilter
  history smoke
  `h2_iter43_sourcealigned_orth8_prefilter_history_smoke.npz` saved
  `candidate_value_norm_min=1.54e-2`,
  `candidate_action_norm_min=9.30e-3`, and
  `candidate_action_condition_max=3.30`; despite nonzero/nonpathological
  candidate columns, held-out robust ratios worsened
  (`val_ratio_w99_max=1.420`, `val_ratio_p99_max=1.717`), so the issue is
  still candidate generalization, not merely null candidate columns.
- Conclusion from the 2026-06-13 H2 diagnostics: the remaining failure is not
  an AD-vs-FD Hessian/action mismatch. The current candidate directions can
  reduce training residuals but do not yet generalize on held-out residuals for
  H2. Next useful steps are turning the saved projection/action-norm diagnostics
  into explicit candidate gates or candidate ranking, and more
  physically constrained residual-enriched candidate generation, not accepting
  the current source-aligned heads or reading their fallback smoke spectra as
  excitation energies.
- 2026-06-13 continuation: strong-oracle diagnostics are now actionable gates
  and ranking keys. CLI thresholds include
  `--enrichment-strong-oracle-max-validation-ratio-winsor99`,
  `--enrichment-strong-oracle-max-validation-ratio-p99`,
  `--enrichment-strong-oracle-max-validation-ratio-pointwise`,
  candidate value/action norm floors, and candidate action-condition ceiling.
  Defaults keep the gate disabled. When enabled, rejected candidates save
  `enrichment_strong_oracle_passed=False` and
  `enrichment_history_strong_oracle_passed=False`; accepted attempt ranking
  prefers lower finite winsor99/p99 oracle ratios before falling back to
  held-out residual objective/capture.
- H2 gate/fallback smokes after the same change: source-aligned readout with
  stronger ridge
  `h2_iter44_sourcealigned_orth8_oraclegate_ridge1e4_smoke.npz` was rejected by
  the new oracle gate (`val_ratio_w99_max=1.585`,
  `candidate_action_condition_max=1.86`). The last-backbone structured
  fallback `h2_iter45_lastbackbone8_oraclegate_smoke.npz` perturbs one final
  backbone Dense output column per tangent; it also reduced training residual
  but worsened held-out strong residual (`val_ratio_w99_max=1.562`,
  `candidate_value_norm_min=3.1e-3`, `candidate_action_norm_min=3.9e-3`) and was
  `strong_oracle_rejected`. A formal learned direct-head 50-epoch smoke
  `h2_iter46_learned_direct8_oraclegate_50ep_smoke.npz` improved held-out
  residual objective strongly (`holdout_capture_ratio_min=2.72`,
  `holdout_objective_delta_min=660`) but failed moment stability
  (`m1_rel=0.136`) and narrowly missed the oracle gate
  (`val_ratio_w99_max=1.035`). A lower-lr/stronger-roughness run
  `h2_iter47_learned_direct8_oraclegate_rough_lr1e3_100ep_smoke.npz` was worse:
  residual-rejected with `holdout_capture_ratio_min=0.405` and
  `val_ratio_w99_max=2.683`.
- Current question for theory help: after action consistency, robust oracle
  gates, source-aligned readout, occupied/gauge projection, last-backbone
  structured tangents, and a small formal learned-head run, H2 still shows
  training/weak-form improvement without held-out strong-residual
  generalization. The next step likely needs theory guidance on the candidate
  generation/validation loop or the strong residual sampling/projection
  definition, not another blind hyperparameter sweep.
- 2026-06-07/08 H2 correlated-basis local update: the existing
  `bond_odd_ee` probe was upgraded from
  `bond_odd_ratio * mean r_ij/(s+r_ij)` to
  `bond_odd_ratio * mean r_ij/(s+r_ij) *
  (w_A(i)w_B(j)+w_B(i)w_A(j))`, where `w_A/w_B` are smooth two-center
  assignment weights at the same scale. This keeps the formal BF-NKSR source
  probe path unchanged but gives H2 a covalent-pair screened odd channel that
  should be screened before concluding the H2 error is a pure training issue.
  Local checks passed (`ruff`, `py_compile`, `tests/response`: 101 passed).
  Workspace-host checks under `.venv-gpu` also passed (`ruff`, `py_compile`,
  `tests/response`: 97 passed). The new reference-root analyzer was exercised
  on saved H/He outputs: H n=2..4 bright roots pass `0.001 eV`, H n=5 still
  fails near the ionization limit, and the old He 1M center value passes only
  as a center-value smoke because it lacks projection-aware/raw-block
  resampling. No new GPU H2 rerun was possible in this pass because
  `mlx worker list` was empty. Next H2 screen should compare the old fixed
  bond-ratio basis against covalent-pair-screened probes, for example:

```bash
python scripts/ferminet_bfnksr_response.py \
  --checkpoint runs/h2_ferminet_formal/train_ckpt_019999.npz \
  --ground-energy -1.1745995283 \
  --output runs/h2_ferminet_formal/h2_bondodd_covalent_pair_screen_65k.npz \
  --seed 701 --n-heads 4 --hidden 16 --hidden-double 4 --layers 2 \
  --determinants-per-head 1 --independent-heads \
  --response-orbital-z-prefactor --response-spatial-parity odd \
  --training-flow none \
  --aux-source-bond-odd-slater-decays 0.25 0.4 0.65 1.0 1.6 \
  --aux-source-bond-odd-ee-slater-decays 0.25 0.4 0.65 1.0 1.6 \
  --aux-source-bond-odd-ee-scales 0.35 0.7 1.4 \
  --final-sampling sobol-envelope-antithetic --final-samples 65536 \
  --final-sobol-replicas 8 --matrix-batch-size 4096 \
  --envelope-decay 0.7 --overlap-cutoff 1e-8 \
  --cutoff-diagnostic-values 1e-10 1e-8 1e-7 1e-6 \
  --final-diagnostic-roots 4
```

For the next learned-head formal H2 screen, use the same auxiliary family but
turn residual enrichment back on and down-weight auxiliary residual channels:

```bash
python scripts/ferminet_bfnksr_response.py \
  --checkpoint runs/h2_ferminet_formal/train_ckpt_019999.npz \
  --ground-energy -1.1745995283 \
  --output runs/h2_ferminet_formal/h2_learned_physical_residual_weighted_65k.npz \
  --seed 703 --n-heads 4 --hidden 16 --hidden-double 4 --layers 2 \
  --determinants-per-head 1 --independent-heads \
  --response-orbital-z-prefactor --response-spatial-parity odd \
  --training-flow residual-enrichment \
  --enrichment-candidate-heads 2 --enrichment-candidate-attempts 3 \
  --enrichment-validation-holdouts 3 --enrichment-holdout-min-pass-fraction 1 \
  --enrichment-strong-residual-samples 256 \
  --enrichment-max-strong-residual-epsilon-over-eta 5 \
  --residual-aux-source-weight 0.05 \
  --aux-source-bond-odd-slater-decays 0.25 0.4 0.65 1.0 1.6 \
  --aux-source-bond-odd-ee-slater-decays 0.25 0.4 0.65 1.0 1.6 \
  --aux-source-bond-odd-ee-scales 0.35 0.7 1.4 \
  --final-sampling sobol-envelope-antithetic --final-samples 65536 \
  --final-sobol-replicas 8 --matrix-batch-size 4096 \
  --envelope-decay 0.7 --overlap-cutoff 1e-8 \
  --cutoff-diagnostic-values 1e-10 1e-8 1e-7 1e-6 \
  --final-diagnostic-roots 4
```

2026-06-08 H2 learned-head result on worker `3890656`:
`h2_iter17_learned_weighted_blockrobust_65k.npz` used the weighted physical
source objective (`residual_source_weights` normalized to physical `0.5` and
each auxiliary `0.025`) plus block-subspace residual capture and robust
held-out early stopping. All three candidate attempts passed held-out residual
gates but failed the strong-residual gate with
`epsilon_over_eta` about `508`, `222`, and `1957`, so `accepted_heads=0` and
the final spectrum fell back to the fixed source+auxiliary basis. Analyzer
comparison against H2 references `0.4680232388, 0.5786030543 Ha` failed:
at `overlap_cutoff=1e-8` the source-bright roots were
`0.5093986732, 0.8853880482 Ha`, errors `+1.126 eV` and `+8.348 eV`; saved
LOO root means were `0.5031653573, 0.8302352266 Ha`, errors `+0.956 eV` and
`+6.847 eV`; bootstrap means were `0.4909888509, 0.7371525275 Ha`, errors
`+0.625 eV` and `+4.314 eV`. Cutoff spread and bootstrap/LOO errors were also
large (`~0.7-0.8 eV` for the first root), so this is not near the
`0.001 eV` goal. Next H2 theory step should not merely relax the strong
residual gate; the learned candidates are capturing held-out weak residuals
while producing very large pointwise residuals. Focus on candidate ansatz/
sampling regularization that reduces strong residuals, or on adding a
physics-shaped fixed/initial response channel closer to the first bright H2
state.
- 2026-06-08 H2 fully direct Schur-complement screen on worker `3890656`:
  `h2_iter18_fulldirect_schur_65k.npz` removed both `--source-prefactor` and
  `--response-orbital-z-prefactor`, keeping only singlet exchange symmetry and
  `--response-spatial-parity odd`. The residual-enrichment objective used the
  old-subspace Schur complement for candidate-block capture and roughness, and
  ran with `--enrichment-lambda-old 0.0`. All three attempts again passed
  held-out weak residual gates but failed the strong-residual gate:
  `epsilon_over_eta` was about `1219`, `338`, and `4976`. The best rejected
  attempt selected a pole near `0.435814 Ha`; another selected `0.448890 Ha`,
  showing again that attractive weak-form poles are not reliable when the
  strong residual is huge. Because `accepted_heads=0`, the final spectrum
  fell back to fixed source+auxiliary probes. Analyzer comparison against H2
  references `0.4680232388, 0.5786030543 Ha` failed: at
  `overlap_cutoff=1e-8`, source-bright roots were
  `0.4956925165, 0.7196978177 Ha`, errors `+0.753 eV` and `+3.839 eV`; LOO
  means were `0.4953053359, 0.7169392153 Ha`, errors `+0.742 eV` and
  `+3.764 eV`; bootstrap means were `0.4921674505, 0.7013901861 Ha`, errors
  `+0.657 eV` and `+3.341 eV`. Conclusion: removing orbital/source prefactors
  and using a Schur-complement block objective is necessary cleanup but not
  sufficient; the immediate bottleneck is still that weak residual training
  creates rough/pointwise-bad candidates. Next useful step is to add training
  pressure or sampling coverage tied to the strong residual/roughness
  pathology, rather than accepting these candidates or only changing final
  matrix statistics.
- 2026-06-11 local theory update: residual-enrichment now has
  `--enrichment-sobolev-metric-weight alpha`, which changes candidate residual
  capture from the pure overlap metric to the Schur-complement Sobolev metric
  `S_C|B + alpha T_C|B`, where `T` is the positive weak-form kinetic/gradient
  matrix. This is a theory-level stabilization, not a snapshot-selection
  workaround: high-curvature candidates become expensive in the Riesz
  representer used to capture the residual. Default `alpha=0` preserves old
  commands; H2/molecular direct-response tests should scan positive values
  before adding ad hoc gates. Local `tests/response` passed with `104 passed`.
- 2026-06-11 H2 fully direct Sobolev metric screen on worker `3899190`:
  `h2_iter19_fulldirect_sobolev1_65k.npz` used no source/orbital prefactor and
  `--enrichment-sobolev-metric-weight 1.0`. The outer workspace SSH target was
  `128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org`; for nested worker
  SSH from the workspace host, use `ssh -F /dev/null ...` so the inner SSH does
  not accidentally read workspace config and route through the jump proxy.
  Sobolev capture reduced the trained objective scale and the fixed
  source+auxiliary first physical bright root moved closer to the H2 FCI
  reference: with `--root-floor 0.35`, cutoff `1e-8` gave
  `0.4611061218 Ha` vs reference `0.4680232388 Ha` (error `-0.188 eV`), and
  cutoffs `1e-7/1e-6` gave about `0.46394/0.46397 Ha` (error `~ -0.11 eV`).
  This is directionally better than the previous fixed-basis `~0.4957 Ha`.
  However, all learned candidates still failed the strong residual gate:
  `epsilon_over_eta` was about `710`, `2052`, and `699`, so
  `accepted_heads=0`. The default root-floor analyzer also sees a low-energy
  small-weight ghost root around `0.284 Ha`, and LOO/bootstrap uncertainties
  are multiple eV. Conclusion: the Sobolev metric is a useful theory-level
  stabilization direction, but `alpha=1` is not enough for a reliable molecular
  high-precision workflow. Next steps should strengthen the metric/regularizer
  and/or improve the positive sampling measure Q so accepted direct neural
  candidates can pass the strong residual gate, not rely on fixed probes or
  post-hoc root floors.
- 2026-06-11 causality check for the Sobolev metric: reran final-only H2
  fixed source+aux matrices from `h2_iter19_fulldirect_sobolev1_65k.npz` with
  `--training-flow none`, restored `active_heads=0`, identical seed/final
  Sobol samples, and only changed `--enrichment-sobolev-metric-weight` between
  `0` and `1`. The projected poles, weights, LOO diagnostics, and bootstrap
  diagnostics were identical; first large-weight pole was `0.5085664329 Ha` in
  both runs. Therefore, when no heads are accepted, the Sobolev metric does not
  leak into final fixed-basis matrices. The earlier fixed-basis shift toward
  `0.461 Ha` must be treated as stochastic/final-estimator/Q-path sensitivity,
  not as a causal accuracy improvement from the metric. This reinforces that
  the next non-patch work is strong-residual decomposition, Q coverage
  diagnostics/upgrades, and robust strong-polish for candidates.
- 2026-06-11 H2 strong-residual decomposition from a final-only audit of
  `h2_iter19_fulldirect_sobolev1_65k.npz` showed the pointwise failure is
  strongly localized. With `512` audit samples at omega `0.4680232388 Ha`,
  global `epsilon_over_eta` was about `5.66e2`; by overlapping regions, the
  node tube contained `52/512` samples and about `52.7%` of the residual
  norm with conditional `epsilon_over_eta ~3.87e3`, the tail contained
  `52/512` samples and about `57.1%` of the residual with
  `epsilon_over_eta ~2.99e3`, high-action samples contained about `37.0%`,
  and the bulk contained only about `5.9%`. Electron-nucleus cusp samples were
  rare (`4/512`) but contributed about `15.6%`; the single electron-electron
  cusp sample had a large conditional ratio but negligible total fraction.
  Conclusion: do not loosen the strong gate or tune root floors. Improve Q
  coverage and candidate regularization around node/tail/high-action regions.
- 2026-06-11 local Q-coverage update: adaptive mixture Q now has optional
  general molecule-safe components `--q-node-weight` with
  `--q-node-ground-power beta` for a tempered
  `envelope^(1-beta)|Psi0|^(2 beta)` density, `--q-tail-weight` with
  `--q-tail-envelope-decay` for a diffuse envelope tail component, and
  `--q-kinetic-weight` for a weak-gradient/source-aux-head kinetic density.
  Defaults are zero, preserving old runs. These components affect the formal
  training/holdout `--enrichment-sampling mixture` path and final
  `--final-sampling mixture`; they are saved in output NPZ files as
  `q_node_weight`, `q_node_ground_power`, `q_tail_weight`,
  `q_tail_envelope_decay`, `q_tail_effective_envelope_decay`, and
  `q_kinetic_weight`. The intended H2 ablation is amplitude-only versus
  node+tail versus node+tail+kinetic while keeping the strong-residual gate
  unchanged.
- 2026-06-11 H2 Q ablation on worker `3899190`: a same-seed small formal
  smoke with node+tail Q
  `h2_iter21_qnodetail_smoke_16k.npz` (`--q-node-weight 0.5`,
  `--q-node-ground-power 0.25`, `--q-tail-weight 0.25`,
  `--q-tail-envelope-decay 0.25`) still rejected learned heads by the strong
  gate. The candidate held-out weak capture improved, but
  `strong_residual_epsilon_over_eta` was about `5.94e2`; the same-size
  amplitude-only baseline `h2_iter22_qbaseline_smoke_16k.npz` had a smaller
  candidate strong residual around `4.14e2`. Because both had
  `accepted_heads=0` and identical final Sobol samples, their final fixed
  source+aux spectra were identical. Conclusion: node+tail Q alone is not a
  reliable improvement; it can increase weak capture while making the
  pointwise candidate rougher. The first kinetic-Q MCMC attempt
  `h2_iter20_qnodetailkinetic_smoke_16k` was stopped because evaluating
  gradient density inside every MH proposal led to a very expensive JIT path
  (GPU memory allocated, GPU util near zero, no progress for several minutes).
  Kinetic/residual-biased Q should be implemented as a staged or cached
  proposal, not as per-proposal gradient evaluation in the MCMC target.
- 2026-06-11 local robust strong-polish update: residual enrichment now has
  optional `--enrichment-strong-polish-epochs`,
  `--enrichment-strong-polish-samples`,
  `--enrichment-strong-polish-batch-size`,
  `--enrichment-strong-polish-learning-rate-scale`, and
  `--enrichment-strong-polish-clip`. It runs after weak/Sobolev training and
  optimizes a soft-clipped pointwise strong residual using fixed current
  correction-vector coefficients; the final strong audit still recomputes
  coefficients independently. A held-out residual pass guard is required
  before a polish update can replace the best candidate.
- 2026-06-11 H2 strong-polish smoke:
  `h2_iter23_strongpolish_smoke_4k.npz` showed the polish subset loss can drop
  (`epsilon_max` roughly `49.6 -> 32.2` in two tiny polish steps) but can
  worsen independent validation: held-out residual pass was only `1/2`, and
  independent candidate `epsilon_over_eta` was about `9.18e2`. The guard was
  tightened to reuse the same held-out residual acceptance rule. The rerun
  `h2_iter25_strongpolish_passguard_smoke_4k.npz` printed
  `response_enrichment_strong_polish_reject ... holdout_pass=1/2`, confirming
  the harmful polish update was discarded. Conclusion: strong-polish is now a
  safe optional stage but not yet an accuracy fix; it needs a better proposal
  for polish samples and likely coefficient-refresh or held-out strong
  validation before it will improve H2.
- 2026-06-12 local/remote graph-metric update: residual enrichment now has
  `--enrichment-graph-metric-eta`. When positive, candidate residual capture
  is evaluated in the frequency-dependent graph metric on the
  S-Schur-complement/overlap-whitened candidate block:
  `(Khat - omega I)^2 + eta_metric^2 I + alpha That + delta I`. This changes
  only enrichment training/selection; final projected spectrum still uses the
  original generalized eigenproblem. Local checks passed (`ruff`,
  `tests/response`: 109 passed at the graph-metric stage; 110 passed after the
  oracle diagnostic).
- 2026-06-12 H2 graph-metric smoke on worker `3900407`
  (`ssh -p 10121 fdbd:dc03:16:270::74`):
  `h2_iter26_graphmetric_eta005_smoke_16k.npz` used
  `--enrichment-graph-metric-eta 0.05` and
  `--enrichment-sobolev-metric-weight 0.5` with the same small H2 settings as
  the baseline. The candidate still failed the strong gate, with
  `strong_residual_epsilon_over_eta ~7.35e2`, worse than the same-size
  amplitude/Sobolev baseline `~4.14e2`. The graph metric increased held-out
  weak capture (`holdout_ratio_min ~4.82`) but did not make a strong-form good
  direction. Do not treat graph metric alone as solved; it may need a more
  conservative eta/anneal, but the immediate diagnostic points to candidate
  direction quality.
- 2026-06-12 strong oracle linear-combination diagnostic was added via
  `--enrichment-strong-oracle-samples` and
  `--enrichment-strong-oracle-ridge`. It freezes the trained candidate block,
  solves the old weak correction vector, constructs candidate action columns
  `(z - Hbar) chi_j` on strong samples, fits a ridge oracle on half the samples,
  and reports train/validation strong residual epsilons. The H2 baseline
  `h2_iter27_oracle_baseline_smoke_16k.npz` printed:
  `train_old_max=1.685e+01`, `train_oracle_max=1.659e+01`,
  `val_old_max=8.011e+00`, `val_oracle_max=8.746e+00`,
  `val_ratio_max=1.092e+00`. Thus the current learned free-direct candidate
  block does not provide a held-out useful strong-form linear combination;
  the bottleneck is likely ansatz/regularity/tangent initialization rather
  than only Galerkin coefficient selection.
- 2026-06-12 GPT-Pro follow-up implementation:
  - Added `--enrichment-local-action-metric-weight`,
    `--enrichment-local-action-metric-clip`, and
    `--enrichment-local-action-metric-samples`.  The local-action correction
    adds the empirical strong-action Gram matrix
    `<A_z chi_i, A_z chi_j>` to the enrichment capture metric only; final
    spectrum extraction is unchanged.  When `samples > 0`, the action metric
    is estimated on a small cached enrichment subset instead of the current
    weak-form training batch, matching the staged/cached recommendation and
    avoiding large per-step Hessian batches.  Local `tests/response` passed
    (`113 passed`) and remote ruff/py_compile/target tests passed.
  - H2 local-action free-head smokes on worker `3900407`:
    `h2_iter29_localaction_cache16_smoke_1k.npz` (`weight=1e-3`) and
    `h2_iter30_localaction_cache16_w1e5_smoke_1k.npz` (`weight=1e-5`) used
    one free direct head, graph eta `0.05`, cached action samples `16`, and
    512/256 envelope samples.  They were essentially identical:
    `strong_oracle val_ratio_max=1.041`, candidate
    `epsilon_over_eta=2.38e2`, final audit `2.51e2`, rejected with
    `holdout_pass=0/2`.  The cached local-action correction alone did not
    make the free-direct candidate useful.
  - Added frozen-feature/readout MVP flags
    `--response-copy-ground-matching-params` and
    `--response-train-only-readout`.  The first copies shape-compatible
    ground FermiNet leaves into the response model; the second masks training
    to `orbital_layer` leaves only.  H2 setup used `n_heads=4`, `hidden=64`,
    `hidden_double=8`, `layers=4`, `determinants_per_head=1`, copied
    `20` leaves from the ground checkpoint, and trained readout only.
    `h2_iter31_frozen_readout_smoke_1k.npz` gave a slightly nonnegative
    oracle signal (`train_old_max=34.7 -> train_oracle_max=20.2`,
    `val_ratio_max=0.994`) but the candidate strong residual was terrible
    (`epsilon_over_eta=2.49e3`, tail/high-action dominated), so it was
    rejected and no heads were accepted.  Adding cached local-action
    (`h2_iter32_frozen_readout_localaction_smoke_1k.npz`, weight `1e-5`)
    reproduced the same oracle/strong-residual outcome.  Current conclusion:
    frozen readout is a better direction probe than random free heads, but it
    is still not a usable strong-form candidate; the next likely step is true
    signed finite-difference/JVP tangent columns rather than more local-action
    sweeps.
  - Added fixed signed finite-difference ground tangent heads via
    `--response-fixed-ground-tangent`.  The default
    `--response-tangent-mode random-readout` builds spin-tied random
    `orbital_layer` directions and evaluates
    `(Psi(theta+eps v)-Psi(theta-eps v))/(2 eps)` as fixed response columns.
    `--response-tangent-mode structured-readout` cycles over
    per-determinant/per-orbital readout output channels with spin-tied hidden
    vectors.  The tangent heads reuse the existing projected value/Laplacian
    action path, so final spectrum extraction is unchanged.  Local
    `tests/response` passed (`117 passed`); remote A100 worker `3901036`
    ruff/py_compile/target tests also passed.
  - H2 fixed-tangent smokes used the formal H2 FermiNet ground checkpoint
    `runs/h2_ferminet_formal/train_ckpt_019999.npz`, ground energy
    `-1.1745995283 Ha`, odd spatial parity, singlet spin projection, and the
    same 20 H2 bond-odd auxiliary sources.  Random readout tangent with
    `4` directions and `eps=1e-3` produced
    `h2_iter33_fixedtangent4_oracle_smoke_1k.npz`: strong oracle improved
    train samples (`3.865 -> 2.113`) but worsened validation
    (`12.88 -> 13.55`, `val_ratio_max=1.052`); strong residual remained large
    (`epsilon_over_eta=4.76e2`, tail/high-action dominated).  Lowering
    `eps` to `1e-4` gave the same oracle ratio, so the issue is not simply
    finite-difference step size.  Random `8` directions improved train
    residual further (`3.865 -> 0.926`) but still worsened validation
    (`val_ratio_max=1.046`), and increasing oracle ridge to `1e-2` did not
    help (`1.045`).  Structured per-determinant/per-orbital readout tangent
    with `8` directions was worse (`val_ratio_max=1.083`).  Current
    conclusion: signed readout tangents have train-sample fitting capacity but
    still do not provide held-out strong-form useful H2 directions; next
    theory/implementation question is whether to add projection-norm/action
    prefilters, include last-backbone tangent directions, or compute explicit
    finite-difference action columns instead of relying on the projected
    value/Laplacian path.
- 2026-06-07 local alignment update: Slater atom-odd and prolate bond-odd
  auxiliary source support were added and local checks passed (`ruff`,
  `tests/response`: 56 passed).
  Code was synced to `/opt/tiger/jaqmc` on the workspace host and remote
  `py_compile` passed, but no active GPU worker was listed by `mlx worker list`
  or `mlx worker list --show-all`; the old `3887622` worker refused SSH.
  Later in the same session, a temporary workspace A100 worker `3888150`
  (`ssh -p 11038 fdbd:dc03:9:651::138`) was launched, `/tmp/jaqmc_venv` was
  recreated with Python 3.12/JAX CUDA, and remote checks passed (`ruff`,
  `tests/response`: 56 passed).

## Core Locations

- Local repo: `/Users/bytedance/Desktop/jaqmc`
- Workspace SSH host: `128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org`
- Remote worker code checkout: `/opt/tiger/jaqmc`
- Remote GPU Python env: `/opt/tiger/jaqmc/.venv-gpu`
- Remote managed Python 3.12 should be created through `uv`.
- Do not use or modify `/opt/tiger/iqc/.venv-gpu` for JaQMC work.

Current worker from the 2026-06-12 response continuation:

- worker id: `3901036`
- GPUs: `1 x A100-SXM-80GB`
- worker SSH target: `ssh -p 11110 fdbd:dc03:16:138::82`
- hostname: `n176-050-082`
- JaQMC checkout: `/opt/tiger/jaqmc`
- GPU env: `/opt/tiger/jaqmc/.venv-gpu`, with JAX `0.9.1` reporting
  `[CudaDevice(id=0)]`

Latest temporary worker from the 2026-06-07 response session, expected stale
after cleanup:

- worker id: `3888454`
- GPUs: `1 x A100-SXM4-80GB`
- worker SSH target: `ssh -p 9572 fdbd:dc03:9:339::138`
- hostname: `mlxlab3kzfmi7r6a1d4490-20260601083632-2z5ewv-nh140d-worker`
- JaQMC checkout: `/opt/tiger/jaqmc`
- GPU env: `/opt/tiger/jaqmc/.venv-gpu`, with JAX `0.9.1` reporting
  `[CudaDevice(id=0)]`

Most recent worker from the 2026-06-06 response session, now stale:

- worker id after rediscovery: `3887622` (`deleted` in 2026-06-07
  `mlx worker list --show-all`)
- GPUs: `1 x A100-SXM-80GB`
- worker SSH target: `ssh -p 10380 fdbd:dc03:16:214::86`
- hostname: `n176-069-086`
- JaQMC checkout: `/opt/tiger/jaqmc`
- GPU env: `/tmp/jaqmc_venv`, created with `uv venv --python 3.12`
- Use `CUDA_VISIBLE_DEVICES=0` by default unless a task explicitly needs
  multiple GPUs.

Previous worker from the 2026-06-06 session:

- active worker id after rediscovery: `3887306`
- GPUs: `8 x A100`
- worker SSH target: `ssh -p 10403 fdbd:dc03:16:344::218`
- hostname: `n176-101-218`

Backup worker from the 2026-06-06 session:

- active worker id after rediscovery: `3887297`
- GPUs: `1 x A100-SXM-80GB`
- worker SSH target: `ssh -p 11163 fdbd:dc03:16:205::24`
- hostname: `n176-067-024`
- `/tmp/jaqmc_venv` may need recreation because the worker had system Python
  3.11 only; use `uv venv --python 3.12`, not `python3 -m venv`.

Current worker from the 2026-06-05 session:

- active worker id after rediscovery: `3886114`
- GPUs: `1 x A100-SXM4-80GB` reported by `nvidia-smi -L`
- worker SSH target: `ssh -p 10147 fdbd:dc03:16:138::82`
- hostname: `n176-050-082`
- `/tmp/jaqmc_venv` was created on this worker and JaQMC/JAX GPU import
  passed with `jax.devices()` reporting `[CudaDevice(id=0)]`

Known worker from the 2026-06-04 session:

- active worker id after rediscovery: `3881980`
- GPUs: `8 x A100-SXM4-80GB` reported by `nvidia-smi -L`
- worker SSH target: `ssh -p 10205 fdbd:dc03:16:130::70`
- hostname: `n176-048-070`

Previous worker `3882361` (`ssh -p 10308 fdbd:dc03:16:325::20`) disappeared from `mlx worker list` during setup.

Treat worker id/IP/port as ephemeral. Re-discover before relying on them.

## Authentication

If SSH fails with `Permission denied (gssapi-with-mic)`, refresh Kerberos locally:

```bash
/usr/bin/kinit --keychain zhang.xiaoyu@BYTEDANCE.COM
```

Then retry the SSH command.

## Find GPU Workers

Run worker discovery on the workspace host:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org 'mlx worker list'
```

Pick a running GPU worker, then connect through the workspace host:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'hostname && nvidia-smi -L'"
```

If multiple workers exist, prefer an 8-card A100/A800 worker unless the task only needs one GPU.

## Run Remote Commands

Use this pattern for commands in the JaQMC checkout:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && CUDA_VISIBLE_DEVICES=0 /tmp/jaqmc_venv/bin/python -V'"
```

Prefer single-card runs unless the user explicitly asks for multi-GPU. Use `CUDA_VISIBLE_DEVICES=0` for smoke tests so unrelated work on other cards is not disturbed.

## First-Time Remote Setup

Create the remote checkout directory:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'mkdir -p /opt/tiger/jaqmc'"
```

Sync tracked local files to the worker from `/Users/bytedance/Desktop/jaqmc`:

```bash
git ls-files -z | tar --null -T - -cf - | \
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && tar -xf -'"
```

This intentionally skips untracked files, `.venv`, caches, and local run outputs. Add explicitly requested untracked files to the tar list only when needed.

Create or refresh the GPU Python environment. Use the ByteDance proxy and package index; this is much faster than frozen lock sync on the worker:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && \
    export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32 LD_LIBRARY_PATH= NCCL_DEBUG=WARN NVIDIA_TF32_OVERRIDE=0 SKIP_MERLIN_OFFICIAL_INTERNAL_INSTALL=TRUE && \
    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118 && \
    export no_proxy=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8 NO_PROXY=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8 && \
    uv --version && \
    uv venv --python 3.12 --clear /tmp/jaqmc_venv && \
    . /tmp/jaqmc_venv/bin/activate && \
    uv pip install -e \".[cuda12]\" --index https://bytedpypi.byted.org/simple/'"
```

Known passing result from 2026-06-04: this installed JaQMC into `/tmp/jaqmc_venv`, then `import jax, folx, jaqmc` passed and JAX reported 8 CUDA devices.

If `uv` is unavailable on the worker, install it into a temporary helper env or user site, then still use `uv venv --python 3.12`. Do not fall back to system `python3 -m venv`: some workers only have Python 3.11 and no `ensurepip`, while JaQMC requires Python >=3.12.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && \
    python3 -m pip install --user uv && \
    export PATH=\$HOME/.local/bin:\$PATH && \
    uv venv --python 3.12 --clear /tmp/jaqmc_venv && \
    . /tmp/jaqmc_venv/bin/activate && \
    uv pip install -e \".[cuda12]\" --index https://bytedpypi.byted.org/simple/'"
```

After setup, always verify CUDA JAX and key optional dependency imports:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && \
    /tmp/jaqmc_venv/bin/python -c \"import jax, folx, jaqmc; print(jax.__version__); print(jax.devices()); print(\\\"folx ok; jaqmc ok\\\")\"'"
```

## Official Update Branch Workflow

When the user asks to sync with the official JaQMC `update` branch on the worker, use this before running jobs:

```bash
cd /opt/tiger/jaqmc
git fetch --depth 1 origin update
git checkout -B update FETCH_HEAD
git rev-parse HEAD
git log -1 --oneline
```

Do not run this if local uncommitted worker edits must be preserved.

## Sync Local Changes To Worker

For a normal task, sync only the changed tracked files:

```bash
git diff --name-only HEAD -z | tar --null -T - -cf - | \
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && tar -xf -'"
```

If the command has no input because there are no tracked local changes, sync specific files instead:

```bash
tar -cf - src/jaqmc/app/cli.py tests/cli_dry_run_test.py | \
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && tar -xf -'"
```

Do not sync unrelated dirty files.

## Common Remote Validation

Confirm Python and JAX see the GPU:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && CUDA_VISIBLE_DEVICES=0 /tmp/jaqmc_venv/bin/python -c \"import jax; print(jax.__version__); print(jax.devices())\"'"
```

Compile the main packages:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && /tmp/jaqmc_venv/bin/python -m compileall -q src/jaqmc src/jaqmc_legacy tests'"
```

Run focused tests on one GPU:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && CUDA_VISIBLE_DEVICES=0 /tmp/jaqmc_venv/bin/python -m pytest -q tests/hydrogen/atom_test.py tests/cli_dry_run_test.py'"
```

Run a short CLI smoke test:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'cd /opt/tiger/jaqmc && CUDA_VISIBLE_DEVICES=0 /tmp/jaqmc_venv/bin/jaqmc hydrogen-atom train train.run.iterations=1'"
```

Check GPU process status without disturbing a run:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10205 fdbd:dc03:16:130::70 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'"
```

## Formal BF-NKSR Response Workflow

Use `scripts/ferminet_bfnksr_response.py` for the internal formal response
workflow. It restores the JaQMC/FermiNet molecule checkpoint for the ground
state, builds direct antisymmetric FermiNet-style response heads, applies the
sample-estimated `Q0` projection against the trained ground state, and reads
excitation energies from the projected BF-NKSR spectrum peaks. The default
formal flow is residual-enrichment with held-out acceptance and moment
diagnostics; Ritz warmup is kept only as a candidate initializer and pole
audit inside that formal flow.

Current implementation notes from the late 2026-06-06 alignment pass:

- residual-enrichment training/holdout sampling supports the paper's adaptive
  positive mixture `Q` via `--enrichment-sampling mixture` (default), with
  `envelope` kept as a diagnostic option. For two-electron He-like
  source-envelope audits, `--enrichment-sampling source-envelope-sobol` and
  `--enrichment-sampling source-envelope-pz-sobol` are also available; both
  require positive `--response-source-envelope-core-decay` and
  `--response-source-envelope-diffuse-decay`.
- held-out acceptance now requires residual capture improvement, held-out
  enrichment-objective improvement, moment diagnostics, and an overlap
  condition cutoff.
- `--independent-heads --enrichment-candidate-heads < --n-heads` runs
  append-and-repeat residual-enrichment blocks, freezing accepted heads while
  training the next candidate block. This is closer to the paper workflow than
  one shared-backbone block.
- Nonfinite candidate attempts and invalid validation matrices are rejected
  and recorded instead of crashing the whole run.
- Output `.npz` files save `response_params/...`; use `--response-params
  previous_output.npz --training-flow none` to freeze trained heads and redo
  final matrix estimation with higher statistics or another sampler.
- When `--response-params` is used with `--training-flow none`, the saved
  `active_response_heads` value is restored automatically.
- Residual-enrichment candidate validation can use multiple independent
  heldouts via `--enrichment-ritz-validation-holdouts`; Ritz-seeded candidates
  are not allowed to bypass the paper residual gate. Candidate checkpoint
  selection is by held-out residual objective/capture; appending still requires
  held-out residual/objective improvement and moment/condition checks.
- Final matrix whitening has a tail-head fallback: if final high-statistics
  whitening fails, invalid trailing response heads are discarded and
  `trained_response_heads`/`final_matrix_retained_heads` record the distinction.
- `--response-source-envelope-init` initializes source-prefactor/orbital-bias
  heads to a source-envelope projection. For H with decay `0.5`, this exactly
  represents the `2p_z` shape `z exp(-r/2)`. For split-spin systems such as He,
  the orbital biases are spin-block identities; all-ones biases make the
  determinant rank deficient and the response head exactly zero.
- Residual-enrichment now has a source-bright checkpoint restore/gate:
  `--enrichment-source-bright-gate` validates held-out bright-pole stability and
  rejects large regressions of already-visible peaks. By default,
  `--enrichment-require-training-improvement` is true, so a candidate must
  improve on its own initialization on held-out residual diagnostics.
- Final Sobol matrix estimation supports `--final-sobol-replicas N`; this keeps
  the total `--final-samples` fixed but averages matrices over independent
  scrambled Sobol replicas. It is important for He, where a single scramble can
  produce false low roots.
- Final mixture matrix estimation supports `--final-mcmc-replicas N`; this
  concatenates independent adaptive-Q MCMC chains and then estimates the final
  `Q0` projection and weak matrices once from the combined samples. Use this
  instead of a single long chain for He diagnostics.
- `--final-sampling source-envelope-sobol` is available for two-electron
  core/diffuse source-envelope response heads. It stratifies
  `core(r1) diffuse(r2)` and `diffuse(r1) core(r2)` Sobol samples and uses the
  exact 50/50 mixture density. This is a He-oriented variance-reduction
  diagnostic for the orbital-z 1s2p-like ansatz, not a general all-molecule
  sampler yet.
- `--final-sampling source-envelope-pz-sobol` is the p-wave version of that
  two-electron sampler. The core electron uses the spherical
  `s^2 ~ exp(-2 a r)` density while the diffuse electron uses the normalized
  `p_z^2 ~ z^2 exp(-2 a r)` density. This better matches dipole-bright 1s2p
  He audits and lowered the 1M/16 He leave-one-out jackknife uncertainty from
  roughly `0.12 eV` with spherical source-envelope Sobol to about `0.029 eV`,
  but it still has not reached the `0.001 eV` target.
- 2026-06-07 paper-alignment update: formal residual-enrichment candidate
  selection now uses held-out residual objective/capture only. Source-bright
  pole diagnostics are gates/audits, not the checkpoint-selection objective;
  this keeps the formal path aligned with the paper's
  residual-candidate/held-out-acceptance loop. The old standalone `ritz`,
  `ritz-plus-residual`, `source-bright-ritz`, and `sequential-enrichment` CLI
  training branches were removed from the internal response entrypoint; use
  `--training-flow residual-enrichment` for training and `--training-flow none`
  only to audit frozen response parameters.
- 2026-06-07 stricter held-out residual acceptance: the formal CLI now exposes
  `--enrichment-validation-holdouts` (old
  `--enrichment-ritz-validation-holdouts` is only a compatibility alias) and
  `--enrichment-holdout-min-pass-fraction` (default `1.0`). Acceptance records
  and gates on worst held-out residual capture ratio, worst held-out objective
  delta, and residual pass fraction; median residual statistics alone cannot
  hide a failing validation set. Output NPZ/history fields include
  `enrichment_holdout_capture_ratio_min`,
  `enrichment_holdout_objective_delta_min`,
  `enrichment_holdout_pass_fraction`, `enrichment_holdout_pass_count`, and
  `enrichment_holdout_count`.
- 2026-06-07/08 strong residual audit update: the formal CLI now has the
  default-off Laplacian audit flags `--strong-residual-audit-samples`,
  `--strong-residual-audit-omegas`, `--strong-residual-audit-source-index`, and
  `--strong-residual-audit-batch-size`. Output NPZ files save
  `strong_residual_omegas`, `strong_residual_epsilon`,
  `strong_residual_epsilon_over_eta`, `strong_residual_norm`,
  `strong_residual_source_norm`, `strong_residual_eta`,
  `strong_residual_source_index`, `strong_residual_samples`,
  `strong_residual_max_epsilon`, and
  `strong_residual_max_epsilon_over_eta`. This follows the paper rule: weak
  form for training/matrices; pointwise Laplacian residual only for validation
  audits. Local and remote response tests passed after this update
  (`62 passed, 1 warning` on both). H smoke
  `h_iter67_strong_residual_smoke_65k.npz` rejected its candidate by the
  residual gate and fell back to source-only (`0.4982247468 Ha`,
  `strong_residual_epsilon=2.272`, `epsilon/eta=113.6`). Frozen audit
  `h_iter68_strong_residual_frozen_iter61_65k.npz` loaded the accurate
  `h_iter61` head and gave first pole `0.3750537861 Ha`, bootstrap SE
  `0.00150 eV`, and `strong_residual_epsilon=0.9059`
  (`epsilon/eta=45.3`). Treat strong residual as a stricter diagnostic than
  moments; do not claim convergence merely because `m0/m1` pass.
- 2026-06-07/08 enrichment strong-residual gate update: the formal CLI also
  has `--enrichment-strong-residual-samples`,
  `--enrichment-strong-residual-omegas`,
  `--enrichment-strong-residual-source-index`,
  `--enrichment-strong-residual-batch-size`, and
  `--enrichment-max-strong-residual-epsilon-over-eta`. This gate runs the
  Laplacian audit on held-out candidate-validation samples before appending a
  candidate. Output NPZ/history fields include
  `enrichment_strong_residual_epsilon_max`,
  `enrichment_strong_residual_epsilon_over_eta_max`,
  `enrichment_strong_residual_passed`, and corresponding
  `enrichment_history_*` arrays. Local tests passed
  (`63 passed, 1 warning`); remote A100 worker `3888253` tests passed
  (`63 passed, 1 warning`). H smoke
  `h_iter69_strong_gate_smoke_32k.npz` deliberately relaxed the weak residual
  improvement gate so the candidate passed held-out weak residuals
  (`holdout_pass=2/2`, ratio min `1.0385`) and moments, then correctly rejected
  it as `strong_residual_rejected` because held-out
  `epsilon/eta=121.38`. This is a gate-validation smoke, not an accuracy
  success; final matrices fell back to source-only (`0.4982869166 Ha`).
- 2026-06-07 validation after stricter held-out gate: local response tests
  passed (`57 passed, 1 warning`), remote worker `3888150` tests passed
  (`57 passed, 1 warning`), and an H atom formal mixture-Q smoke
  `h_smoke_paper_holdout_2x.npz` ran with
  `--enrichment-validation-holdouts 2`. The low-stat candidate was correctly
  rejected by the residual gate despite a small median improvement:
  `holdout_capture_ratio_min=1.00833156`, `holdout_pass=0/2`,
  `reason=residual_rejected`. For one-electron H, do not use
  `source-envelope-sobol` or `source-envelope-pz-sobol`; these samplers
  currently require `electron_shape=(2, 3)`.
- Final replica diagnostics now print and save both leave-one-block jackknife
  uncertainty (`final_replica_loo_jackknife_se` in Ha and
  `final_replica_loo_jackknife_se_ev` in eV) and paper-style block bootstrap
  uncertainty (`final_bootstrap_se` in Ha and `final_bootstrap_se_ev` in eV).
  Bootstrap resamples final Monte Carlo/Sobol blocks, rebuilds `(S,K,p)`, and
  reruns the projected spectrum map for each resample. Do not treat a peak as
  converged unless these uncertainties are below the requested tolerance.
- 2026-06-07 smoke after cleanup: remote response tests passed
  (`37 passed, 1 warning`) on worker `3887622`. A low-statistics H formal
  residual-enrichment smoke with analytic source-envelope decay
  `--initial-decay-min 0.5 --initial-decay-max 0.5` accepted one candidate with
  held-out residual reason `residual` and produced first pole
  `0.3749170547 Ha`; the leave-one-out jackknife SE was `1.114e-03 eV`. This is
  a workflow smoke, not a high-statistics accuracy claim.
- 2026-06-07 pz sampler He update:
  `he_iter73_pz_sourceenv_alpha2_orbz_jastrow_1M_16rep.npz` used frozen
  source-envelope/orbital-z/Jastrow alpha-2 response parameters with
  `--final-sampling source-envelope-pz-sobol`. It gave first pole
  `0.7806304070 Ha`, LOO mean `0.7806288314 Ha`, and LOO jackknife SE
  `2.888e-02 eV`; this is closer and less noisy than the old spherical
  source-envelope audit, but still high by about `0.024 eV` against the
  `~0.779747 Ha` He first singlet-P reference.
- 2026-06-07 formal pz-enrichment test:
  `he_iter77_formal_pz_enrich_pz_final_alpha2_epoch5_262k_8rep.npz` used
  `--enrichment-sampling source-envelope-pz-sobol` and
  `--final-sampling source-envelope-pz-sobol`. The residual gate accepted the
  4-head candidate (`capture_ratio=1.032226`, `pole_spread=4.982e-03 Ha`), but
  final first pole was `0.7805736911 Ha` with LOO jackknife SE
  `1.321e-01 eV`. This proves the formal pz sampling path works, not that He is
  converged.
- 2026-06-07/08 source-block/radial-prefactor iteration:
  - `--aux-source-gaussian-exponents ...` adds fixed one-body
    `p_z exp(-alpha r^2) Psi0` source probes to the projected basis while the
    physical dipole remains source channel 0. This implements the paper's
    source-design idea, but by itself it only weakly improved the frozen He
    pz-Sobol audit at 131k/8 replicas:
    `he_iter79_noaux_frozen_pz_131k_8rep.npz = 0.7832583257 Ha`,
    `he_iter79_auxsrc_frozen_pz_131k_8rep.npz = 0.7831356847 Ha`.
    Formal training with only auxiliary sources and no accepted heads
    (`he_iter80_auxsrc_formal_pz_epoch5_131k_8rep.npz`) fell back to aux-only
    final matrices and gave `0.8206179094 Ha`; do not treat aux sources alone
    as the He fix.
  - 2026-06-07/08 correlated auxiliary sources were added for multi-electron
    tests: `--aux-source-dipole-ee-scales` adds
    `sum_i z_i * mean_{i<j} r_ij/(s+r_ij) Psi0`, and
    `--aux-source-bond-odd-ee-slater-decays` plus
    `--aux-source-bond-odd-ee-scales` adds correlated two-center odd probes
    from the cross product of Slater decays and pair scales. They are included
    in source counts, matrix construction, residual enrichment, final samplers,
    NPZ metadata, and tests. H2 trials did not solve the first bright state:
    `h2_iter17_corr_ee_aux_forcedreject_65k.npz` landed at
    `0.5765504313 Ha`, close to the second bright reference
    (`~0.578603 Ha`) while missing the first (`~0.468023 Ha`), and
    `h2_iter18_atom_bond_dipoleee_forcedreject_65k.npz` worsened to
    `0.6483493800 Ha`. Treat these sources as experimental diagnostics, not a
    production H2 fix.
  - `--response-orbital-radial-powers N` adds trainable bounded radial powers
    `r/(scale+r)` on the p-like orbital column, initialized to zero so the
    source-envelope seed is unchanged. With `N=2`, the formal residual gate
    accepted the He candidate when source-bright spread tolerance was relaxed
    to `0.05 Ha`: `he_iter82_radial2_formal_pz_epoch5_gate005_131k_8rep.npz`
    gave first pole `0.7806810849 Ha`. A higher-stat frozen audit
    `he_iter83_radial2_frozen_pz_524k_16rep.npz` gave `0.7817620711 Ha`
    with bootstrap SE `4.949e-02 eV`; adding auxiliary sources on top gave
    `he_iter84_radial2_auxsrc_frozen_pz_524k_16rep.npz = 0.7815759306 Ha`
    with bootstrap SE `4.696e-02 eV`. This is directionally better than the
    131k no-radial frozen audit but still not within the `0.001 eV` target.
  - Adaptive mixture final sampling for the same radial response params,
    `he_iter85_radial2_frozen_mixture_262k_8rep.npz`, gave a low root
    `0.7369448549 Ha` but with bootstrap SE `4.907e-01 eV` and tiny source
    weight. Treat this as an unreliable ghost/noisy-sampling result, not an
    accuracy improvement.
  - `--final-sampling source-envelope-pz-jastrow-sobol` was added as a
    rejection-sampled proposal with density
    `q_base * exp(2 J_ee)` for the antiparallel `SimpleEEJastrow` form. It is
    paper-compatible as an unnormalized positive final sampler, but the first
    He audit was worse: `he_iter86_radial2_frozen_pzjastrow_a2_524k_16rep.npz`
    gave `0.7846170575 Ha` with bootstrap SE `9.682e-02 eV`. Do not use this
    sampler as the current He default.
  - Increasing pz Sobol block count without changing total samples improved
    the center but not the uncertainty:
    `he_iter87_radial2_frozen_pz_524k_64rep.npz` gave `0.7803742774 Ha`
    with bootstrap SE `5.070e-02 eV` and LOO range
    `0.7797854402..0.7813938273 Ha`. This is the closest radial2 frozen audit
    so far, about `0.017 eV` high versus the `~0.779747 Ha` reference, but
    still above the `0.001 eV` target and statistically unresolved.
  - `--response-orbital-radial-powers 4` did not help in the first screens.
    At LR `5e-4`, `he_iter88_radial4_formal_pz_epoch5_131k_8rep.npz` was
    rejected by the residual gate and fell back to source-only. At LR `1e-4`,
    `he_iter89_radial4_formal_pz_epoch10_lr1e4_131k_8rep.npz` was also
    rejected with best epoch at initialization. Prefer radial2 unless a better
    optimizer/objective is added.
  - `--response-envelope-decays` now allows explicit response-head envelope
    decays instead of only the linear `initial_decay_min/max` profile.
    `--response-orbital-radial-coeff-init` initializes the trainable bounded
    radial coefficients; accepted lengths are `radial_powers`,
    `n_heads*radial_powers`, or
    `n_heads*determinants_per_head*radial_powers`. These are ansatz
    initializers; coefficients remain trainable in residual enrichment.
  - The residual objective was corrected to train over all source-block
    channels when `--aux-source-gaussian-exponents` is used, matching the
    paper's `sum_{a l}` enrichment objective. Tests cover the single-source
    limit and multi-source capture. He screens were still not improved:
    `he_iter90_multisource_aux_radial2_formal_131k_8rep.npz` rejected a
    4-head block because source-bright pole spread was `8.930e-02 Ha` and
    fell back to aux-only final matrices (`0.8381905707 Ha`). A safer
    independent single-head sequence with stronger old-subspace penalty,
    `he_iter91_multisource_aux_radial2_seq1_lambda1_131k_8rep.npz`, accepted
    two heads but final first pole was `0.8035869358 Ha`. Current conclusion:
    the multi-source objective is now formally correct, but auxiliary sources
    are not the current He accuracy path without a better source family or
    optimizer.
  - `--final-sampling source-envelope-pz-radial-sobol` was added to sample the
    pz source-envelope proposal reweighted by the trained response radial
    prefactor squared. It is ansatz-matched, but the first He audit was worse
    than ordinary pz Sobol:
    `he_iter92_radial2_frozen_pzradial_524k_64rep.npz = 0.7818458428 Ha`
    with bootstrap SE `4.888e-02 eV`, compared with ordinary pz Sobol
    `he_iter87_radial2_frozen_pz_524k_64rep.npz = 0.7803742774 Ha`.
    Do not use radial-matched final sampling as the He default.
  - `--response-en-jastrow` adds an optional direct-response electron-nuclear
    Pade/Jastrow factor, initialized to zero by default so old response heads
    are unchanged. This is paper-compatible with the direct response
    form-domain/cusp requirement, but it is not a validated He improvement yet.
    `he_iter93_enjastrow_radial2_formal_pz_131k_8rep.npz` rejected the
    candidate by the held-out residual gate and fell back to source-only
    (`1.3017045877 Ha`). A 3-attempt screen,
    `he_iter94_enjastrow_radial2_attempts3_formal_pz_131k_8rep.npz`, also
    rejected all candidates; one attempt had small held-out residual
    improvement but failed source-bright stability (`pole_spread=5.767e-02 Ha`
    to `6.679e-02 Ha`). Do not use EN-Jastrow as the current He default without
    a better optimizer/stability treatment.
  - `--final-sampling source-envelope-pz-sobol-antithetic` adds global
    inversion pairs to the pz source-envelope Sobol final sampler. For the
    current radial2 frozen He response parameters it is the best final sampler
    screen so far:
    `he_iter95_radial2_frozen_pz_antithetic_524k_64rep.npz` gave
    `0.7796274022 Ha`, about `-0.0033 eV` against the
    `~0.7797471179 Ha` He first singlet-P reference. This improves the center
    relative to ordinary pz Sobol (`0.7803742774 Ha`, about `+0.017 eV`), but
    it is still outside the `0.001 eV` target and its block/bootstrap
    uncertainty remains large (`bootstrap_se_ev=9.754e-02`), so it is not a
    validated final answer.
    A higher-statistics continuation with the same trained response params,
    `he_iter102_radial2_frozen_pz_antithetic_1M_128rep.npz`, gave
    `0.7797697346 Ha`. Against the NIST first singlet-P excitation
    `21.2180230218 eV` (`0.7797479640 Ha`), this center is high by
    `5.92e-04 eV`, meeting the `0.001 eV` center-error target for this He
    audit. Cutoff sensitivity is flat from `1e-10` through `1e-7`; more
    aggressive whitening cutoffs raise the pole. Important caveat:
    block/bootstrap uncertainty remains much larger than `0.001 eV`
    (`bootstrap_se_ev=5.764e-02`, `loo_jackknife_se_ev=5.710e-02`), so this is
    a center-value success, not yet a rigorous small-error-bar validation.
  - A decay scan with `source-envelope-pz-sobol-antithetic` at `131k/16rep`
    showed that small independent Sobol-block estimates can be badly biased:
    baseline `he_iter99_23_065... = 0.7695944298 Ha`,
    `he_iter99_20_055... = 0.7694014622 Ha`, and
    `he_iter99_23_075... = 0.7692789472 Ha`. Do not tune decays from these
    small runs alone.
  - `--final-sampling source-envelope-pz-sobol-antithetic-global` draws one
    global Sobol base sequence and then cuts antithetic blocks for diagnostics.
    It improved the small `131k/16rep` baseline to
    `he_iter100_23_065... = 0.7781980307 Ha`, confirming that independent
    small Sobol scrambles were a variance/bias source, but its 524k result was
    worse than ordinary antithetic:
    `he_iter101_23_065... = 0.7809068526 Ha`. Keep it as a diagnostic, not the
    current He default.
  - `--final-sampling source-envelope-pz-sobol-symmetrized` extends the
    antithetic idea to the full `R, -R, swap(R), -swap(R)` orbit. At fixed total
    sample count it was worse, likely because it quarters the number of
    independent Sobol base points:
    `he_iter96_radial2_frozen_pz_symmetrized_524k_64rep.npz` gave
    `0.7759649415 Ha`. Do not use the symmetrized sampler as the He default.
  - `--response-spatial-parity odd|even` projects direct response heads onto
    charge-center inversion parity. This is useful for atom/centrosymmetric
    audits but is not a general molecular default. Applying odd parity to the
    existing radial2 frozen He parameters with the antithetic final sampler,
    `he_iter97_radial2_oddparity_frozen_pz_antithetic_524k_64rep.npz`, gave
    `0.7796093486 Ha`, essentially the same as the unprojected antithetic
    result and still about `-0.0037 eV` from reference. Retraining with odd
    parity passed the held-out residual gate, but the final pole was much too
    low: `he_iter98_oddparity_radial2_formal_pz_antithetic_131k_8rep.npz`
    gave `0.7741970636 Ha`. Do not treat odd-parity retraining as an accuracy
    improvement without further objective/optimizer changes.

Recommended H atom validation command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10380 fdbd:dc03:16:214::86 'cd /opt/tiger/jaqmc && \
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32 \
    /tmp/jaqmc_venv/bin/python scripts/ferminet_bfnksr_response.py \
      --checkpoint /opt/tiger/jaqmc/runs/hydrogen_ferminet_formal/train_ckpt_000799.npz \
      --ground-energy -0.4999999 \
      --output /opt/tiger/jaqmc/runs/hydrogen_ferminet_formal/h_bfnksr_formal_weighted_rough.npz \
      --seed 181 --n-heads 4 --hidden 20 --hidden-double 4 --layers 2 \
      --determinants-per-head 1 --training-flow residual-enrichment \
      --enrichment-candidate-attempts 3 --enrichment-attempt-lr-decay 0.5 \
      --epochs 500 --train-samples 65536 --enrichment-holdout-samples 16384 \
      --train-batch-size 4096 --learning-rate 0.001 \
      --residual-omegas 0.30 0.34 0.375 0.40 0.445 \
      --residual-omega-weights 4 6 20 6 1 \
      --residual-eta 0.025 --enrichment-lambda-rough 0.10 \
      --final-samples 150000 --final-walkers 4096 --burn-in 500 --steps-between 12 \
      --mcmc-width 1.2 --matrix-batch-size 1024 \
      --envelope-decay 0.32 --initial-decay-min 0.12 --initial-decay-max 0.75 \
      --eta 0.004 --omega-min 0.25 --omega-max 0.65 --grid-size 4001 \
      --log-every 100'"
```

Known H response results from 2026-06-06 with the formal residual-enrichment flow:

- direct block, equal frequency weights:
  `h_iter15_paper_enrichment_attempts.npz`, first pole `0.4369215876 Ha`.
- sequential independent heads:
  `h_iter16_paper_sequential_attempts.npz`, first pole `0.4381054772 Ha`.
- explicit frequency weights:
  `h_iter17_paper_weighted_enrichment.npz`, first pole `0.4471710706 Ha`.
- frequency weights plus `--enrichment-lambda-rough 0.10`:
  `h_iter18_paper_weighted_rough.npz`, first pole `0.4113522572 Ha`.
- frequency weights plus `--enrichment-lambda-rough 0.50`:
  `h_iter19_paper_weighted_rough05.npz`, first pole `0.4252736599 Ha`.
- Moment checks passed in all runs, but the formal residual-enrichment flow
  still misses the exact H `n=2` pole (`0.3750000000 Ha`) and tends toward the
  `n=3` region (`0.4444444444 Ha`). Treat this as an open convergence/training
  issue, not a validated accuracy result.
- held-out Ritz restore with source-prefactor heads and default mixture final
  sampling improved the first pole but remained noisy:
  `h_iter25_holdout_ritz_prefactor.npz` gave `0.3747256985 Ha` with
  300k final samples; `h_iter28_holdout_ritz_prefactor_1Msamples.npz` gave
  `0.3755644153 Ha` with 1M final samples.
- envelope IID and Sobol final estimators were added as explicit
  `--final-sampling envelope` and `--final-sampling sobol-envelope` diagnostics.
  For H, `h_iter33_holdout_ritz_prefactor_sobol_final.npz` and
  `h_iter34_holdout_ritz_prefactor_sobol_4M.npz` converged to about
  `0.37515826 Ha`, an error of about `0.00431 eV`; this is better than the
  residual-enrichment flow but still above the `0.001 eV` target.
- A float32 continuation of the H ground checkpoint to
  `/opt/tiger/jaqmc/runs/hydrogen_ferminet_precision/train_ckpt_001599.npz`
  did not change the response result:
  `h_iter36_precision_ckpt_ritz_sobol.npz` gave `0.3751863149 Ha`.
- After adaptive-mixture training and stricter paper-style acceptance were
  added, formal residual-enrichment still did not reach H accuracy:
  `h_iter38_paper_mixture_enrichment.npz` (shared 4-head block) gave first pole
  `0.4727759230 Ha`; `h_iter40_paper_iterative_independent_condition1e6.npz`
  (independent 2-head append blocks, `--enrichment-max-overlap-condition 1e6`)
  gave `0.4240883837 Ha` with mixture final sampling and
  `0.4248331264 Ha` in the Sobol frozen-head audit; and
  `h_iter41_paper_singlehead_condition1e5_sobol.npz` (single-head append,
  `--enrichment-lambda-rough 0.1`, condition `1e5`) gave `0.4520791239 Ha`.
  These are all high versus exact H `n=2`, `0.3750000000 Ha`, so the residual
  objective/candidate generation remains the open issue. Do not present the
  formal residual flow as numerically validated yet.
- With the paper residual gate tightened so Ritz no longer bypasses residual
  acceptance, H still exposes a training-objective problem:
  `h_iter51_paper_residual_gate_3heads_sobol.npz` gave first pole
  `0.3755562370 Ha` (error about `0.0151 eV`) and
  `h_iter52_paper_residual_gate_1head_sobol.npz` gave `0.3755108519 Ha`
  (error about `0.0139 eV`). Moment diagnostics passed, so moments alone are
  not sufficient to guarantee peak accuracy.
- Source-envelope initialization validates the ansatz/weak-form matrix path:
  `h_iter53_source_envelope_init_none_sobol.npz` gave `0.3749999244 Ha`, and
  Ritz training from that initialization restored epoch 0 in
  `h_iter54_source_envelope_init_ritz_sobol.npz`, giving `0.3749999491 Ha`.
  These are within the `0.001 eV` target for H, but this is a seeded Ritz/ansatz
  validation, not the plain residual-enrichment convergence result.
- With source-bright restore/gate and absolute candidate-capture acceptance,
  the formal residual-enrichment path can keep the source-envelope seed:
  `h_iter55_source_envelope_residual_gate_restore_sobol.npz` gave
  `0.3749999757 Ha`, an error of about `6.6e-7 eV` against exact H
  `0.3750000000 Ha`. This validates the H seeded formal path.
- H regression after source-bright/core-diffuse/Sobol-replica changes:
  `h_iter56_regression_source_envelope_4rep.npz` used 4 Sobol replicas with
  total 1M samples and gave `0.3749998696 Ha`, still within the `0.001 eV`
  target.
- After the 2026-06-07 paper-aligned candidate-selection change, H requires the
  source-envelope decay to be fixed to the analytic 2p value. With
  `--initial-decay-min 0.5 --initial-decay-max 0.5`, the formal
  residual-enrichment path accepts the trained candidate and remains accurate
  for short training:
  `h_iter60_decay05_epoch_1_sobol262k.npz` gave `0.3750003816 Ha`,
  `h_iter60_decay05_epoch_5_sobol262k.npz` gave `0.3750184182 Ha`, and the
  higher-statistics confirmation `h_iter61_decay05_epoch5_sobol1M_4rep.npz`
  gave `0.3750089879 Ha` (about `0.000245 eV` high versus exact
  `0.3750000000 Ha`). Longer residual training drifts: epoch 10 gave
  `0.3750688591 Ha` and epoch 100 gave `0.3763580281 Ha`. Do not use the old
  broad H initialization range `0.12..0.75` for accuracy regressions; it is not
  the analytic 2p source-envelope seed and gives peaks around `0.384 Ha`.
- H multi-peak audits from 2026-06-08 show that a single learned response head
  is insufficient beyond the first bright pole:
  `h_iter61_decay05_epoch5_sobol1M_4rep.npz` retains only two roots,
  `0.3750089879 Ha` and a spurious `0.6504153724 Ha`.
  Adding six fixed Gaussian auxiliary source probes
  (`h_iter62_auxsrc_highpeaks_frozen_262k_16rep.npz`) keeps the first pole
  accurate and moves the next root to `0.4603476526 Ha`, but it still does not
  resolve the exact H `n=3` root (`0.4444444444 Ha`) or `n=4`
  (`0.4687500000 Ha`). The source-block probes are too tied to `Psi0` for
  Rydberg p states.
- Direct Slater-type response heads are the right H high-bright-state path.
  A frozen source-prefactor decay ladder,
  `h_iter63_direct_decay_ladder_frozen_524k_16rep.npz`, gives
  `n=2 = 0.3749991059 Ha` and `n=3 = 0.4444420904 Ha`, both within
  `0.001 eV`, while `n=4 = 0.4698610237 Ha` is still high by about
  `0.030 eV`. A 6-head orbital-z/radial-ladder audit,
  `h_iter64_orbitalz_radial_ladder_frozen_524k_16rep.npz`, improves
  `n=4` to `0.4689188339 Ha` but still misses by about `0.0046 eV`; the
  radial coefficients are zero in frozen mode, so the improvement is mainly
  from the larger decay ladder.
- The best H multi-peak audit so far is the 10-head direct source-prefactor
  decay ladder:
  `h_iter65_direct_decay_ladder10_frozen_524k_16rep.npz` gives
  `n=2 = 0.3750036981 Ha`, `n=3 = 0.4444484652 Ha`, and
  `n=4 = 0.4687506444 Ha`. Errors versus exact H
  `0.3750000000`, `0.4444444444`, and `0.4687500000 Ha` are about
  `1.0e-04 eV`, `1.1e-04 eV`, and `1.8e-05 eV`, respectively. This validates
  the direct response Slater-ladder idea for bright states well below the
  ionization limit. A more diffuse 14-head ladder,
  `h_iter66_direct_decay_ladder14_frozen_524k_16rep.npz`, worsened moment
  diagnostics (`m1_rel=1.202e-02`) and shifted the first pole to
  `0.3751288296 Ha`; do not use it as the default. Near-threshold H `n>=5`
  likely needs a dedicated Rydberg/continuum treatment rather than simply more
  diffuse heads.
- 2026-06-07 radial auxiliary source continuation on worker `3888272`
  (`ssh -p 10384 fdbd:dc03:16:323::150`, `/opt/tiger/jaqmc/.venv-gpu`):
  local and remote checks passed (`ruff`, `tests/response`: 65 passed).
  New H fixed-source smokes using the trained `h_iter61` head show the code path
  works but the family does not solve high peaks by itself:
  `h_iter70_radial_auxsrc_scale1_frozen_262k_16rep.npz` roots
  `0.3750054076, 0.4876420024, 0.8351704602 Ha`;
  `h_iter71_radial_auxsrc_scale4_frozen_131k_8rep.npz` roots
  `0.3750071135, 0.4677343597, 0.6609622070 Ha`;
  combined Gaussian+radial
  `h_iter72_gauss_radial_auxsrc_scale4_frozen_131k_8rep.npz` roots
  `0.3750012134, 0.4608078353, 0.5862633327 Ha`. Exact H bright roots are
  `n=2 0.3750000000`, `n=3 0.4444444444`, `n=4 0.4687500000 Ha`; the radial
  source block preserves the first peak but still misses `n=3`, confirming that
  fixed source probes tied to `Psi0` are not enough for H Rydberg peaks. Keep
  the direct Slater response-head ladder as the H high-peak validation path.
- 2026-06-07 explicit-decay response-head continuation on worker `3888303`
  (`ssh -p 9791 fdbd:dc03:9:339::138`): local checks passed (`ruff`,
  `tests/response`: 71 passed), remote checks passed (`ruff`,
  `tests/response`: 70 passed before the final response-flow label fix).
  `h_iter73_exactdecay4_frozen_262k_16rep.npz` used explicit H decays
  `1/2,1/3,1/4,1/5` and gave roots
  `0.3750050423, 0.4444666522, 0.4689890877, 0.5021388033 Ha`;
  n=2/n=3 were within `0.001 eV`, n=4 and n=5 were not. Radial-polynomial
  initializers were stable but did not solve all roots:
  `h_iter74_exactdecay_radialpoly4_frozen_262k_16rep.npz` gave
  `0.3750015427, 0.4444452040, 0.4690654050, 0.4831589720 Ha`;
  `h_iter75_exactdecay_radialpoly4_scale40_frozen_262k_16rep.npz` gave
  `0.3750028415, 0.4444470148, 0.4678659765, 0.4808979537 Ha`. The best
  new frozen audit is `h_iter76_explicitdecay10_frozen_262k_16rep.npz`, using
  explicit decays `0.5,0.4,1/3,2/7,0.25,2/9,0.2,2/11,1/6,2/13`; roots were
  `0.3750009416, 0.4444469789, 0.4687418819, 0.4807101360 Ha`. Errors versus
  exact H n=2/n=3/n=4 are `+0.000026`, `+0.000069`, and `-0.000221 eV`;
  n=5 is still high by `+0.0193 eV`, consistent with needing a more dedicated
  near-threshold/Rydberg treatment. Also fixed `response_flow_name` so
  `--training-flow none` initialized-head audits are labeled as frozen, even
  without `--response-params`.
- 2026-06-07 multi-root diagnostics update on worker `3888332`
  (`ssh -p 10529 fdbd:dc03:9:339::138`, `/opt/tiger/jaqmc/.venv-gpu`):
  `final_replica_pole_diagnostics` now records source-bright root vectors, not
  only the first bright pole. New NPZ arrays include
  `final_replica_block_bright_poles`, `final_replica_loo_bright_poles`,
  `final_replica_loo_root_*`, `final_bootstrap_bright_poles`, and
  `final_bootstrap_root_*`; CLI option `--final-diagnostic-roots` controls how
  many bright roots are tracked. This aligns the bootstrap/LOO diagnostics with
  the paper's whole-map `(S,K,p)->spectrum` uncertainty recommendation.
  Local checks passed (`ruff`, `tests/response`: 73 passed); remote checks
  passed (`ruff`, `tests/response`: 73 passed).
  H diagnostic reruns showed why the new multi-root output is necessary:
  `h_iter77_multirootdiag_ladder10_131k_8rep.npz` wrote the new arrays but had
  large high-root uncertainty and biased roots
  `0.3750029293, 0.4435939407, 0.4659323004 Ha`. Higher statistics at
  `overlap_cutoff=1e-10`,
  `h_iter78_multirootdiag_ladder10_524k_16rep.npz`, still retained a noisy
  overlap direction (`condition=2.112e8`) and gave
  `0.3749996069, 0.4442285368, 0.4653539248 Ha`. Reanalyzing the older good
  `h_iter76_explicitdecay10_frozen_262k_16rep.npz` showed that
  `overlap_cutoff=1e-8` discards the bad direction and recovers
  `0.3750009416, 0.4444469789, 0.4687418819 Ha`; the same file at `1e-10`
  retains one extra direction with condition `7.374e8`. New cut-`1e-8` reruns
  (`h_iter80_multirootdiag_ladder10_cut1e8_262k_16rep.npz`,
  `h_iter81_multirootdiag_ladder10_seed176_cut1e8_262k_16rep.npz`) confirmed
  that n=2 is statistically tight but n=3/n=4 remain Sobol-seed sensitive at
  this sample size. Do not claim high-root completion from a single moderate
  frozen audit; use multi-root bootstrap/LOO and cutoff sensitivity.
- 2026-06-07 projection-aware final bootstrap update on worker `3888360`
  (`ssh -p 10561 fdbd:dc03:16:339::150`): the final block estimator now saves
  unprojected weak matrices plus per-block `Q0` projection moments, so
  leave-one-out/bootstrap resampling recomputes the ground projection for each
  resampled `(S,K,p)` estimator instead of reusing the all-sample projected
  block matrices. New output fields include
  `final_replica_projection_resampling=True`,
  `final_replica_raw_overlaps`, `final_replica_raw_hamiltonians`,
  `final_replica_projection_numerators`, `final_replica_projection_norms`,
  `final_replica_ground_hamiltonians`, and
  `final_replica_ground_hamiltonian_norms`. This is closer to the paper's
  full-pipeline bootstrap. Local checks passed (`ruff`, `tests/response`:
  74 passed); remote checks passed (`ruff`, `tests/response`: 74 passed).
  H smoke `h_iter82_projectionaware_ladder10_smoke_65k_4rep.npz` ran the
  ladder with `overlap_cutoff=1e-8`, logged
  `projection_resampling=True`, and saved readable multi-root diagnostics:
  first root `0.3749980601 Ha`, LOO root means
  `0.3749980536,0.4447075252,0.4641759056,0.4955524088 Ha`, bootstrap root
  means `0.3749980974,0.4446510168,0.4642879099,0.4977250773 Ha`. This is a
  smoke test for the new diagnostic path, not an accuracy claim for high roots
  because `n=3/n=4` uncertainties remain much larger than `0.001 eV`.
- 2026-06-07 cutoff-sensitivity output update on worker `3888372`
  (`ssh -p 10979 fdbd:dc03:16:339::150`): final runs now save an explicit
  whitening cutoff scan controlled by `--cutoff-diagnostic-values`; the active
  `--overlap-cutoff` is always included. New NPZ arrays include
  `cutoff_diagnostic_values`, `cutoff_diagnostic_success`,
  `cutoff_diagnostic_retained`, `cutoff_diagnostic_condition`,
  `cutoff_diagnostic_moment_*`, `cutoff_diagnostic_bright_roots`,
  `cutoff_diagnostic_bright_norm_weights`, and
  `cutoff_diagnostic_bright_root_spread_ev`. This makes the overlap-whitening
  stability check part of the formal output instead of an offline-only
  analysis. Local checks passed (`ruff`, `tests/response`: 75 passed); remote
  checks passed (`ruff`, `tests/response`: 75 passed). H smoke
  `h_iter83_cutoffdiag_ladder10_smoke_65k_4rep.npz` confirmed saved fields and
  printed cutoff spread across `1e-10,1e-9,1e-8,1e-7,1e-6`:
  root spreads were `4.853e-04,3.969e-03,3.205e-03,1.181e-01 eV`. This shows
  root 0 is stable while higher roots are not at 65k/4 replicas, so use the
  saved cutoff diagnostics together with projection-aware bootstrap before
  claiming any high-root `<0.001 eV` result.
- 2026-06-07/08 retrofit raw-block diagnostics update: use
  `scripts/retrofit_bfnksr_raw_blocks.py` to recompute paper-style raw
  final-block diagnostics from older output `.npz` files that saved `samples`
  and `response_params/*` but not `final_replica_raw_*`. The tool now rebuilds
  densities for `source-envelope-sobol`, `source-envelope-pz-sobol*`,
  `source-envelope-spz-sobol*`, `one-electron-pz-envelope-mixture-sobol*`, and
  `sobol-envelope*` (the last requires `--envelope-decay`), restores saved
  auxiliary source probes, saves the same final replica/LOO/bootstrap/cutoff
  field set as the formal run, and optionally runs the strong residual audit
  through `--strong-residual-audit-samples`. Local checks on 2026-06-07 passed
  (`ruff`, `tests/response`: 91 passed). Workspace-host checks passed for the
  new retrofit test file (`ruff`; `.venv-gpu` pytest: 3 passed). No GPU
  numerical retrofit was run in this pass because `mlx worker list` was empty.
  When a GPU worker is available, retrofit the old He 1M pz-antithetic audit
  before making any `<0.001 eV` claim:

```bash
python scripts/retrofit_bfnksr_raw_blocks.py \
  --input runs/helium_ferminet_formal/he_iter102_radial2_frozen_pz_antithetic_1M_128rep.npz \
  --checkpoint runs/helium_ferminet_formal/train_ckpt_000399.npz \
  --ground-energy -2.9038051867 \
  --output runs/helium_ferminet_formal/he_iter102_retrofit_rawblocks_current.npz \
  --n-heads 4 --hidden 16 --hidden-double 4 --layers 2 \
  --response-orbital-z-prefactor --response-orbital-radial-powers 2 \
  --response-orbital-radial-scale 1.0 --response-orbital-bias \
  --response-ee-jastrow --response-jastrow-alpha-init 2.0 \
  --opposite-spin-symmetry singlet \
  --matrix-batch-size 8192 --overlap-cutoff 1e-8 \
  --cutoff-diagnostic-values 1e-10 1e-8 1e-7 1e-6 \
  --diagnostic-roots 4 --bootstrap-replicates 200
```

Frozen-head final audit pattern:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10380 fdbd:dc03:16:214::86 'cd /opt/tiger/jaqmc && \
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32 \
    /tmp/jaqmc_venv/bin/python scripts/ferminet_bfnksr_response.py \
      --checkpoint /opt/tiger/jaqmc/runs/hydrogen_ferminet_formal/train_ckpt_000799.npz \
      --response-params /opt/tiger/jaqmc/runs/hydrogen_ferminet_formal/h_iter40_paper_iterative_independent_condition1e6.npz \
      --ground-energy -0.4999999 \
      --output /opt/tiger/jaqmc/runs/hydrogen_ferminet_formal/h_iter40_sobol_audit.npz \
      --seed 402 --n-heads 8 --hidden 16 --hidden-double 4 --layers 2 \
      --determinants-per-head 1 --independent-heads --source-prefactor \
      --training-flow none --final-sampling sobol-envelope --final-samples 1048576 \
      --matrix-batch-size 2048 --envelope-decay 0.32 \
      --eta 0.002 --omega-min 0.34 --omega-max 0.50 --grid-size 4001 \
      --overlap-cutoff 1e-8'"
```

Recommended He smoke/diagnostic command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 10380 fdbd:dc03:16:214::86 'cd /opt/tiger/jaqmc && \
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32 \
    /tmp/jaqmc_venv/bin/python scripts/ferminet_bfnksr_response.py \
      --checkpoint /opt/tiger/jaqmc/runs/helium_ferminet_formal/train_ckpt_000399.npz \
      --ground-energy -2.9038051867 \
      --output /opt/tiger/jaqmc/runs/helium_ferminet_formal/he_bfnksr_formal_jastrow_singlet.npz \
      --seed 101 --n-heads 4 --hidden 20 --hidden-double 6 --layers 2 \
      --determinants-per-head 1 --response-ee-jastrow --opposite-spin-symmetry singlet \
      --training-flow residual-enrichment \
      --enrichment-candidate-attempts 3 --enrichment-attempt-lr-decay 0.5 \
      --epochs 600 --train-samples 65536 --enrichment-holdout-samples 16384 \
      --train-batch-size 4096 --learning-rate 0.001 \
      --residual-omegas 0.60 0.75 0.90 1.05 1.20 --residual-eta 0.03 \
      --final-samples 100000 --final-walkers 4096 --burn-in 500 --steps-between 12 \
      --mcmc-width 0.45 --matrix-batch-size 2048 \
      --envelope-decay 0.8 --initial-decay-min 0.20 --initial-decay-max 2.00 \
      --eta 0.005 --omega-min 0.4 --omega-max 2.0 --grid-size 4001 \
      --log-every 200'"
```

Known He result from 2026-06-06 with the residual-enrichment flow:

- candidate block accepted with held-out capture ratio `379.070087`
- moment checks passed: `m0_rel=1.096e-15`, `m1_rel=5.103e-15`
- first pole was `1.0913116029 Ha`, still high versus the first singlet
  dipole reference of about `0.7797471179 Ha`
- with EE Jastrow, singlet opposite-spin symmetry, candidate attempts, and
  weighted residual grid:
  `he_iter20_paper_jastrow_singlet_attempts.npz`, first pole
  `1.0712259824 Ha`; moment checks passed (`m0_rel=5.604e-16`,
  `m1_rel=1.750e-14`), overlap condition `1.354e6`. This is still high by
  about `0.291479 Ha` (`7.93 eV`) versus `0.7797471179 Ha`.
- 2026-06-07 He diagnostics:
  - `he_iter21_scan_decay_*.npz` exposed a bug in source-envelope init: split
    spin orbital biases set to all ones made the response head zero, so every
    decay returned the source-only pole `1.2017774439 Ha`.
  - After the spin-block identity fix, `he_iter22_scan_decay_*.npz` retained
    the response head, but one product-envelope source-prefactor head still gave
    first bright poles around `1.20-1.24 Ha`; this seed is not sufficient for
    He.
  - Moving the residual grid to the He first bright region
    (`0.70 0.76 0.80 0.86 0.94 Ha`) did not solve He:
    `he_iter23_residual_grid078_gate_sobol.npz` rejected the candidate and
    fell back to source-only (`1.3024871624 Ha`), while
    `he_iter24_residual_capture_grid078_sobol.npz` accepted a candidate but
    still gave `1.3535133546 Ha`.
  - Pure Ritz can create low roots but they are not reliable bright peaks:
    `he_iter25_ritz_diffuse_repro_mixture.npz` had a low root
    `0.7353002551 Ha` with tiny weight `2.22e-4`, consistent with a dark/ghost
    root rather than a validated bright excitation.
  - Source-bright Ritz remained high: envelope-trained
    `he_iter26_source_bright_ritz_grid078_mixture.npz` gave `1.1410060645 Ha`;
    mixture-trained `he_iter27_source_bright_ritz_mixture_train.npz` was worse,
    with bright roots around `2.2 Ha`. He is not within the `0.001 eV` target.
  - Soft-bright training plus Sobol validation did not solve He:
    `he_iter28_softbright_sobol_validate.npz` gave `1.2325137854 Ha`.
  - Adding explicit core/diffuse source-envelope initialization was a useful
    ansatz improvement. A 4-head profile seed with `core=2.0`, `diffuse=0.4`,
    and profile from `initial_decay_max=1.3` to `initial_decay_min=0.7` gave
    `he_iter33_corediff_profile_4heads_1M.npz`, first pole `0.8526350911 Ha`.
    Training from this seed with soft-bright Ritz improved to
    `he_iter35_corediff_softbright_train_sobol.npz`, final first pole
    `0.8376171525 Ha`; the 4-replica frozen audit
    `he_iter35_frozen_4rep_audit.npz` gave `0.8247679885 Ha`.
  - Single scrambled Sobol estimates can be misleading for He: with the same
    profile family, some single-seed 1M runs bracketed the reference
    (`he_iter38_profile_core2p3_diff_0.35_1M.npz` gave `0.7645999093 Ha`), but
    4-replica matrix averaging moved the same region back high:
    `he_iter40_profile_core2p3_diff_0.35_1M_4rep.npz` gave `0.9311481805 Ha`
    and `he_iter40_profile_core2p3_diff_0.39_1M_4rep.npz` gave `0.9115785174 Ha`.
    Treat single-Sobol low He roots as unvalidated until replica/bootstrap
    uncertainty is small.
  - Orbital-z prefactor with source-envelope spin-block identities gives a much
    more physical He 1s2p-like response head than a global source prefactor.
    With no response training and 1M/4-replica Sobol final matrices, scanning
    diffuse decay at core `2.30` gave
    `he_iter42_orbz_core2p3_diff_0.55_1M_4rep.npz = 0.7876068622 Ha`,
    `diff_0.65 = 0.7871279335 Ha`, and
    `diff_0.80 = 0.7961104598 Ha` in the seed-642 run. A later seed-643 core
    scan at diffuse `0.65` was less favorable:
    core `1.60/1.80/2.00/2.30/2.60/3.00` gave
    `0.8221087166/0.8074018210/0.7995083867/0.7962260425/0.7998417335/0.8107176809 Ha`.
    This confirms He still has substantial final-matrix/sampling sensitivity.
  - Adding response EE Jastrow to orbital-z heads improves the seed-644 Sobol
    audit at core `2.30`, diffuse `0.65`: alpha
    `0.25/0.50/1.00` gave
    `0.7873718970/0.7867075086/0.7856622209 Ha`. A seed-645 run gave
    alpha `2.00 = 0.7784038722 Ha` and `4.00 = 0.7803093288 Ha`, but the
    same region was not stable under seed-646 Sobol (`alpha 4.00 =
    0.7914976331 Ha`) or adaptive-mixture final audits
    (`alpha 2.00 = 0.7966843569 Ha`, `alpha 4.00 = 0.8177790000 Ha`).
    Current reliable conclusion: the orbital-z + EE-Jastrow ansatz is the best
    He direction so far, but He is not validated to the `0.001 eV` target.
  - 2026-06-07/08 final-sampling diagnostics:
    * The final Sobol replica path was corrected to concatenate all Sobol
      replicas and estimate the `Q0` projection once globally, rather than
      averaging already-projected per-replica matrices.
    * Antithetic Sobol and non-scrambled Sobol were tested and should not be
      treated as validated He fixes. Antithetic source-envelope samples gave
      high poles around `0.816-0.823 Ha`; non-scrambled Sobol can hit Coulomb
      singular zero-measure points and was not kept as a CLI mode.
    * Single-chain adaptive mixture audits are too noisy for He. With alpha 2.0
      and orbital-z/Jastrow/source-envelope heads, 1M total samples over
      8 independent mixture chains gave
      `he_iter55_mix8_alpha2_orbz_jastrow_1M.npz = 0.7751604338 Ha`,
      `he_iter58_mix8_alpha2p0_seed657_orbz_jastrow_1M.npz = 0.7808729637 Ha`,
      and `he_iter59_mix8_alpha2p0_seed659_orbz_jastrow_1M.npz = 0.7865612648 Ha`.
      This seed spread is much larger than `0.001 eV`; do not claim He is
      converged from mixture sampling yet.
    * The He source-envelope Sobol sampler is faster and better matched to the
      1s2p-like ansatz, but still needs replica/seed uncertainty. Alpha 2.0
      with 1M/4 scrambled replicas gave `0.7766606793 Ha` for seed 661 and
      `0.7844648366 Ha` for seed 662; 1M/16 replicas gave
      `he_iter64_sourceenv_sobol16_alpha2_orbz_jastrow_1M.npz =
      0.7807798728 Ha`. Nearby alphas with 1M/16 replicas were
      alpha 1.985: `0.7813941729 Ha`, alpha 1.95: `0.7827279813 Ha`,
      alpha 1.5: `0.7841544605 Ha`. Current best reliable He error is still
      around `0.03 eV` in favorable 16-replica source-envelope Sobol audits,
      not the `0.001 eV` target.

For old Ritz comparison runs, keep `--ritz-restore-root-floor` available as a
diagnostic guard. It prevents best-checkpoint restore from selecting near-zero
or negative ground-contaminated Ritz roots.

## BF-NKSR H2 Action-Oracle Update, 2026-06-13

- Current remote GPU worker after the user switched workers:
  `mlx worker list` shows worker `3903266`, A100-SXM-80GB,
  pod IP `fdbd:dc03:16:193::28`, port `9723`. Nested worker SSH from
  workspace host:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 \
  128129.zhang-xiaoyu.ws@ssh-candy-lq.workspace.byted.org \
  "ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=12 \
   -o StrictHostKeyChecking=no -p 9723 fdbd:dc03:16:193::28 '...'"
```

- Local and remote tests after the action-oracle update passed:
  local `ruff`, `py_compile`, and `tests/response` (`131 passed`);
  remote `.venv-gpu` `ruff`, `py_compile`, action-oracle targeted tests, and
  full `tests/response` (`131 passed`).
- Directly training an `action-minres` loss by differentiating through
  per-step Laplacians/Hessians was attempted but is not a usable formal path:
  even tiny H2 runs consumed CPU host time and did not enter effective GPU
  kernels. Do not use `--enrichment-training-objective action-minres` for
  production H2 tests. The paper-aligned no-action-backprop path is now
  `--enrichment-selection-objective action-oracle`: train candidate heads with
  the regular FermiNet weak residual flow, then use held-out strong action
  columns to accept only candidates with nonnegative L2 and winsor99
  epsilon^2 improvement. In action-oracle mode, default `-inf` action
  improvement thresholds are promoted to `0.0`; stricter user thresholds are
  honored.
- H2 action-oracle smokes on worker `3903266`:
  - `h2_iter50_weak_actionoracle_tiny_unbuffered.npz` was a 2-head tiny
    path check. It ran end-to-end and correctly rejected a bad candidate:
    `val_ratio_w99_max=8.822`, `val_action_rel_improve=-77.9`.
  - `h2_iter51_weak_actionoracle_direct8_medium.npz` used 8 heads, 64 train
    samples, 16 holdout samples, 20 epochs. It overfit strong action badly:
    train oracle residual went to `1.186e-04`, but validation worsened
    (`val_ratio_w99_max=19.13`,
    `val_action_rel_improve=-351`), so it was rejected.
  - `h2_iter52_weak_actionoracle_direct8_full.npz` used the previous
    formal-size smoke settings, 8 heads, 128 train samples, 32 holdout samples,
    50 epochs. Weak held-out acceptance passed (`holdout_pass=2/2`,
    `holdout_capture_ratio_min=2.594647`), but the held-out strong
    action-space oracle failed:
    `val_ratio_w99_max=14.09`,
    `val_ratio_p99_max=21.03`,
    `val_action_rel_improve=-224`,
    `val_w99_action_rel_improve=-197`. It was correctly
    `strong_oracle_rejected`.
- Current interpretation: Version A of the GPT Pro suggestion
  (no-action-backprop action-space oracle acceptance) is implemented and is
  working as a safety/selection gate, but the H2 learned candidate generation
  still overfits action-space training samples and does not generalize on
  held-out strong residuals. The next theory/code question is no longer
  "is the oracle/acceptance checking the right thing?", but how to generate
  candidate heads that generalize in action space: region-balanced residual
  Q/cache, stronger action-space regularization that does not backprop through
  Hessians each step, or a cached two-stage minimal-residual correction head.
- 2026-06-15 continuation on worker `3905019` (`fdbd:dc03:16:141::24`,
  port `9562`): implemented old-action-space Schur complement for the strong
  oracle and a new `--enrichment-selection-objective
  action-minres-composite` path. The composite path keeps weak neural heads as
  a raw pool, solves the complex cached action-MINRES oracle, compresses
  real/imag oracle coefficients with SVD into a small real composite basis,
  stores it as `params/__response_head_transform`, then reruns the strong
  oracle on the actual composite heads. Local and remote
  `tests/response` passed (`134 passed`). H2 Schur raw-block smoke
  `h2_iter53_weak_actionoracle_schur_direct8_full.npz` rejected with
  `val_ratio_w99=32.77` (`raw_val_w99=2.76`). H2 composite smoke
  `h2_iter54_composite_schur_direct8_full.npz` built an `8x2` transform and
  improved candidate action condition from `27.7` to `1.28`, but still failed
  held-out strong oracle (`val_ratio_w99=1.559`, relative epsilon2
  improvement `-1.437`). No orbital-z prefactor ablation
  `h2_iter55_composite_schur_nopref_direct8_full.npz` behaved similarly
  (`val_ratio_w99=1.587`, condition `1.24`). Interpretation: composite
  selection/regularization works mechanically and reduces conditioning, but
  the current learned raw candidate pool still lacks a held-out
  action-generalizing direction. Next work should target region-balanced
  residual/action caches and a source-conditioned correlated candidate family,
  not further loosening the oracle gate.
- 2026-06-15 later continuation on the same worker: aligned the Schur path
  with GPT Pro guidance by using Schur only as a diagnostic/novelty view and
  using raw held-out residual metrics for the actual strong-oracle gate.
  Added fixed source-conditioned correlated response heads via
  `--response-fixed-correlated-dictionary`, with smooth nearest-nucleus
  distance (`--response-fixed-correlated-en-softmin-beta`), optional e-e/e-n
  grids, singlet/opposite-spin and odd/even spatial projections, and a
  ground-prefactor so the heads are `Psi0(R) * feature(R)` rather than bare
  geometry. Added `params/__response_head_transform` support for this fixed
  family, nonfinite diagnostics for the Laplacian strong-residual audit, and
  `--enrichment-region-balanced-cache` to select node/e-n-cusp/e-e-cusp/tail/
  bulk-balanced holdout subsets for strong-oracle and E2 audits.
  Local `tests/response` passed (`138 passed`), and remote GPU targeted tests
  passed.
- H2 fixed-dictionary smoke results with official FermiNet ground
  `runs/h2_ferminet_formal/train_ckpt_019999.npz`:
  - `h2_iter58_fixedcorr_softmin_rawgate_schur_smoke.npz` used 21 geometry
    heads without the ground prefactor. E2 no longer produced NaNs after the
    diagnostics fix, but rejected strongly:
    `strong_residual_epsilon_over_eta=1.118e3`,
    worst region `node_tube/tail`; raw/composite oracle failed.
  - `h2_iter60_fixedcorr_softmin_regioncache_256to64_smoke.npz` used
    region-balanced holdout selection from 256 to 64 samples; the same
    geometry-only pool was much worse under balanced hard-region coverage
    (`raw_val_w99=32.28`, composite `raw_val_w99=99.84`), proving the
    region-balanced cache is a useful measuring instrument.
  - `h2_iter61_fixedcorr_groundpref_regioncache_256to64_smoke.npz` multiplied
    the fixed dictionary by `Psi0`. This greatly improved raw action
    conditioning (`cand_action_cond` about `6.5` instead of about `898`) and
    raw `w99` (`1.843` instead of `32.28`), but 4 composite heads over-compressed
    the oracle direction and failed E1/E2.
  - `h2_iter62_fixedcorr_groundpref_regioncache_comp8_smoke.npz` allowed up to
    8 composites and produced the rank-limited 6 real heads. This nearly
    passed E1 (`raw_val_w99=1.013`, L2 relative epsilon2 improvement
    `+8.8%`, winsor99 relative improvement `-2.7%`) and improved E2 to
    `strong_residual_epsilon_over_eta=574`, but still far above the threshold
    `5`. Worst E2 region was e-n cusp/high-action.
  - `h2_iter63_fixedcorr_groundpref_enfeatures_regioncache_smoke.npz` added
    explicit e-n decay features (`0.5,1.0,2.0`) to make 63 raw heads. It
    worsened the strong oracle (`raw_val_w99=8.35`, composite
    `raw_val_w99=15.1`) and E2 (`epsilon_over_eta=771`), so simple sharper
    e-n factors are not the fix.
- Current H2 diagnosis: the formal acceptance loop is now conservative and
  informative (raw gate, composite rerun, region-balanced cache, E2 audit all
  agree on rejection). The best candidate so far is the ground-prefactored
  21-head dictionary with 6 composite heads, but it still has too much
  high-action/e-n-cusp residual. Next question for GPT Pro should focus on
  how to regularize/generate source-conditioned candidates so the pointwise
  action stays controlled near e-n cusps, rather than on loosening gates.
- 2026-06-15 GPT Pro cusp-safe follow-up was implemented on worker `3905688`
  (`fdbd:dc03:16:129::20`, port `9650`; previous worker `3905019` stopped
  accepting connections). The fixed correlated dictionary now uses:
  - `Psi0 * feature` ground prefactor by default;
  - partial-wave p-source cusp lift near each nucleus,
    `q_1(r)=exp[(Z_A/2) tau(r)]` with `tau'(0)=1`;
  - cusp-neutral scalar radius
    `s0(r)=sqrt(r^2+r0^2)-r0` for tail/e-n grids;
  - cusp-neutral e-e Gaussian grids `exp[-(r_ij/s)^2]`;
  - CLI knobs `--response-fixed-correlated-cusp-neutral-radius` and
    `--response-fixed-correlated-source-cusp-radius`.
  Local `tests/response` passed (`140 passed`), and remote targeted tests
  passed.
- Tiny H2 cusp-safe experiments, all with official FermiNet ground, region
  cache, one omega `0.44`, and small 16-sample E2 audits:
  - `h2_iter67_cuspsafe_puresource_tiny_smoke.npz`: one almost-flat envelope
    (`1e-6`) source head, no e/eN grids. Conditioning was good
    (`cand_action_cond=1.0`, raw `w99=1.478`), but E2 still rejected:
    `epsilon_over_eta=157.6`; e-n cusp region remained large
    (`1471`).
  - `h2_iter68_cuspsafe_enneutral_tiny_smoke.npz`: added cusp-neutral e-n
    scalar decays (`0.5,1.0,2.0`) with almost-flat envelope. E2 improved to
    `epsilon_over_eta=93.3`, and e-n cusp dropped to `153`, but strong oracle
    worsened (`raw w99=2.372`, composite `raw w99=2.017`), so it still failed
    acceptance.
  - `h2_iter69_cuspsafe_eeneutral_tiny_smoke.npz`: added cusp-neutral e-e
    Gaussian scales (`0.5,1.0,2.0`) with almost-flat envelope. It was bad:
    raw candidate norms became tiny, `raw w99=12.89`, E2
    `epsilon_over_eta=843`, worst e-n cusp `9894`.
  - A 7-head source/tail run (`h2_iter66...`) showed source/tail grids can be
    slow and still fail (`raw w99=6.786`, E2 `epsilon_over_eta=980`), while
    the accidentally e-e-grid 21-head run (`h2_iter64...`) had E2
    `epsilon_over_eta=461` but terrible oracle (`raw w99=50`).
- New practical issue: the fixed cusp-safe dictionary's AD action/E2 path is
  very CPU-heavy on the worker. Several tiny runs used `0%` GPU with high CPU
  during Hessian/action compilation/evaluation. This may need an implementation
  change if more scans are required.
- Current cusp-safe diagnosis: p-source cusp-lift plus cusp-neutral e-n grids
  reduces E2 substantially in tiny audits, but the oracle/accepted correction
  direction does not improve enough. The e-e Gaussian grid is not helpful in
  this H2 screen. The next GPT Pro question should ask whether the source cusp
  decomposition is missing a local constant/cutoff/partition term, whether the
  e-n scalar grid should be orthogonalized/normalized before action-MINRES, or
  whether action/E2 must be evaluated with analytic local-energy formulas
  rather than generic Hessians.
- 2026-06-15 compact local source continuation: the p-source correction now
  uses a compact C-infinity bump and non-overlapping per-nucleus support radii
  (`min(requested_radius, 0.45 * nearest_nuclear_distance)`) so other nuclei do
  not leave tiny wrong-slope linear terms in a nucleus cusp region. The default
  `--response-fixed-correlated-source-cusp-radius` was raised from `0.5` to
  `1.0`; output NPZ files now save fixed-correlated dictionary metadata and the
  effective support radii. Local `tests/response` passed (`142 passed`), and
  remote ruff/targeted tests passed on worker `3905688`.
- H2 compact-source smokes on worker `3905688`, same tiny official FermiNet
  setup as iter67/68 (`omega=0.44`, `eta=0.02`, region-balanced oracle 32 and
  E2 16):
  - `h2_iter70_compactsource_puresource_tiny_smoke.npz` used source radius
    `0.5`. E2 improved vs old pure source from `157.6 eta` to `114.8 eta`,
    and e-n cusp dropped to about `115 eta`, but raw/selection oracle still
    failed (`w99=1.143`).
  - `h2_iter71_compactsource_enneutral_tiny_smoke.npz` used compact source
    radius `0.5` plus cusp-neutral e-n decays `0.5, 1.0`. It did not improve:
    E2 `117.9 eta`, oracle `w99=1.199`, composite condition about `11.2`.
  - `h2_iter72_compactsource_radius1_puresource_tiny_smoke.npz` used source
    radius `1.0` (clipped internally to non-overlap). This is the best fixed
    source result so far: E2 `42.9 eta`, e-n cusp about `37.7 eta`; the worst
    region moved to high-action (`144.8 eta`). Schur oracle looked good
    (`w99=0.863`) but the raw gate still failed (`w99=1.247`).
- Updated diagnosis after compact source: local source support and radius were
  real issues and fixed a large part of the cusp error. The remaining problem
  is no longer just e-n cusp; it is high-action/generalization plus disagreement
  between Schur-improved directions and the raw gate. Before training learned
  heads, ask GPT Pro whether to (1) keep raw gate as decisive or accept Schur
  when E2 improves, (2) add analytic action/local-energy formulas for
  `Psi0*f` columns to remove Hessian CPU bottlenecks and action noise, or
  (3) enrich the cusp-safe p-wave radial basis beyond a single source-lift
  radius.

## H Atom Analytic EOM Convergence Template

Use this pattern when the task is to compare H atom analytic get-orbitals EOM convergence across determinant counts. Keep `BATCH_SIZE` explicit and remove large `eom_stats.h5` files after each EOM run.

```bash
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=float32
export LD_LIBRARY_PATH=
export NCCL_DEBUG=WARN
export NVIDIA_TF32_OVERRIDE=0
export SKIP_MERLIN_OFFICIAL_INTERNAL_INSTALL=TRUE
export BATCH_SIZE=20480

RUN_ROOT=/opt/tiger/jaqmc_runs/h_atom_analytic_get_orbitals_scale100_ndets1_vs5_full_train_eom
rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

for NDETS in 1 5; do
  CASE_DIR="${RUN_ROOT}/ndets_${NDETS}"
  TRAIN_DIR="${CASE_DIR}/train"
  mkdir -p "${TRAIN_DIR}"

  jaqmc molecule train \
    workflow.save_path="${TRAIN_DIR}" \
    workflow.batch_size="${BATCH_SIZE}" \
    workflow.seed=42 \
    system.module=atom \
    system.symbol=H \
    wf.ndets="${NDETS}" \
    wf.hidden_dims_single='[64,64,64,64]' \
    wf.hidden_dims_double='[8,8,8,8]' \
    wf.hydrogen_analytic_orbitals=true \
    wf.hydrogen_analytic_1s_scale=100 \
    sampler.steps=30 \
    sampler.adapt_frequency=5 \
    pretrain.run.iterations=1000 \
    train.run.burn_in=100 \
    train.run.iterations=50000 \
    train.run.stop_on_nan=true \
    estimators.energy.kinetic.mode=forward_laplacian

  for ITERS in 200 500 1000 2000 5000; do
    EOM_DIR="${CASE_DIR}/eom_iter_${ITERS}"
    mkdir -p "${EOM_DIR}"

    jaqmc molecule eom \
      workflow.source_path="${TRAIN_DIR}" \
      workflow.save_path="${EOM_DIR}" \
      workflow.batch_size="${BATCH_SIZE}" \
      workflow.seed=43 \
      system.module=atom \
      system.symbol=H \
      wf.ndets="${NDETS}" \
      wf.hidden_dims_single='[64,64,64,64]' \
      wf.hidden_dims_double='[8,8,8,8]' \
      wf.hydrogen_analytic_orbitals=true \
      wf.hydrogen_analytic_1s_scale=100 \
      eom.sampling=tangent_reweighted \
      eom.sampler.steps=30 \
      eom.sampler.adapt_frequency=5 \
      eom.run.burn_in=100 \
      eom.run.iterations="${ITERS}" \
      eom.estimators.energy.kinetic.mode=forward_laplacian

    rm -f "${EOM_DIR}/eom_stats.h5"
  done
done
```
