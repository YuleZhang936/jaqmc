# Dipole NQS-LIT Spectra

The `jaqmc molecule lit` command computes electric-dipole response spectra from
a trained molecular ground-state wavefunction. The current implementation uses
the NQS-LIT workflow: it restores the ground-state neural wavefunction, samples a
large source-state pool, optimizes a response NQS for each complex-frequency
point, and writes the Lorentz integral transform spectrum.

Use this workflow after a normal `jaqmc molecule train` run has produced a
ground-state checkpoint. The LIT command restores that checkpoint from
`workflow.restore_path`; equivalently, set `lit.nqs_checkpoint_path` when the
ground-state checkpoint lives somewhere else. JaQMC keeps ground-state
optimization and response-spectrum optimization as separate commands so that
the same trained wavefunction can be reused for different frequency windows,
broadening widths, and dipole axes.

## Running A Spectrum

For a trained molecule run in `./runs/<molecule-ground>`, run:

```bash
jaqmc molecule lit --yml molecule_lit.yml \
  workflow.save_path=./runs/<molecule-lit> \
  workflow.restore_path=./runs/<molecule-ground>
```

The command always writes `lit_spectrum.npz` under `workflow.save_path`. The
file contains the scan grid, LIT values, fidelity, reverse-KL and reweighting
effective-sample-size diagnostics, warm-start/continuation diagnostics,
selected checkpoint iterations, the two fidelity error factors, and matched
jackknife blocks for correlated statistical errors. The canonical transform is
the un-clipped `signed_lit`; finite-width `broadened` values are diagnostic and
are not a substitute for an inversion. The LIT workflow never performs that
inversion automatically; use the separate `jaqmc response invert-lit` CPU
postprocessor after inspecting the raw result.

The frequency scan is intentionally serial. After a negative-frequency warm
start, JaQMC optimizes frequencies in strictly increasing order. At each point,
it evaluates candidates on the fixed held-out source pool and passes the best
parameters—not necessarily the final iteration—to the next frequency. The
SPRING history is reset at each new frequency because it belongs to the local
optimization problem and may not match an earlier selected checkpoint.

The negative warm start is not allowed to jump directly to a distant positive
window. JaQMC inserts an adaptive, unreported bridge before the first spectrum
point. If a state solves the equation at `omega`, its inherited relative
residual at `omega + delta` is approximately
`|delta| * sqrt(L / source_norm)`. The default bounds this by `0.2`, matching
the visible `delta_omega / Gamma = 2/10` grid ratio in the nuclear calculation,
then checks inherited fidelity on the held-out pool and bisects a proposed step
when needed. The nuclear article does not publish its hidden `-100` to `200 MeV`
transfer grid; this adaptive rule makes that missing engineering choice explicit
without pretending that one large jump is frequency continuation.

Frequency blocks are deliberately unsupported because they break this
predecessor chain at block boundaries. Within one frequency,
`lit.nqs_data_parallel=local_devices` instead shards the
unchanged global Monte Carlo batch across every GPU visible to one process. All
normalizations and action-state gradients are reduced globally; SPRING uses a
distributed score matrix but one replicated direction and history. Thus a
global batch can be configured directly, or by setting a per-device batch that
is multiplied by the number of local GPUs; the response-parameter continuation
chain remains unique. This mode requires one JAX process on one worker. Batch
divisibility and the exact train/evaluation pool sizes are checked immediately
after loading, before response training.

The published stabilization objective is
`fidelity - lambda * KL(pi_action || pi_source)`. The defaults use
`lambda=1`, scale-invariant SPRING damping
`epsilon=1e-3 * mean(diag(S))`, and a configurable SPRING decay. The nuclear
LIT article does not report the decay value; the default `0.99` comes from the
general SPRING reference rather than from that LIT calculation.

Before the resolvent optimization, JaQMC distills an independently initialized
response NQS toward the fixed dipole source on the training `pi_Phi` pool and
selects its best checkpoint on a separate held-out pool. Atomic responses use
the automatically diagnosed parity opposite to the ground state; multi-center
molecules are currently accepted only when their discovered spatial sector is
C1. There is no source-aligned ansatz, direct-`Psi` fallback, or soft point-group
penalty in the production path.

## End-To-End H Atom Reference

This section gives a complete H atom reference workflow: first train a
ground-state FermiNet, then run an NQS-LIT dipole spectrum from that checkpoint.
For larger atoms or molecules, use the same two-stage structure but adjust the
system definition, wavefunction size, and optimization budget.

### Train The Ground State

Save as `h_atom_ground.yml`:

```yaml
logging:
  level: info

workflow:
  seed: 1234
  batch_size: 4096

system:
  module: atom
  symbol: H
  electron_init_width: 1.0
  basis: cc-pVTZ

wf:
  module: jaqmc.app.molecule.wavefunction.ferminet
  ndets: 8
  hidden_dims_single: [64, 64, 64, 64]
  hidden_dims_double: [16, 16, 16, 16]
  use_last_layer: false
  envelope: abs_isotropic
  orbitals_spin_split: true
  full_det: true

sampler:
  steps: 20
  initial_width: 0.35
  adapt_frequency: 100

pretrain:
  run:
    burn_in: 200
    iterations: 2000
    save_step_interval: 1000
    save_time_interval: 600
  optim:
    learning_rate:
      rate: 0.0003
      delay: 2000
      decay: 1.0

train:
  run:
    burn_in: 500
    iterations: 30000
    save_step_interval: 1000
    save_time_interval: 600
  optim:
    learning_rate:
      rate: 0.02
      delay: 10000
      decay: 1.0
    damping: 0.001
    norm_constraint: 0.001
    curvature_ema: 0.95
    inverse_update_period: 1
  writers:
    console:
      fields: pmove:.3f,energy=total_energy:.8f,variance=total_energy_var:.6f
```

Run ground-state training:

```bash
jaqmc molecule train --yml h_atom_ground.yml \
  workflow.save_path=./runs/h_atom-ground
```

For a converged H atom calculation, the final ground-state energy should be
close to `-0.5 Ha`.

### Run The LIT Spectrum

The following YAML configures an H atom reference calculation for the
`1s -> 3p` bright transition near `0.444444 Ha`. It assumes a ground-state
FermiNet checkpoint has already been trained and is supplied through
`workflow.restore_path` on the command line.

Save as `h_atom_lit.yml`:

```yaml
logging:
  level: info

workflow:
  seed: 1234
  batch_size: 4096

system:
  module: atom
  symbol: H
  electron_init_width: 1.0
  basis: cc-pVTZ

wf:
  module: jaqmc.app.molecule.wavefunction.ferminet
  ndets: 8
  hidden_dims_single: [64, 64, 64, 64]
  hidden_dims_double: [16, 16, 16, 16]
  use_last_layer: false
  envelope: abs_isotropic
  orbitals_spin_split: true
  full_det: true

sampler:
  steps: 20
  initial_width: 0.35
  adapt_frequency: 100

lit:
  eta: 0.005
  omega_min: 0.420
  omega_max: 0.490
  omega_points: 71
  # Optional: set explicit, strictly increasing points instead of linspace.
  # omega_values: [0.444, 0.46875, 0.480]
  axes: x
  output_filename: lit_spectrum.npz

  nqs_allow_untrained_ground: false
  nqs_ground_energy: null
  nqs_energy_steps: 32
  nqs_burn_in: 150

  nqs_source_center_steps: 64
  nqs_source_center_override: null
  nqs_source_norm_override: null
  nqs_source_burn_in: 150
  nqs_source_floor: 0.0001
  nqs_train_pool_batches: 64
  nqs_eval_pool_batches: 16
  nqs_pool_stride: 4
  # Choose either a global batch or the corresponding per-device batch.
  nqs_train_update_batch_size: 4096
  nqs_eval_batch_size: 4096
  nqs_train_update_batch_size_per_device: 0
  nqs_eval_batch_size_per_device: 0
  # Optional: shard each single-frequency batch across all locally visible GPUs.
  nqs_data_parallel: "off"
  nqs_reuse_source_pool: true
  nqs_save_source_pool: true

  # Fit the independent response NQS to Phi on the fixed source pools before
  # applying the resolvent action.
  nqs_source_distillation_iterations: 1000

  nqs_learning_rate: 0.005
  nqs_reverse_kl_weight: 1.0
  nqs_spring_epsilon: 0.001
  nqs_spring_decay: 0.99
  nqs_spring_damping_floor: 1.0e-12
  nqs_sr_max_norm: 0.05
  nqs_sr_score_eps: 1.0e-10
  nqs_warm_start_omega: -3.674932217565499
  nqs_warm_start_iterations: 800
  # Optional paper-style early stopping on the cumulative best held-out
  # fidelity. Zero patience disables plateau stopping.
  nqs_fidelity_plateau_start_iteration: 800
  nqs_fidelity_plateau_patience_iterations: 400
  nqs_fidelity_plateau_min_delta: 1.0e-5
  # Maximum optimizer budget for each continuation bridge.
  nqs_continuation_iterations: 2000
  nqs_continuation_step_fraction: 0.2
  nqs_continuation_fidelity_retention: 0.95
  # null uses min(0.2 * eta, the reported spectrum spacing).
  nqs_continuation_min_step: null
  nqs_continuation_max_points: 256
  nqs_iterations: 2000
  nqs_selection_interval: 100
  nqs_log_interval: 100

  nqs_response_ndets: 8
  nqs_response_hidden_dims_single: [128, 128, 128, 128]
  nqs_response_hidden_dims_double: [16, 16, 16, 16]
  nqs_response_use_last_layer: false
  nqs_response_envelope: abs_isotropic
  nqs_response_orbitals_spin_split: true
  nqs_parity_eval_batch_size: 256
  nqs_sector_tolerance: 1.0e-5
  nqs_atomic_source_parity_max_loss: 1.0e-3
  nqs_atomic_ground_parity_max_loss: 1.0e-3
```

Run it from the repository root:

```bash
jaqmc molecule lit --yml h_atom_lit.yml \
  workflow.save_path=./runs/h_atom-lit \
  workflow.restore_path=./runs/h_atom-ground
```

If the checkpoint directory differs from the workflow restore directory, add:

```bash
lit.nqs_checkpoint_path=./runs/h_atom-ground
```

## Formal Inversion

`jaqmc molecule lit` always stops after writing the raw `lit_spectrum.npz`.
It never chooses a pole count or runs an inversion. This separation keeps the
expensive NQS calculation independent of later model-order hypotheses.

Run an inversion explicitly on the CPU with a separate configuration. For
example, the following file tests a three-pole hypothesis without supplying
reference or experimental peak locations:

```yaml
inversion:
  input_paths: [runs/h_atom-lit/lit_spectrum.npz]
  output_path: runs/h_atom-lit/inversion_k3.npz
  threshold: 0.5
  ionized_energy: null
  require_determined: true
  pole_count: 3
  pole_search_energy_min: null  # defaults to min(omega)
  pole_search_energy_max: null  # defaults to max(omega)
  fit_pole_energies: true
  max_fitted_poles: 3
```

```bash
jaqmc response invert-lit --yml invert_k3.yml
```

The command is never invoked by a LIT run. Reuse the same raw NPZ with separate
output paths to compare `pole_count: 1`, `2`, and so on. A pole count is a
manual model-order hypothesis, not a measured property or a peak-location
prior. The postprocessor selects starting energies from the saved LIT by
covariance-weighted greedy nonnegative fitting and continuously refines them.
It does not consume experimental transition energies.

`pole_count` is mutually exclusive with explicit `pole_energies` and
`pole_energy_bounds`. For a known theoretical model, explicit ordered initial
energies and non-overlapping bounds remain supported. A continuum grid may
also be supplied when its first node is the threshold.

For a threshold tied to the ground state obtained in the same run, set
`inversion.ionized_energy` instead of `inversion.threshold`. The postprocessor
reads `ground_energy` from every raw archive and uses
`ionized_energy - ground_energy`. The two settings are mutually exclusive.
Set `require_determined: true` for formal analyses to reject rank-deficient
fits instead of writing them as results.

The output is pickle-free and self-describing. It stores the validated input,
statistical and fidelity/D covariance terms, fitted poles and continuum,
forward-fit residual, solver diagnostics, and delete-one-block jackknife errors
for pole energies, strengths, and continuum density. A failed NNLS or pole fit
fails the postprocessing command without changing the raw LIT archive.

### Optional Correlated Multi-Width Inversion

One finite `eta` defines a complete LIT data set and may be inverted on its
own. A second width is useful as a resolution-stability check, as done in the
reference PRL, but is not a mathematical prerequisite. For an optional joint
fit, run the same frequency grid with multiple finite `eta` values and reuse
the same saved evaluation pool by setting one common, explicit
`lit.nqs_source_pool_dir` for every run. Each output records matched
delete-one-block jackknife pseudo-values and the evaluation-pool digest. The
strict loader rejects independently generated pools and keeps the verified
cross-frequency and cross-width covariance. List every raw archive explicitly
in the manual inversion configuration:

```yaml
inversion:
  input_paths:
    - runs/h_atom-lit-eta005/lit_spectrum.npz
    - runs/h_atom-lit-eta010/lit_spectrum.npz
  output_path: runs/h_atom-lit-eta-joint/inversion_k3.npz
  threshold: 0.5
  pole_count: 3
  fit_pole_energies: true
  max_fitted_poles: 3
```

The lower-level programmatic interface remains available for custom analyses:

```python
import numpy as np

from jaqmc.response import aggregate_lit_npz, invert_signed_lit

data = aggregate_lit_npz(
    [
        "runs/h_atom-lit-eta005/lit_spectrum.npz",
        "runs/h_atom-lit-eta010/lit_spectrum.npz",
    ]
)
result = invert_signed_lit(
    data.omega,
    data.eta,
    data.signed_lit,
    threshold=0.5,
    pole_energies=np.array([0.375, 4 / 9, 15 / 32]),
    continuum_grid=np.linspace(0.5, 1.0, 24),
    covariance=data.covariance,
    continuum_regularization=1.0e-3,
)
```

Pole strengths and continuum densities are constrained to be nonnegative, and
pole energies can optionally be refined within explicit or automatically
generated non-overlapping bounds. A single-width full-rank fit is not marked
underdetermined; instead `result.diagnostics.cross_width_validated` is false
until more than one width is included. `data.statistical_covariance` retains the
matched-block Monte Carlo correlations. Following the PRL likelihood,
`data.covariance` additionally contains
`diag(data.systematic_error**2)`, where `systematic_error` is the raw-LIT
fidelity/D monitor. Supplement Eq. (19) retains only the leading term in
`sqrt(1 - fidelity)`, so away from high fidelity this quantity is not a
rigorous upper bound. This diagonal systematic term is the paper's explicit
approximation; it should not be interpreted as evidence that optimization
errors at different frequencies are physically independent. Any invalid or
non-finite monitor makes the formal loader fail instead of silently assigning
zero uncertainty.

## Expected H Atom Sanity Checks

For a well-converged H atom ground state, the restored ground energy should be
close to `-0.5 Ha`. In the `0.420-0.490 Ha` window, the first strong bound
bright peak is the exact hydrogen transition

```text
1s -> 3p: 0.444444 Ha
```

Higher bound bright transitions in the same window are

```text
1s -> 4p: 0.468750 Ha
1s -> 5p: 0.480000 Ha
1s -> 6p: 0.486111 Ha
1s -> 7p: 0.489796 Ha
```

These energies are references for an inverted response, not targets for local
maxima of a finite-width curve. Raw local maxima can contain side lobes or
optimizer artifacts, especially near the ionization threshold where Rydberg
levels become dense. Check `fidelity`, `reweight_ess_fraction`,
`reweight_max_fraction`, `error_d_valid`, and `error_bound_monitor` before using
any transform point in an inversion.

## Faster Smoke Tests

For a quick launch test, keep the same system and checkpoint but reduce the
sampling and optimization budget:

```bash
jaqmc molecule lit --yml h_atom_lit.yml \
  workflow.save_path=./runs/h_atom-lit-smoke \
  workflow.restore_path=./runs/h_atom-ground \
  lit.omega_min=0.444 lit.omega_max=0.444 lit.omega_points=1 \
  lit.nqs_train_pool_batches=2 lit.nqs_eval_pool_batches=1 \
  lit.nqs_warm_start_iterations=1 lit.nqs_iterations=1
```

This only checks that checkpoint restoration, source-pool construction, and the
response update path run successfully. It is not a physics calculation.
