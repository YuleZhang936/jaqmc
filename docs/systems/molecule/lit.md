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

The command writes `lit_spectrum.npz` under `workflow.save_path`. The file
contains the scan grid, LIT values, fidelity, reverse-KL and reweighting
effective-sample-size diagnostics, warm-start/continuation diagnostics,
selected checkpoint iterations, and peak-picking output.

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

Frequency blocks and remote frequency workers are rejected because they break
this predecessor chain at block boundaries. Use `lit.scan_parallel=off`.
Within one frequency, `lit.nqs_data_parallel=local_devices` instead shards the
unchanged global Monte Carlo batch across every GPU visible to one process. All
normalizations and action-state gradients are reduced globally; SPRING uses a
distributed score matrix but one replicated direction and history. Thus a
global batch of 1024 on eight GPUs means 128 walkers per GPU, not 8192 walkers,
and the response-parameter continuation chain remains unique. This mode
requires source-sampled training (`nqs_direct_psi_train: false`), a batch size
divisible by the local device count, and one JAX process on one worker. Both
the configured global training-update and evaluation chunk sizes are checked
at startup, before sampling. A reused held-out source pool must also have a
total walker count divisible by the device count so its final evaluation chunk
can be sharded; this pool-size check runs immediately after loading, before
covariance evaluation or response training.

The published stabilization objective is
`fidelity - lambda * KL(pi_action || pi_source)`. The defaults use
`lambda=1`, scale-invariant SPRING damping
`epsilon=1e-3 * mean(diag(S))`, and a configurable SPRING decay. The nuclear
LIT article does not report the decay value; the default `0.99` comes from the
general SPRING reference rather than from that LIT calculation.

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
  peak_min_height_fraction: 0.02
  preview_roots: 8

  scan_parallel: "off"

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
  nqs_train_update_batch_size: 4096
  nqs_eval_batch_size: 4096
  # Optional: shard each single-frequency batch across all locally visible GPUs.
  nqs_data_parallel: "off"
  nqs_reuse_source_pool: true
  nqs_save_source_pool: true

  # Zero selects the published source-pool importance-reweighting protocol.
  # A positive threshold enables JaQMC's optional direct-action fallback.
  nqs_reweight_ess_fraction_min: 0.0
  nqs_direct_psi_train: false
  nqs_direct_psi_burn_in: 5
  nqs_direct_psi_batches: 1
  nqs_direct_psi_train_batches: null
  nqs_direct_psi_eval_batches: null
  nqs_direct_psi_stride: 1
  # Engineering controls for the optional direct-action fallback.
  nqs_direct_psi_precompile: true
  nqs_direct_psi_persistent_sampler: true

  nqs_learning_rate: 0.005
  nqs_reverse_kl_weight: 1.0
  nqs_spring_epsilon: 0.001
  nqs_spring_decay: 0.99
  nqs_spring_damping_floor: 1.0e-12
  nqs_sr_max_norm: 0.05
  nqs_sr_score_eps: 1.0e-10
  nqs_warm_start_omega: -3.674932217565499
  nqs_warm_start_iterations: 800
  nqs_continuation_iterations: 800
  # Optional cumulative quality-check milestones on the same optimizer
  # trajectory. The first entry must equal nqs_continuation_iterations.
  # Only a pure fidelity miss extends to the next milestone.
  nqs_continuation_iteration_schedule: null
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

Peak positions are the main diagnostic. Raw local maxima can contain side lobes
or optimizer artifacts, especially near the ionization threshold where Rydberg
levels become dense and the reweighting effective sample size can be small. Use
the `fidelity` and `reweight_ess_fraction` arrays in `lit_spectrum.npz` when
judging whether a high-energy feature is reliable.

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
