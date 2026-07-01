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
contains the scan grid, LIT values, fidelity diagnostics, reweighting effective
sample-size diagnostics, and peak-picking output.

On a multi-GPU node, the reference LIT configuration below uses
`lit.scan_parallel=auto`. JaQMC splits the frequency grid into local worker
processes, using one process per visible GPU by default for this setting. Each
frequency point starts from the same axis warm-started response parameters, so
peak positions are not path-dependent on neighboring frequencies or worker block
boundaries.

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
  axes: x
  output_filename: lit_spectrum.npz
  peak_min_height_fraction: 0.02
  preview_roots: 8

  scan_parallel: auto
  scan_parallel_workers: 0
  scan_parallel_procs_per_device: 1
  scan_parallel_min_points_per_worker: 2

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
  nqs_reuse_source_pool: true
  nqs_save_source_pool: true

  # Fall back to direct pi_Psi sampling when source-pool reweighting ESS collapses.
  # The large source pool is evaluated in chunks to keep device memory bounded.
  nqs_reweight_ess_fraction_min: 0.05
  nqs_direct_psi_burn_in: 5
  nqs_direct_psi_batches: 1
  nqs_direct_psi_stride: 1

  nqs_learning_rate: 0.005
  nqs_sr_damping: 0.01
  nqs_sr_max_norm: 0.05
  nqs_sr_score_eps: 1.0e-10
  nqs_warm_start_omega: -3.674932217565499
  nqs_warm_start_iterations: 800
  nqs_iterations: 2000
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
