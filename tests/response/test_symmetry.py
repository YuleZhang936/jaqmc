# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.response.symmetry import (
    SpinProjector,
    SymmetryProjector,
    atomic_angular_spatial_projector,
    finite_spatial_irrep_projectors,
    identity_spatial_projector,
    linear_spatial_projector,
    make_dipole_spatial_projectors,
    make_ground_spatial_projector,
    parity_spatial_projector,
    project_value,
    select_spatial_projector,
    spin_project_value,
    spin_squared_value,
    transform_electrons,
)


def _one_electron_data(z: float = 0.7) -> MoleculeData:
    return MoleculeData(
        electrons=jnp.asarray([[0.2, -0.1, z]], dtype=jnp.float32),
        atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )


def _two_electron_data() -> MoleculeData:
    return MoleculeData(
        electrons=jnp.asarray(
            [[0.4, 0.0, 0.0], [-0.2, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charges=jnp.asarray([2.0], dtype=jnp.float32),
    )


def test_parity_projector_keeps_dipole_odd_component():
    data = _one_electron_data()
    odd = SymmetryProjector(
        spatial=parity_spatial_projector("odd"),
        spin=SpinProjector((1, 0), None),
        label="odd",
    )
    even = SymmetryProjector(
        spatial=parity_spatial_projector("even"),
        spin=SpinProjector((1, 0), None),
        label="even",
    )

    def z_value(local_data):
        return local_data.electrons[0, 2]

    np.testing.assert_allclose(float(jnp.real(project_value(z_value, data, odd))), 0.7)
    np.testing.assert_allclose(float(jnp.real(project_value(z_value, data, even))), 0.0)


def test_lowdin_singlet_projector_removes_triplet_spatial_component():
    data = _two_electron_data()
    singlet = SpinProjector((1, 1), 0.0)

    def symmetric_value(local_data):
        del local_data
        return jnp.asarray(1.0)

    def antisymmetric_value(local_data):
        return local_data.electrons[0, 0] - local_data.electrons[1, 0]

    np.testing.assert_allclose(
        float(spin_squared_value(symmetric_value, data, (1, 1))),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(spin_squared_value(antisymmetric_value, data, (1, 1))),
        2.0 * float(antisymmetric_value(data)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(spin_project_value(symmetric_value, data, singlet)),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(spin_project_value(antisymmetric_value, data, singlet)),
        0.0,
        atol=1e-6,
    )


def test_combined_projector_applies_spatial_and_spin_sectors():
    data = _two_electron_data()
    projector = SymmetryProjector(
        spatial=identity_spatial_projector(),
        spin=SpinProjector((1, 1), 0.0),
        label="singlet",
    )

    def mixed_value(local_data):
        symmetric = jnp.asarray(3.0)
        triplet = local_data.electrons[0, 0] - local_data.electrons[1, 0]
        return symmetric + triplet

    np.testing.assert_allclose(
        float(project_value(mixed_value, data, projector)),
        3.0,
        atol=1e-6,
    )


def test_finite_irrep_projectors_discover_c2v_water_sectors():
    atoms = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]],
        dtype=np.float64,
    )
    charges = np.asarray([8.0, 1.0, 1.0])

    projectors = finite_spatial_irrep_projectors(atoms, charges)
    ground = make_ground_spatial_projector(atoms, charges, mode="auto")
    response = make_dipole_spatial_projectors(atoms, charges, mode="auto", axis=0)

    assert len(projectors) == 4
    assert all(projector.dimension == 1 for projector in projectors)
    assert ground.label == "A1"
    assert [projector.label for projector in response] == [
        projector.label for projector in projectors
    ]


def test_explicit_irrep_selection_for_ground_and_response():
    atoms = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]],
        dtype=np.float64,
    )
    charges = np.asarray([8.0, 1.0, 1.0])
    projectors = finite_spatial_irrep_projectors(atoms, charges)

    selected = select_spatial_projector(projectors, "irrep:irrep_02_d1")
    ground = make_ground_spatial_projector(
        atoms,
        charges,
        mode="auto",
        irrep_label="irrep_02_d1",
    )
    response = make_dipole_spatial_projectors(
        atoms,
        charges,
        mode="auto",
        axis=0,
        irrep_labels=("A1", "irrep_02_d1"),
    )

    assert selected.label == "irrep_02_d1"
    assert ground.label == "irrep_02_d1"
    assert [projector.label for projector in response] == ["A1", "irrep_02_d1"]


def test_unknown_irrep_selection_reports_available_labels():
    atoms = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]],
        dtype=np.float64,
    )
    charges = np.asarray([8.0, 1.0, 1.0])
    projectors = finite_spatial_irrep_projectors(atoms, charges)

    try:
        select_spatial_projector(projectors, "B9")
    except ValueError as exc:
        assert "available projectors" in str(exc)
    else:
        raise AssertionError("unknown irrep label should raise ValueError")


def test_finite_irrep_projectors_handle_nonabelian_tetrahedral_group():
    atoms = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    charges = np.asarray([6.0, 1.0, 1.0, 1.0, 1.0])

    projectors = finite_spatial_irrep_projectors(atoms, charges)

    assert len(projectors) == 5
    assert sorted(projector.dimension for projector in projectors) == [1, 1, 2, 3, 3]
    assert sum(projector.dimension**2 for projector in projectors) == 24


def test_auto_atom_response_uses_atomic_angular_sector():
    atoms = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    charges = np.asarray([2.0])

    projectors = make_dipole_spatial_projectors(atoms, charges, mode="auto", axis=0)
    ground = make_ground_spatial_projector(atoms, charges, mode="auto")

    assert [projector.label for projector in projectors] == ["L1_odd"]
    assert ground.label == "L0_even"


def test_atomic_l1_projector_keeps_vector_and_removes_f_wave():
    data = _one_electron_data()
    projector = SymmetryProjector(
        spatial=atomic_angular_spatial_projector(1, parity="odd", quadrature_order=7),
        spin=SpinProjector((1, 0), None),
        label="L1",
    )

    def z_value(local_data):
        return local_data.electrons[0, 2]

    def f_wave(local_data):
        x_value, y_value, z_coord = local_data.electrons[0]
        radius2 = x_value**2 + y_value**2 + z_coord**2
        return z_coord * (5.0 * z_coord**2 - 3.0 * radius2)

    np.testing.assert_allclose(
        float(jnp.real(project_value(z_value, data, projector))),
        float(z_value(data)),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        float(jnp.real(project_value(f_wave, data, projector))),
        0.0,
        atol=2e-5,
    )


def test_chunked_projector_matches_explicit_sum():
    data = _one_electron_data()
    projector = SymmetryProjector(
        spatial=atomic_angular_spatial_projector(1, parity="odd", quadrature_order=3),
        spin=SpinProjector((1, 0), None),
        label="L1",
    )

    def polynomial(local_data):
        x_value, y_value, z_value = local_data.electrons[0]
        return x_value + 0.3 * y_value**2 - 0.2j * z_value

    chunked = project_value(polynomial, data, projector, chunk_size=7)
    explicit = sum(
        jnp.asarray(coefficient)
        * polynomial(transform_electrons(data, matrix, projector.spatial.origin))
        for matrix, coefficient in zip(
            projector.spatial.matrices,
            projector.spatial.coefficients,
            strict=True,
        )
    )

    np.testing.assert_allclose(complex(chunked), complex(explicit), atol=2e-5)


def test_chunked_projector_can_trace_outside_shard_map():
    data = _one_electron_data()
    projector = SymmetryProjector(
        spatial=atomic_angular_spatial_projector(1, parity="odd", quadrature_order=3),
        spin=SpinProjector((1, 0), None),
        label="L1",
    )

    def polynomial(local_data):
        return local_data.electrons[0, 0] + 1j * local_data.electrons[0, 2]

    @jax.jit
    def projected(local_data):
        return project_value(polynomial, local_data, projector, chunk_size=7)

    value = projected(data)

    assert np.isfinite(complex(value))


def test_linear_auto_response_uses_sigma_and_pi_sectors():
    atoms = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64)
    charges = np.asarray([1.0, 1.0])

    parallel = make_dipole_spatial_projectors(atoms, charges, mode="auto", axis=2)
    perpendicular = make_dipole_spatial_projectors(atoms, charges, mode="auto", axis=0)
    ground = make_ground_spatial_projector(atoms, charges, mode="auto")

    assert [projector.label for projector in parallel] == ["Lambda0_u"]
    assert [projector.label for projector in perpendicular] == ["Lambda1_u"]
    assert ground.label == "Lambda0_g"


def test_linear_lambda_projector_separates_parallel_and_perpendicular_vectors():
    data = MoleculeData(
        electrons=jnp.asarray([[0.4, 0.2, 0.7]], dtype=jnp.float32),
        atoms=jnp.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=jnp.float32),
        charges=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    )
    lambda1 = SymmetryProjector(
        spatial=linear_spatial_projector(
            1,
            np.asarray([0.0, 0.0, 1.0]),
            parity="u",
            quadrature_order=16,
        ),
        spin=SpinProjector((1, 0), None),
        label="Lambda1",
    )

    def x_value(local_data):
        return local_data.electrons[0, 0]

    def z_value(local_data):
        return local_data.electrons[0, 2]

    np.testing.assert_allclose(
        float(jnp.real(project_value(x_value, data, lambda1))),
        float(x_value(data)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(jnp.real(project_value(z_value, data, lambda1))),
        0.0,
        atol=1e-6,
    )
