"""Tests for pauli module."""

import pytest
from epic.core.data_structure.pauli import PauliChar, PauliEigenState, PauliString


class TestPauliChar:
    """Test suite for PauliChar enum."""

    def test_enum_values(self):
        """Test PauliChar enum values."""
        assert PauliChar.I.value == "I"
        assert PauliChar.X.value == "X"
        assert PauliChar.Y.value == "Y"
        assert PauliChar.Z.value == "Z"

    def test_dual_x_z(self):
        """Test that X and Z are duals."""
        assert PauliChar.X.dual() == PauliChar.Z
        assert PauliChar.Z.dual() == PauliChar.X

    def test_dual_y(self):
        """Test that Y is self-dual."""
        assert PauliChar.Y.dual() == PauliChar.Y

    def test_dual_identity(self):
        """Test that I is self-dual."""
        assert PauliChar.I.dual() == PauliChar.I


class TestPauliEigenState:
    """Test suite for PauliEigenState enum."""

    def test_z_plus_preparation(self):
        """Test Z+ eigenstate preparation from |0>."""
        gates = PauliEigenState.Z_plus.pauli_from_zero()
        assert gates == []

    def test_z_minus_preparation(self):
        """Test Z- eigenstate preparation from |0>."""
        gates = PauliEigenState.Z_minus.pauli_from_zero()
        assert gates == [PauliChar.X]

    def test_x_plus_preparation(self):
        """Test X+ eigenstate preparation from |0>."""
        gates = PauliEigenState.X_plus.pauli_from_zero()
        assert gates == ["H"]

    def test_x_minus_preparation(self):
        """Test X- eigenstate preparation from |0>."""
        gates = PauliEigenState.X_minus.pauli_from_zero()
        assert gates == ["H", PauliChar.Z]


class TestPauliString:
    """Test suite for PauliString class."""

    def test_initialization_single_qubit(self):
        """Test PauliString initialization with single qubit."""
        ps = PauliString(string=(PauliChar.X,))
        assert len(ps.string) == 1
        assert ps.string[0] == PauliChar.X

    def test_initialization_multi_qubit(self):
        """Test PauliString initialization with multiple qubits."""
        ps = PauliString(string=(PauliChar.X, PauliChar.Y, PauliChar.Z))
        assert len(ps.string) == 3
        assert ps.string == (PauliChar.X, PauliChar.Y, PauliChar.Z)

    def test_weight_all_identity(self):
        """Test weight of all-identity string."""
        ps = PauliString(string=(PauliChar.I, PauliChar.I, PauliChar.I))
        assert ps.weight == 0

    def test_weight_single_operator(self):
        """Test weight with single non-identity operator."""
        ps = PauliString(string=(PauliChar.I, PauliChar.X, PauliChar.I))
        assert ps.weight == 1

    def test_weight_all_operators(self):
        """Test weight with all non-identity operators."""
        ps = PauliString(string=(PauliChar.X, PauliChar.Y, PauliChar.Z))
        assert ps.weight == 3

    def test_commutation_same_operators(self):
        """Test that identical Pauli strings commute."""
        ps1 = PauliString(string=(PauliChar.X, PauliChar.Y))
        ps2 = PauliString(string=(PauliChar.X, PauliChar.Y))
        assert ps1.commutes_with(ps2)

    def test_commutation_all_identity(self):
        """Test that identity strings commute with everything."""
        ps1 = PauliString(string=(PauliChar.I, PauliChar.I))
        ps2 = PauliString(string=(PauliChar.X, PauliChar.Z))
        assert ps1.commutes_with(ps2)
        assert ps2.commutes_with(ps1)

    def test_anticommutation_single_difference(self):
        """Test anticommutation with single different operator."""
        ps1 = PauliString(string=(PauliChar.X, PauliChar.I))
        ps2 = PauliString(string=(PauliChar.Z, PauliChar.I))
        assert not ps1.commutes_with(ps2)

    def test_commutation_two_differences(self):
        """Test commutation with two different operators (even)."""
        ps1 = PauliString(string=(PauliChar.X, PauliChar.X))
        ps2 = PauliString(string=(PauliChar.Z, PauliChar.Z))
        assert ps1.commutes_with(ps2)

    def test_commutation_length_mismatch_raises(self):
        """Test that different length strings raise ValueError."""
        ps1 = PauliString(string=(PauliChar.X,))
        ps2 = PauliString(string=(PauliChar.X, PauliChar.Z))
        with pytest.raises(ValueError, match="same length"):
            ps1.commutes_with(ps2)

    def test_immutability(self):
        """Test that PauliString is frozen/immutable."""
        ps = PauliString(string=(PauliChar.X, PauliChar.Y))
        with pytest.raises(Exception):  # Pydantic raises ValidationError or similar
            ps.string = (PauliChar.Z, PauliChar.Z)
