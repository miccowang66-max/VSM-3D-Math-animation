"""
tests/test_data_gen.py — Unit tests for data generation module.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_gen import generate_datasets, kernel_lift_z


def test_generate_datasets_structure():
    """All expected keys are present in the output dict."""
    data = generate_datasets(n_per_class=50)
    expected_keys = {
        "n", "linear_red", "linear_blue",
        "nonlinear_red", "nonlinear_blue",
        "z_linear_red", "z_linear_blue",
        "z_nonlinear_red", "z_nonlinear_blue",
        "w", "b", "sv_indices",
        "db_line", "margin_pos", "margin_neg",
        "hyperplane_z",
    }
    assert set(data.keys()) == expected_keys, f"Missing keys: {expected_keys - set(data.keys())}"


def test_particle_count():
    """Each class has exactly n particles."""
    for n in [50, 80, 100]:
        data = generate_datasets(n_per_class=n)
        assert data["n"] == n
        assert len(data["linear_red"]) == n
        assert len(data["linear_blue"]) == n
        assert len(data["nonlinear_red"]) == n
        assert len(data["nonlinear_blue"]) == n


def test_linear_separability():
    """Linear dataset is perfectly separable by w·x + b."""
    data = generate_datasets(n_per_class=80)
    w = np.array(data["w"])
    b = data["b"]

    red = np.array(data["linear_red"])
    blue = np.array(data["linear_blue"])

    red_scores = np.dot(red, w) + b
    blue_scores = np.dot(blue, w) + b

    # All red should be on one side, all blue on the other
    assert np.all(red_scores < 0) or np.all(red_scores > 0), "Red points not separable"
    assert np.all(blue_scores < 0) or np.all(blue_scores > 0), "Blue points not separable"
    assert np.sign(red_scores[0]) != np.sign(blue_scores[0]), "Classes not on opposite sides"


def test_nonlinear_not_linearly_separable():
    """Nonlinear dataset should NOT be separable by a straight line through origin."""
    data = generate_datasets(n_per_class=80)
    red = np.array(data["nonlinear_red"])
    blue = np.array(data["nonlinear_blue"])

    # Red near origin, blue on outer ring — no single line can separate
    red_norms = np.linalg.norm(red, axis=1)
    blue_norms = np.linalg.norm(blue, axis=1)
    assert np.max(red_norms) < np.min(blue_norms), "Nonlinear data is unexpectedly separable"


def test_support_vectors_exist():
    """At least some support vectors are detected."""
    data = generate_datasets(n_per_class=80)
    assert len(data["sv_indices"]) > 0, "No support vectors found"
    # Each SV index is within [0, 2*n)
    n = data["n"]
    assert all(0 <= i < 2 * n for i in data["sv_indices"]), "SV index out of range"


def test_boundary_geometry():
    """Decision boundary and margin lines are valid 2D coordinates."""
    data = generate_datasets(n_per_class=80)
    for key in ["db_line", "margin_pos", "margin_neg"]:
        pts = data[key]
        assert len(pts) == 2, f"{key}: expected 2 endpoints"
        assert all(len(p) == 2 for p in pts), f"{key}: endpoints must be 2D"


def test_kernel_lift_positive():
    """Kernel lift z-values are in (0, 1]."""
    data = generate_datasets(n_per_class=80)
    for key in ["z_linear_red", "z_linear_blue", "z_nonlinear_red", "z_nonlinear_blue"]:
        z_vals = data[key]
        assert all(0 < z <= 1 for z in z_vals), f"{key}: values out of (0, 1]"


def test_kernel_lift_function():
    """Direct test of kernel_lift_z helper."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    z = kernel_lift_z(pts)
    np.testing.assert_almost_equal(z[0], 1.0, decimal=5)          # exp(0) = 1
    np.testing.assert_almost_equal(z[1], np.exp(-1.0), decimal=5) # exp(-1)
    np.testing.assert_almost_equal(z[2], np.exp(-8.0), decimal=5) # exp(-8)


def test_reproducibility():
    """Same seed produces identical output."""
    d1 = generate_datasets(n_per_class=60)
    d2 = generate_datasets(n_per_class=60)
    assert d1["linear_red"] == d2["linear_red"]
    assert d1["w"] == d2["w"]
    assert d1["sv_indices"] == d2["sv_indices"]


def test_hyperplane_z_between():
    """Hyperplane z is between the mean lifted z of the two classes."""
    data = generate_datasets(n_per_class=80)
    z_red = np.mean(data["z_nonlinear_red"])
    z_blue = np.mean(data["z_nonlinear_blue"])
    assert min(z_red, z_blue) <= data["hyperplane_z"] <= max(z_red, z_blue)


def test_w_norm_positive():
    """Weight vector has nonzero norm."""
    data = generate_datasets()
    w = np.array(data["w"])
    assert np.linalg.norm(w) > 0


if __name__ == "__main__":
    # Run all tests
    import pytest
    pytest.main([__file__, "-v"])
