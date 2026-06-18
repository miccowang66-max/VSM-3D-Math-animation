"""
src/data_gen.py — Dataset generation and SVM computation.

Generates linear (perfectly separable) and nonlinear (circular ring)
datasets, computes SVM decision boundary / margins / support vectors,
and calculates kernel-lift z-coordinates for 3D visualization.

All randomness uses seed 42 for reproducibility.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_datasets(n_per_class: int = 80) -> dict:
    """
    Generate linear + nonlinear datasets and all SVM parameters.

    Parameters
    ----------
    n_per_class : int
        Number of particles per class (default 80, range 50-100).

    Returns
    -------
    dict
        Keys: n, linear_red, linear_blue, nonlinear_red, nonlinear_blue,
              z_linear_red, z_linear_blue, z_nonlinear_red, z_nonlinear_blue,
              w, b, sv_indices, db_line, margin_pos, margin_neg, hyperplane_z
    """
    rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # Linear dataset (perfectly linearly separable)
    # ------------------------------------------------------------------
    class_a_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([-3, -3])
    class_b_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([3, 3])

    X_linear = np.vstack([class_a_lin, class_b_lin])
    y_linear = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # ------------------------------------------------------------------
    # Nonlinear dataset (same particles morphed)
    #   Red  → tight cluster at origin
    #   Blue → ring around origin  (NOT linearly separable in 2D)
    # ------------------------------------------------------------------
    class_a_nonlin = rng.randn(n_per_class, 2) * 1.0
    angles = rng.uniform(0, 2 * np.pi, n_per_class)
    radii = rng.uniform(3.5, 5.0, n_per_class)
    class_b_nonlin = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # ------------------------------------------------------------------
    # SVM for linear case
    # ------------------------------------------------------------------
    try:
        from sklearn.svm import SVC

        svm = SVC(kernel="linear", C=1e10, random_state=42)
        svm.fit(X_linear, y_linear)
        w = svm.coef_[0].astype(np.float64)
        b = float(svm.intercept_[0])
        sv_indices = [int(i) for i in svm.support_]
    except ImportError:
        w, b, sv_indices = _fallback_svm(X_linear, class_a_lin, class_b_lin, n_per_class)

    # ------------------------------------------------------------------
    # Decision boundary & margin geometry
    # ------------------------------------------------------------------
    p1, p2, p1_pos, p2_pos, p1_neg, p2_neg = _compute_boundary_geometry(w, b)

    # ------------------------------------------------------------------
    # Kernel lift: z = exp(-(x² + y²))
    # ------------------------------------------------------------------
    z_lin_red = kernel_lift_z(class_a_lin)
    z_lin_blue = kernel_lift_z(class_b_lin)
    z_nonlin_red = kernel_lift_z(class_a_nonlin)
    z_nonlin_blue = kernel_lift_z(class_b_nonlin)

    # Separation hyperplane in kernel space
    sep_z = float(0.5 * (np.mean(z_nonlin_red) + np.mean(z_nonlin_blue)))

    return {
        "n": n_per_class,
        "linear_red": class_a_lin.tolist(),
        "linear_blue": class_b_lin.tolist(),
        "nonlinear_red": class_a_nonlin.tolist(),
        "nonlinear_blue": class_b_nonlin.tolist(),
        "z_linear_red": z_lin_red.tolist(),
        "z_linear_blue": z_lin_blue.tolist(),
        "z_nonlinear_red": z_nonlin_red.tolist(),
        "z_nonlinear_blue": z_nonlin_blue.tolist(),
        "w": w.tolist(),
        "b": b,
        "sv_indices": sv_indices,
        "db_line": [p1.tolist(), p2.tolist()],
        "margin_pos": [p1_pos.tolist(), p2_pos.tolist()],
        "margin_neg": [p1_neg.tolist(), p2_neg.tolist()],
        "hyperplane_z": sep_z,
    }


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def kernel_lift_z(pts: np.ndarray) -> np.ndarray:
    """Compute z = exp(-(x² + y²)) for an (N, 2) array."""
    r_sq = np.sum(pts**2, axis=1)
    return np.exp(-r_sq)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_boundary_geometry(w: np.ndarray, b: float, extent: float = 7.0):
    """Compute decision boundary & margin line endpoints."""
    w_norm = w / np.linalg.norm(w)
    perp = np.array([-w_norm[1], w_norm[0]], dtype=np.float64)
    p_center = -b * w_norm
    p1 = p_center + perp * extent
    p2 = p_center - perp * extent
    p1_pos = p1 + w_norm
    p2_pos = p2 + w_norm
    p1_neg = p1 - w_norm
    p2_neg = p2 - w_norm
    return p1, p2, p1_pos, p2_pos, p1_neg, p2_neg


def _fallback_svm(X_linear, class_a_lin, class_b_lin, n_per_class):
    """
    Analytical SVM fallback when scikit-learn is unavailable.
    Works for two well-separated Gaussian blobs.
    """
    centroid_a = class_a_lin.mean(axis=0)
    centroid_b = class_b_lin.mean(axis=0)
    w = centroid_b - centroid_a
    w = w / np.linalg.norm(w)
    b = float(-np.dot(w, (centroid_a + centroid_b) / 2))

    dists_a = np.abs(np.dot(class_a_lin, w) + b)
    dists_b = np.abs(np.dot(class_b_lin, w) + b)
    threshold_a = np.percentile(dists_a, 20)
    threshold_b = np.percentile(dists_b, 20)
    sv_indices = (
        [int(i) for i in np.where(dists_a <= threshold_a)[0]]
        + [int(n_per_class + i) for i in np.where(dists_b <= threshold_b)[0]]
    )

    all_dists = np.concatenate([dists_a, dists_b])
    margin_actual = float(np.min(all_dists[np.nonzero(all_dists)]))
    if margin_actual > 0:
        b = b / margin_actual
        w = w / margin_actual

    return w, b, sv_indices
