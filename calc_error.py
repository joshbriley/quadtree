import numpy as np
import h5py
import os
from scipy.interpolate import LinearNDInterpolator
from build_quadtree import load_quadtree


# --------------------------------------------------
# Load fine reference data
# --------------------------------------------------
def load_reference_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        print("Reference fields:", list(f["Table_Values"].keys()))

        X = f["den"][:]    
        Y = f["temp"][:]   
        F = f["Table_Values"]["f"][:]

    ref_points = np.column_stack([X.ravel(), Y.ravel()])
    ref_values = F.ravel()

    print("Building reference interpolator...")
    ref_interp = LinearNDInterpolator(ref_points, ref_values)

    return ref_points, ref_values, ref_interp


# --------------------------------------------------
# Error computation
# --------------------------------------------------
def calculate_norms_for_quadtree(quadtree_file, ref_interp, x_test, y_test):
    """
    Compare quadtree approximation to fine-table interpolant.
    """

    tree = load_quadtree(quadtree_file)

    nx = len(x_test)
    ny = len(y_test)

    qt_vals = np.empty((nx, ny))
    ref_vals = np.empty((nx, ny))

    for i, x in enumerate(x_test):
        for j, y in enumerate(y_test):

            qt_vals[i, j] = tree.evaluate(x, y)

            ref_val = ref_interp(x, y)

            # handle outside convex hull (rare)
            if np.isnan(ref_val):
                ref_val = 0.0

            ref_vals[i, j] = ref_val

    # --------------------------------------------------
    # Compute errors
    # --------------------------------------------------
    err = np.abs(qt_vals - ref_vals)

    l1 = np.mean(err)
    l2 = np.sqrt(np.mean(err**2))
    linf = np.max(err)

    size_kb = os.path.getsize(quadtree_file) / 1000.0

    return l1, l2, linf, size_kb


# --------------------------------------------------
# Example main (optional)
# --------------------------------------------------
if __name__ == "__main__":

    # --------------------------------------------
    # Config
    # --------------------------------------------
    REF_FILE = "fine_reference_data.hdf5"
    TREE_FILE = "tables/quadtree-7-1e-3.npz"

    TEST_RESOLUTION = 200

    # --------------------------------------------
    # Load reference table
    # --------------------------------------------
    ref_points, ref_values, ref_interp = load_reference_hdf5(REF_FILE)

    xmin, xmax = ref_points[:,0].min(), ref_points[:,0].max()
    ymin, ymax = ref_points[:,1].min(), ref_points[:,1].max()

    # --------------------------------------------
    # Build evaluation grid
    # --------------------------------------------
    x_test = np.linspace(xmin, xmax, TEST_RESOLUTION)
    y_test = np.linspace(ymin, ymax, TEST_RESOLUTION)

    # --------------------------------------------
    # Compute error
    # --------------------------------------------
    l1, l2, linf, size_kb = calculate_norms_for_quadtree(
        TREE_FILE,
        ref_interp,
        x_test,
        y_test
    )

    print("\n--- Quadtree Error ---")
    print(f"L1   : {l1:.3e}")
    print(f"L2   : {l2:.3e}")
    print(f"Linf : {linf:.3e}")
    print(f"Size : {size_kb:.2f} kB")
