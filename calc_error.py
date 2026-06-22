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

# Compute errors by comparing the quadtree table to the (denser) testing data table
def compute_quadtree_error(quadtree_file, ref_points, ref_values):

    # Load the quadtree data
    print(f"Loading quadtree from {quadtree_file}...")
    root = load_quadtree(quadtree_file)

    # Evaluate quadtree at all reference points
    qt_at_ref = np.empty(len(ref_points))
    for i, (x, y) in enumerate(ref_points):
        qt_at_ref[i] = root.evaluate(x, y)

    # Compute error norms
    error = qt_at_ref - ref_values
    l1 = np.mean(np.abs(error))
    l2 = np.sqrt(np.mean(error**2))
    linf = np.max(np.abs(error))

    # Compute size of the quadtree file
    size_kb = os.path.getsize(quadtree_file) / 1024

    return l1, l2, linf, size_kb


# --------------------------------------------------
# Example main (optional)
# --------------------------------------------------
if __name__ == "__main__":

    # --------------------------------------------
    # Config
    # --------------------------------------------
    TEST_FILE = "testing_data.hdf5"
    TREE_FILE = "tables/quadtree-7-0.1.npz"

    # Load the test_file to get the test points and reference values
    ref_points, ref_values, ref_interp = load_reference_hdf5(TEST_FILE)
    
    # Compute the error norms for the quadtree
    l1, l2, linf, size_kb = compute_quadtree_error(TREE_FILE, ref_points, ref_values)


    print("\n--- Quadtree Error ---")
    print(f"L1   : {l1:.3e}")
    print(f"L2   : {l2:.3e}")
    print(f"Linf : {linf:.3e}")
    print(f"Size : {size_kb:.2f} kB")
