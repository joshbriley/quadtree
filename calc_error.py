import numpy as np
import h5py
import os
from scipy.interpolate import RegularGridInterpolator
from build_quadtree import load_quadtree

# --------------------------------------------------
# Load fine reference data
# --------------------------------------------------
def load_reference_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        X = f["den"][:]    
        Y = f["temp"][:]   
        F = f["Table_Values"]["f"][:]

    # Extract 1D coordinate vectors (regular grid)
    # den is constant along rows, varies along columns
    # temp is constant along columns, varies along rows
    x_coords = X[:, 0]  # 1D array of x coordinates
    y_coords = Y[0, :]  # 1D array of y coordinates

    # Build fast regular grid interpolator
    ref_interp = RegularGridInterpolator(
        (x_coords, y_coords), 
        F.T,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    # For compatibility with rest of code, also return flattened points/values
    ref_points = np.column_stack([X.ravel(), Y.ravel()])
    ref_values = F.ravel()

    return ref_points, ref_values, ref_interp

# Compute errors by comparing the quadtree table to the (denser) testing data table
def compute_quadtree_error(quadtree_file, ref_points, ref_values, n_points=None, random_seed=None):
    """
    Evaluate quadtree error at reference points.
    
    Parameters:
    n_points : int or None
        If int, randomly sample n_points from ref_points.
        If None, use all points.
    random_seed : int or None
        Seed for reproducibility when sampling.
    """

    # Load the quadtree data
    root = load_quadtree(quadtree_file)

    # Randomly sample n_points
    if random_seed is not None:
        np.random.seed(random_seed)
        
    indices = np.random.choice(len(ref_points), size=min(n_points, len(ref_points)), replace=False)
    ref_points_sub = ref_points[indices]
    ref_values_sub = ref_values[indices]
    
    # Evaluate quadtree at subsampled reference points
    qt_at_ref = np.empty(len(ref_points_sub))
    for i, (x, y) in enumerate(ref_points_sub):
        qt_at_ref[i] = root.evaluate(x, y)

    # Compute relative error norms
    error = (qt_at_ref - ref_values_sub) / (np.abs(ref_values_sub) + 1e-12)  # Avoid division by zero
    l1 = np.mean(np.abs(error))
    l2 = np.sqrt(np.mean(error**2))
    linf = np.max(np.abs(error))

    # Compute size of the quadtree file
    size_kb = os.path.getsize(quadtree_file) / 1024

    return l1, l2, linf, size_kb


if __name__ == "__main__":

    # Parameters
    test_file = "../hdf5_data/uniform_evaluations-256.hdf5" # Dense reference data
    tree_file = "tables/quadtree-6-0.0001-128.npz" # Quadtree file to evaluate
    n_random_points = int(1e+5)  # Number of random points to sample for error evaluation

    # Load the test_file to get the test points and reference values
    ref_points, ref_values, ref_interp = load_reference_hdf5(test_file)
    
    # Compute the error norms for the quadtree at random points
    l1, l2, linf, size_kb = compute_quadtree_error(
        tree_file, ref_points, ref_values, 
        n_points=n_random_points,
        random_seed=None
    )

    print("\n+-- Quadtree Relative Error --+")
    print(f"| L1   : {l1:.3e}            |")
    print(f"| L2   : {l2:.3e}            |")
    print(f"| Linf : {linf:.3e}            |")
    print(f"| Size : {size_kb:.2f} kB           |")
    print("+-----------------------------+")
    print(f"Number of random points evaluated: {n_random_points:.1e}\n")
