import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from python.build_quadtree import load_quadtree

def symlog(x, linthresh=1.0, base=10.0):
    x = np.asarray(x)
    return np.sign(x) * np.log1p(np.abs(x) / linthresh) / np.log(base)

def inverse_symlog(y, linthresh=1.0, base=10.0):
    y = np.asarray(y)
    return np.sign(y) * linthresh * (np.expm1(np.abs(y) * np.log(base)))

# Load fine reference data
def load_reference_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        X = f["den"][:]    
        Y = f["temp"][:]   
        F = f["Table_Values"]["f"][:]

    # Extract 1D coordinate vectors (regular grid)
    # den is constant along rows, varies along columns
    # temp is constant along columns, varies along rows
    x_coords = X[0, :]  # 1D array of x coordinates
    y_coords = Y[:, 0]  # 1D array of y coordinates

    # Build fast regular grid interpolator
    ref_interp = RegularGridInterpolator(
        (x_coords, y_coords), 
        F.T,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    # Quick check: evaluate the interpolator at a specific point (for testing purposes)
    # test_x, test_y = 1.3e-12, 1050  # Example density value, 1050
    # interpolated_value = ref_interp((test_x, test_y))
    # print(f"=====\nReference interpolated value at ({test_x:.3e}, {test_y:.3e}): {interpolated_value:.6e}\n=====")
    # input("Press Enter to continue...")  # Pause for user to see the output

    # For compatibility with rest of code, also return flattened points/values
    ref_points = np.column_stack([X.ravel(), Y.ravel()])
    ref_values = F.ravel()

    return ref_points, ref_values, ref_interp

# Compute errors by comparing the quadtree table to the (denser) testing data table
def compute_quadtree_error(
    quadtree_file,
    ref_points,
    ref_values,
    n_points=None,
    random_seed=None,
    rel_floor=1e-6,):
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

    # print("Evaluating sample point:")
    # print(root.evaluate(ref_points[0,0], ref_points[0,1]))

    # Randomly sample n_points
    if random_seed is not None:
        np.random.seed(random_seed)
        
    if n_points is None:
        n_eval = len(ref_points)
    else:
        n_eval = min(int(n_points), len(ref_points))

    indices = np.random.choice(len(ref_points), size=n_eval, replace=False)
    ref_points_sub = ref_points[indices]
    ref_values_sub = ref_values[indices]
    # ref_values_sub = symlog(ref_values_sub)  # Transform reference values to symlog space for error computation
    # print(f"ref_values min: {ref_values_sub.min()}")
    # print(f"ref_values max: {ref_values_sub.max()}")
    
    # Evaluate quadtree at subsampled reference points in the same space used
    # during training: log10 inputs, symlog outputs.
    qt_at_ref = np.empty(len(ref_points_sub))
    for i, (x, y) in enumerate(ref_points_sub):
        qt_at_ref[i] = root.evaluate(x, y)
        

    # Compare in symlog space to match the tree output.
    # ref_values_symlog = symlog(ref_values_sub)
    # qt_at_ref = inverse_symlog(qt_at_ref)
    err = np.abs(qt_at_ref - ref_values_sub) / np.abs(ref_values_sub)
    # print("err min:", err.min())
    # print("err mean:", err.mean())
    # print("err max:", err.max())

    # print("99th percentile:", np.percentile(err, 99))
    # print("99.9th percentile:", np.percentile(err, 99.9))
    l1 = np.mean(np.abs(err))
    l2 = np.sqrt(np.mean(err**2))
    linf = np.max(np.abs(err))

    idx = np.argmax(err)

    x = ref_points_sub[idx,0]
    y = ref_points_sub[idx,1]

    # print("\nWorst point diagnostics:")

    # print("x =", x)
    # print("y =", y)
    # print("tree_symlog =", root.evaluate(x, y))
    # print("tree_raw =", inverse_symlog(root.evaluate(x, y)))
    # print("table_interp_raw =", ref_interp([[x, y]])[0])
    # print("table_interp_symlog =", symlog(ref_interp([[x, y]])[0]))
    # print("reference_raw =", ref_values_sub[idx])
    # print("reference_symlog =", ref_values_symlog[idx])

    # print("reference =", ref_values_sub[idx])
    # print("tree raw  =", root.evaluate(x,y))
    # print("symlog(ref) =", symlog(ref_values_sub[idx]))

    # Compute size of the quadtree file
    size_mb = os.path.getsize(quadtree_file) / (1024 * 1024)

    return l1, l2, linf, size_mb

def plot_error_norms(size_mb, l1, l2, linf):
    order = np.argsort(size_mb)
    size_mb = size_mb[order]
    l1 = l1[order]
    l2 = l2[order]
    linf = linf[order]

    plt.figure(figsize=(8, 6))
    plt.loglog(size_mb, l1, marker="o", linewidth=2, label=r"$L_1$")
    plt.loglog(size_mb, l2, marker="s", linewidth=2, label=r"$L_2$")
    plt.loglog(size_mb, linf, marker="^", linewidth=2, label=r"$L_\infty$")
    plt.xlabel("Table size (MB)")
    plt.ylabel("Relative error norm")
    plt.title("Error vs Table Size")
    plt.figtext(0.5, 0.01, "Consistent D_max = 10, error thresholds = [0.1, 0.01, 0.001, 0.0001, 1e-05] Trained in with symlog error.", ha="center", fontsize=8)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ErrorVsSize-10-symlog-error.png", dpi=500)
    plt.show()

if __name__ == "__main__":

    # Parameters
    test_file = "../hdf5_data/testing_data.hdf5" # Dense reference data
    tree_files = ["tables/quadtree-8-0.01-4321-symlog-error.h5",]
                #   "tables/quadtree-10-0.01-4321-symlog-error.h5",]
                #   "tables/quadtree-10-0.001-4321-symlog-error.h5",
                #   "tables/quadtree-10-0.0001-4321-symlog-error.h5",
                #   "tables/quadtree-10-1e-05-4321-symlog-error.h5",]
                #   "tables/quadtree-10-0.01-4321.h5",
                #   "tables/quadtree-10-0.001-4321.h5",
                #   "tables/quadtree-10-0.0001-4321.h5",]

    n_random_points = int(1e+6)  # Number of random points to sample for error evaluation

    # Load the test_file to get the test points and reference values
    ref_points, ref_values, ref_interp = load_reference_hdf5(test_file)
    l1 = np.zeros(len(tree_files))
    l2 = np.zeros(len(tree_files))
    linf = np.zeros(len(tree_files))
    size_mb = np.zeros(len(tree_files))
    
    # Compute the error norms for the quadtree at random points
    for i in range(len(tree_files)):
        l1[i], l2[i], linf[i], size_mb[i] = compute_quadtree_error(tree_files[i], ref_points, ref_values, n_points=n_random_points, random_seed=1)

        print("\n+-- Quadtree Relative Error --+")
        print(f"| L1   : {l1[i]:.5e}")
        print(f"| L2   : {l2[i]:.5e}")
        print(f"| Linf : {linf[i]:.5e}")
        print(f"| Size : {size_mb[i]:.2f} MB")
        print("+-----------------------------+")
        print(f"Number of random points evaluated: {n_random_points:.1e}")
        print(f"Quadtree file: {tree_files[i]}\n")

    plot_error_norms(size_mb, l1, l2, linf)
