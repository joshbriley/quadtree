import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py
from scipy.interpolate import RegularGridInterpolator
from python.build_quadtree import load_quadtree

def inverse_symlog(y, linthresh=1.0, base=10.0):
    y = np.asarray(y)
    return np.sign(y) * linthresh * (np.expm1(np.abs(y) * np.log(base)))

def get_leaf_cells(h5_file):
    with h5py.File(h5_file, "r") as data:
        bounds = data["bounds"][:]
        is_leaf = data["is_leaf"][:]
        children = data["child_indices"][:]

        leaf_cells = []
        stack = [(0, 0)]  # (node index, depth)

        while stack:
            i, d = stack.pop()
            xmin, xmax, ymin, ymax = bounds[i]

            if is_leaf[i]:
                leaf_cells.append((xmin, xmax, ymin, ymax))
            else:
                for c in children[i]:
                    if c >= 0:
                        stack.append((int(c), d + 1))

    return leaf_cells

def plot_decomp(depth, err, cells):
    fig, ax = plt.subplots(figsize=(8, 6))

    for xmin, xmax, ymin, ymax in cells:
        ax.add_patch(Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="black",
            linewidth=0.5
        ))

    cells = np.array(cells)
    ax.set_xlim(cells[:, 0].min(), cells[:, 1].max())
    ax.set_ylim(cells[:, 2].min(), cells[:, 3].max())

    ax.set_xlabel("log10(Density)")
    ax.set_ylabel("log10(Temperature)")
    ax.set_title("Quadtree Decomposition")
    ax.set_aspect("equal")

    plt.tight_layout()
    output=f"figs/quadtree_decomp-{depth}-{err}-symlog_err.png"
    plt.savefig(output, dpi=300)
    plt.close()

def plot_heatmap(test_file, output="figs/heatmap.png"):
    with h5py.File(test_file, "r") as f:
        X_raw = f["den"][:]
        Y_raw = f["temp"][:]
        F = f["Table_Values"]["f"][:]

    X = np.log10(X_raw)
    Y = np.log10(Y_raw)
    F = np.log10(abs(F))

    plt.figure(figsize=(15, 6))
    plt.pcolormesh(X, Y, F, shading="auto", cmap="viridis")
    plt.colorbar(label="log10(f)")
    plt.xlabel("log10(Density)")
    plt.ylabel("log10(Temperature)")
    plt.title("Heatmap of f")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    
    plt.close()

def plot_surface(test_file, output="figs/surface.png"):
    # To see interactive surface plot I am running on OOD's interactive desktop.    
    with h5py.File(test_file, "r") as f:
        X_raw = f["den"][:]
        Y_raw = f["temp"][:]
        F = f["Table_Values"]["f"][:]

    X = np.log10(X_raw)
    Y = np.log10(Y_raw)
    F = np.log10(abs(F))

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, F, cmap="viridis", edgecolor="none")
    ax.set_xlabel("log10(Density)")
    ax.set_ylabel("log10(Temperature)")
    ax.set_zlabel("log10(f)")
    ax.set_title("Surface plot of f")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
    plt.close()

def plot_error_heatmap(depth, err, quadtree_file, test_file, n_points=int(1e6), random_seed=42):
    # Load quadtree object for robust evaluation
    quadtree = load_quadtree(quadtree_file)

    with h5py.File(test_file, "r") as f:
        X_raw = f["den"][:]
        Y_raw = f["temp"][:]
        F_true = f["Table_Values"]["f"][:]

    # RegularGridInterpolator expects 1D monotonic axes in the same space
    # as the queried coordinates (physical den/temp space here).
    x_coords = X_raw[0, :]
    y_coords = Y_raw[:, 0]

    # Interpolate true values in physical den/temp space.
    true_interp = RegularGridInterpolator(
        (x_coords, y_coords),
        F_true.T,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,)

    # Generate random points in physical domain
    np.random.seed(random_seed)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    print(f"X range: {x_min} to {x_max}")
    print(f"Y range: {y_min} to {y_max}")

    x_random = 10 ** (np.log10(x_min) + (np.log10(x_max) - np.log10(x_min)) * np.random.rand(n_points))
    y_random = 10 ** (np.log10(y_min) + (np.log10(y_max) - np.log10(y_min)) * np.random.rand(n_points))
    print(f"X random range: {x_random.min()} to {x_random.max()}")
    print(f"Y random range: {y_random.min()} to {y_random.max()}")

    # Evaluate quadtree and reference in the same physical space.
    error = np.empty(n_points)
    for i, (x, y) in enumerate(zip(x_random, y_random)):
        value_qt = quadtree.evaluate(x, y)
        true_value = float(true_interp((x, y)))
        rel_err = np.abs(value_qt - true_value) / (np.abs(true_value))
        # print(f"Point {i}: x={x:.3e}, y={y:.3e}, qt={value_qt:.3e}, true={true_value:.3e}, rel_err={rel_err:.3e}")
        error[i] = np.log10(rel_err)

    x_plot = np.log10(x_random)
    y_plot = np.log10(y_random)
    # print(f"X plot range: {x_plot.min()} to {x_plot.max()}")
    # print(f"Y plot range: {y_plot.min()} to {y_plot.max()}")

    # Plot the error heatmap
    plt.figure(figsize=(15, 6))
    plt.scatter(x_plot, y_plot, c=error, cmap="viridis", s=1, alpha=0.5)
    plt.colorbar(label="log10(relative error)")
    plt.xlabel("log10(Density)")
    plt.ylabel("log10(Temperature)")
    plt.title("Error Heatmap of Quadtree Approximation")
    plt.tight_layout()
    output=f"figs/error_heatmap-{depth}-{err}-symlog_err.png"
    plt.savefig(output, dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    depth = 10
    err = 0.0001
    tree_file = f"tables/quadtree-{depth}-{err}-4321-symlog-error.h5"
    test_file = "../hdf5_data/testing_data.hdf5"

    cells = get_leaf_cells(tree_file)
    plot_decomp(depth, err, cells)
    # plot_heatmap(test_file)
    # plot_error_heatmap(depth, err, tree_file, test_file)
    # plot_surface(test_file)

    # print(f"{len(cells)} leaf cells plotted.")
