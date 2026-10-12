import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py
from scipy.interpolate import RegularGridInterpolator
from python.build_quadtree import load_quadtree

# Font sizes shared by all plots so figures look consistent on a poster
POSTER_FONTS = dict(title=32, label=26, tick=22)

def _savefig_kwargs(poster):
    # Poster figures get a small margin so the large title isn't flush with the edge
    return dict(bbox_inches="tight", pad_inches=0.2) if poster else {}

def _fonts(poster):
    return POSTER_FONTS if poster else dict(title=None, label=None, tick=None)

def inverse_symlog(y, linthresh=1.0, base=10.0):
    y = np.asarray(y)
    return np.sign(y) * linthresh * (np.expm1(np.abs(y) * np.log(base)))

def _collect_leaf_cells_recursive(group, leaf_cells):
    bounds = group.attrs.get("bounds")
    if bounds is None:
        raise KeyError("Missing 'bounds' attribute in quadtree node.")

    if bool(group.attrs.get("is_leaf", False)):
        xmin, xmax, ymin, ymax = bounds
        leaf_cells.append((float(xmin), float(xmax), float(ymin), float(ymax)))
        return

    for child_key in ("NE", "NW", "SE", "SW"):
        if child_key in group:
            _collect_leaf_cells_recursive(group[child_key], leaf_cells)

def get_leaf_cells(h5_file):
    with h5py.File(h5_file, "r") as data:
        leaf_cells = []

        # New format: recursive groups with per-node attributes.
        if "bounds" in data.attrs:
            _collect_leaf_cells_recursive(data, leaf_cells)
            return leaf_cells

        # Legacy format: flat arrays at root level.
        if all(name in data for name in ("bounds", "is_leaf", "child_indices")):
            bounds = data["bounds"][:]
            is_leaf = data["is_leaf"][:]
            children = data["child_indices"][:]

            stack = [0]
            while stack:
                i = stack.pop()
                xmin, xmax, ymin, ymax = bounds[i]

                if is_leaf[i]:
                    leaf_cells.append((float(xmin), float(xmax), float(ymin), float(ymax)))
                else:
                    for c in children[i]:
                        if c >= 0:
                            stack.append(int(c))

            return leaf_cells

        raise KeyError(
            "Unsupported quadtree file format: expected recursive node attrs or legacy flat datasets."
        )

    return leaf_cells

def plot_decomp(depth, err, cells, poster=False):
    sizes = _fonts(poster)
    fig, ax = plt.subplots(figsize=(15, 6) if poster else (8, 6))

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

    ax.set_xlabel("log10(Density)", fontsize=sizes["label"])
    ax.set_ylabel("log10(Temperature)", fontsize=sizes["label"])
    ax.set_title("Quadtree Decomposition", fontsize=sizes["title"])
    ax.set_aspect("equal")
    if poster:
        ax.tick_params(labelsize=sizes["tick"])

    plt.tight_layout()
    output=f"figs/quadtree_decomp-{depth}-{err}-symlog_err{'_poster' if poster else ''}.png"
    plt.savefig(output, dpi=300, **_savefig_kwargs(poster))
    plt.close()

def plot_heatmap(test_file, output="figs/heatmap.png", poster=False):
    with h5py.File(test_file, "r") as f:
        X_raw = f["den"][:]
        Y_raw = f["temp"][:]
        F = f["Table_Values"]["f"][:]

    X = np.log10(X_raw)
    Y = np.log10(Y_raw)
    F = np.log10(abs(F))

    # Larger fonts for research-poster use
    sizes = _fonts(poster)

    plt.figure(figsize=(15, 6))
    plt.pcolormesh(X, Y, F, shading="auto", cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("log10(|F|)", fontsize=sizes["label"])
    plt.xlabel("log10(Density)", fontsize=sizes["label"])
    plt.ylabel("log10(Temperature)", fontsize=sizes["label"])
    plt.title("Heatmap of Helmholtz Free Energy", fontsize=sizes["title"])
    if poster:
        plt.tick_params(labelsize=sizes["tick"])
        cbar.ax.tick_params(labelsize=sizes["tick"])
    plt.tight_layout()
    plt.savefig(output, dpi=300, **_savefig_kwargs(poster))
    
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

def plot_error_heatmap(depth, err, quadtree_file, test_file, n_points=int(1e6), random_seed=42, poster=False):
    sizes = _fonts(poster)
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
    cbar = plt.colorbar()
    cbar.set_label("log10(relative error)", fontsize=sizes["label"])
    plt.xlabel("log10(Density)", fontsize=sizes["label"])
    plt.ylabel("log10(Temperature)", fontsize=sizes["label"])
    plt.title("Error Heatmap of Quadtree Approximation", fontsize=sizes["title"])
    if poster:
        plt.tick_params(labelsize=sizes["tick"])
        cbar.ax.tick_params(labelsize=sizes["tick"])
    plt.tight_layout()
    output=f"figs/error_heatmap-{depth}-{err}-symlog_err{'_poster' if poster else ''}.png"
    plt.savefig(output, dpi=300, **_savefig_kwargs(poster))
    plt.show()
    plt.close()

if __name__ == "__main__":
    depth = 8
    err = 0.01
    tree_file = f"tables/quadtree-{depth}-{err}-4321-symlog-error.h5"
    test_file = "../hdf5_data/testing_data.hdf5"

    cells = get_leaf_cells(tree_file)
    plot_decomp(depth, err, cells)
    plot_decomp(depth, err, cells, poster=True)
    print(f"\nPlotted quadtree decomposition for depth={depth}, error={err}!")
    plot_heatmap(test_file)
    plot_heatmap(test_file, output="figs/heatmap_poster.png", poster=True)
    plot_error_heatmap(depth, err, tree_file, test_file)
    plot_error_heatmap(depth, err, tree_file, test_file, poster=True)
    # plot_surface(test_file)

    # print(f"{len(cells)} leaf cells plotted.")
