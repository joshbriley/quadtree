import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import h5py
from scipy.interpolate import RegularGridInterpolator
from python.build_quadtree import FORMAT as QUADTREE_FORMAT, load_quadtree, hermite_basis, preorder
from python.common import inverse_symlog as _inverse_symlog

# Font sizes shared by all plots so figures look consistent on a poster
POSTER_FONTS = dict(title=32, label=26, tick=22)

# Palette: Hermite / Lagrange keep the same colour in every figure.
HERMITE_COLOR = "#2a78d6"
LAGRANGE_COLOR = "#eb6834"
SIGN_CHANGE_COLOR = "#e34948"
TRUTH_COLOR = "#c3c2b7"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID_COLOR = "#e1e0d9"

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
        flat_hermite = data.attrs.get("format") == QUADTREE_FORMAT
    if flat_hermite:
        return [tuple(map(float, node.bounds)) for node in preorder(load_quadtree(h5_file)) if node.is_leaf]

    with h5py.File(h5_file, "r") as data:
        leaf_cells = []

        # Earlier format: recursive groups with per-node attributes.
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

def load_log_table(h5_file):
    """(log10 den, log10 temp, f) 2D arrays from a table file."""
    with h5py.File(h5_file, "r") as f:
        return np.log10(f["den"][:]), np.log10(f["temp"][:]), f["Table_Values"]["f"][:]

def plot_decomp(depth, err, cells, poster=False, sign_change=None):
    """Leaf cells; sign_change=(X, Y, F) from load_log_table also draws the f = 0 curve."""
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

    if sign_change is not None:
        ax.contour(*sign_change, levels=[0.0], colors=SIGN_CHANGE_COLOR, linewidths=2 if poster else 1.5)
        ax.legend(handles=[Line2D([], [], color=SIGN_CHANGE_COLOR, lw=2, label="f = 0 (sign change)")],
                  loc="upper left", fontsize=sizes["tick"], framealpha=0.9)

    cells = np.array(cells)
    ax.set_xlim(cells[:, 0].min(), cells[:, 1].max())
    ax.set_ylim(cells[:, 2].min(), cells[:, 3].max())

    ax.set_xlabel("log10(Density)", fontsize=sizes["label"])
    ax.set_ylabel("log10(Temperature)", fontsize=sizes["label"])
    ax.set_title(f"Quadtree Decomposition ({len(cells)} leaves)", fontsize=sizes["title"])
    ax.set_aspect("equal")
    if poster:
        ax.tick_params(labelsize=sizes["tick"])

    plt.tight_layout()
    output=f"figs/quadtree_decomp-{depth}-{err}-hermite{'_poster' if poster else ''}.png"
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
    output=f"figs/error_heatmap-{depth}-{err}-hermite{'_poster' if poster else ''}.png"
    plt.savefig(output, dpi=300, **_savefig_kwargs(poster))
    plt.show()
    plt.close()

_LAGRANGE_NODES = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

def _lagrange_basis(x, lo, hi):
    t = (np.asarray(x, dtype=float) - lo) / (hi - lo)
    return np.stack([
        np.prod([(t - _LAGRANGE_NODES[j]) / (_LAGRANGE_NODES[i] - _LAGRANGE_NODES[j]) for j in range(4) if j != i], axis=0)
        for i in range(4)], axis=-1)

def _leaves_on_row(group, y0, leaves):
    xmin, xmax, ymin, ymax = map(float, group.attrs["bounds"])
    if not ymin <= y0 < ymax:
        return
    if bool(group.attrs.get("is_leaf", False)):
        leaves.append(((xmin, xmax, ymin, ymax), group["coefficients"][:]))
    for child_key in ("NE", "NW", "SE", "SW"):
        if child_key in group:
            _leaves_on_row(group[child_key], y0, leaves)

def tree_log_slope_on_row(tree_file, y0, x_range, n_per_leaf=200):
    """d ln|f| / d ln(den) of a saved tree along log10(temp) = y0, one segment per leaf.

    Works for Hermite trees and for the earlier cubic Lagrange trees (nested-group format).
    Each leaf is differentiated with its own polynomial, so jumps between leaves are kept
    rather than smoothed over.
    """
    with h5py.File(tree_file, "r") as f:
        hermite = f.attrs.get("format") == QUADTREE_FORMAT
        leaves = []
        if not hermite:
            _leaves_on_row(f, y0, leaves)
    if hermite:
        root = load_quadtree(tree_file)
        linthresh = root.value_linthresh
        leaves = [(tuple(map(float, node.bounds)), node.coefficients) for node in preorder(root)
                  if node.is_leaf and node.bounds[2] <= y0 < node.bounds[3]]

    segments = []
    for (xmin, xmax, ymin, ymax), C in sorted(leaves, key=lambda leaf: leaf[0]):
        lo, hi = max(xmin, x_range[0]), min(xmax, x_range[1])
        if lo >= hi:
            continue
        if hermite:
            def leaf_f(xs):
                return _inverse_symlog(hermite_basis(xs, xmin, xmax) @ C @ hermite_basis(y0, ymin, ymax), linthresh)
        else:
            def leaf_f(xs):
                return _lagrange_basis(xs, xmin, xmax) @ C @ _lagrange_basis(y0, ymin, ymax)
        xs = np.linspace(lo, hi, n_per_leaf)
        h = 1e-6
        slope = (leaf_f(xs + h) - leaf_f(xs - h)) / (2 * h) / (leaf_f(xs) * np.log(10))
        segments.append((xs, slope, (xmin, xmax)))
    return segments

def plot_slope_slice(depth, err, lagrange_file, hermite_file, test_file, log_temp, x_range, poster=False):
    """Compare each tree's d ln|f|/d ln(den) with the table's analytic derivative along one temperature."""
    sizes = _fonts(poster)
    with h5py.File(test_file, "r") as f:
        log_den = np.log10(f["den"][0, :])
        row = int(np.argmin(np.abs(np.log10(f["temp"][:, 0]) - log_temp)))
        y0 = float(np.log10(f["temp"][row, 0]))
        F = f["Table_Values"]["f"][row, :]
        true_slope = 10.0 ** log_den * f["Table_Values"]["df_dden"][row, :] / F
    keep = (log_den >= x_range[0]) & (log_den <= x_range[1])

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(15, 9) if poster else (10, 6.5))
    panels = ((axes[0], lagrange_file, LAGRANGE_COLOR, "Cubic Lagrange", "previous"),
              (axes[1], hermite_file, HERMITE_COLOR, "Bicubic Hermite", "C1"))
    all_slopes = [true_slope[keep]]
    for ax, tree_file, color, name, note in panels:
        segments = tree_log_slope_on_row(tree_file, y0, x_range)
        all_slopes += [slope for _, slope, _ in segments]
        for _, _, (xmin, xmax) in segments:
            ax.axvline(xmin, color=GRID_COLOR, lw=0.8, zorder=0)
        ax.plot(log_den[keep], true_slope[keep], color=TRUTH_COLOR, lw=4, solid_capstyle="round", zorder=1)
        for xs, slope, _ in segments:
            ax.plot(xs, slope, color=color, lw=2 if poster else 1.5, zorder=2)
        jumps = np.abs([a[1][-1] - b[1][0] for a, b in zip(segments, segments[1:])])
        ax.set_title(f"{name} ({note}): largest slope jump at a cell edge = {jumps.max():.1e}",
                     loc="left", fontsize=sizes["label"], color=INK)
        ax.legend(handles=[Line2D([], [], color=color, lw=2, label=name),
                           Line2D([], [], color=TRUTH_COLOR, lw=4, label="Test table (analytic)"),
                           Line2D([], [], color=GRID_COLOR, lw=1.5, label="Leaf edge")],
                  loc="upper right", fontsize=sizes["tick"], frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=sizes["tick"], colors=SECONDARY_INK)
    axes[1].set_xlabel("log10(Density)", fontsize=sizes["label"], color=SECONDARY_INK)
    axes[1].set_xlim(*x_range)
    all_slopes = np.concatenate(all_slopes)
    lo, hi = all_slopes.min(), all_slopes.max()
    axes[1].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    fig.suptitle(f"Slope along log10(Temperature) = {y0:.2f}", fontsize=sizes["title"], color=INK)
    fig.supylabel("d ln|f| / d ln(Density)", fontsize=sizes["label"], color=SECONDARY_INK)

    plt.tight_layout()
    output = f"figs/slope_slice-{depth}-{err}-hermite{'_poster' if poster else ''}.png"
    plt.savefig(output, dpi=300, **_savefig_kwargs(poster))
    plt.close()

if __name__ == "__main__":
    depth = 8
    err = 0.01
    tree_file = f"tables/quadtree-{depth}-{err}-4321-hermite.h5"
    lagrange_file = f"tables/quadtree-{depth}-{err}-4321-symlog-error.h5"
    test_file = "../hdf5_data/testing_data.hdf5"

    cells = get_leaf_cells(tree_file)
    sign_change = load_log_table(test_file)
    plot_decomp(depth, err, cells, sign_change=sign_change)
    plot_decomp(depth, err, cells, poster=True, sign_change=sign_change)
    print(f"\nPlotted quadtree decomposition for depth={depth}, error={err}!")
    plot_heatmap(test_file)
    plot_heatmap(test_file, output="figs/heatmap_poster.png", poster=True)
    plot_error_heatmap(depth, err, tree_file, test_file)
    plot_error_heatmap(depth, err, tree_file, test_file, poster=True)
    plot_slope_slice(depth, err, lagrange_file, tree_file, test_file, log_temp=5.0, x_range=(1.0, 13.5))
    plot_slope_slice(depth, err, lagrange_file, tree_file, test_file, log_temp=5.0, x_range=(1.0, 13.5), poster=True)
    # plot_surface(test_file)

    # print(f"{len(cells)} leaf cells plotted.")
