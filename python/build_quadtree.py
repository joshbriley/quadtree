"""Adaptive quadtree builder for tabulated HDF5 data."""

import os
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import count
from typing import NamedTuple

import h5py
import numpy as np
from cpp import polyinterp  # HPCC
from scipy.interpolate import RectBivariateSpline
from python.common import symlog, inverse_symlog, load_hdf5_table

# Config parameters
hdf5_file = "../hdf5_data/training_data.hdf5"
error_threshold = 1e-2
max_depth = 8
# Leaves interpolate symlog(f, value_linthresh). It must exceed |f| near where f changes sign,
# otherwise symlog(f) jumps there and the interpolant overshoots wildly.
value_linthresh = 1e21

FORMAT = "quadtree_v3"
INTERPOLANT = "bicubic_hermite_c1_symlog"

# Corner (a, b): a selects xmin/xmax, b selects ymin/ymax.
CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))
QUADRANTS = (("SW", (0, 0)), ("SE", (1, 0)), ("NW", (0, 1)), ("NE", (1, 1)))


def hermite_basis(x, lo, hi, deriv=0):
    """Cubic Hermite basis [value@lo, value@hi, slope@lo, slope@hi] at x (or its d/dx).

    Must match hermite_basis() in cpp/polyinterp.cpp.
    """
    h = hi - lo
    t = (np.asarray(x, dtype=float) - lo) / h
    if deriv == 0:
        cols = (2 * t**3 - 3 * t**2 + 1, 3 * t**2 - 2 * t**3, h * (t**3 - 2 * t**2 + t), h * (t**3 - t**2))
    else:
        cols = ((6 * t**2 - 6 * t) / h, (6 * t - 6 * t**2) / h, 3 * t**2 - 4 * t + 1, 3 * t**2 - 2 * t)
    return np.stack(cols, axis=-1)


def hermite_matrix(corner_data):
    """4x4 Hermite matrix from (f, f_x, f_y, f_xy) at each corner, in CORNERS order.

    Row/column index 0/1 is the value at the lower/upper bound, 2/3 the derivative there.
    """
    F = np.empty((4, 4))
    for (a, b), (f, fx, fy, fxy) in zip(CORNERS, corner_data):
        F[a, b], F[2 + a, b], F[a, 2 + b], F[2 + a, 2 + b] = f, fx, fy, fxy
    return F


def patch_derivatives(F, bounds, x, y):
    """(f, f_x, f_y, f_xy) of the Hermite patch F on bounds at the point (x, y)."""
    xmin, xmax, ymin, ymax = bounds
    bx, dbx = hermite_basis(x, xmin, xmax), hermite_basis(x, xmin, xmax, deriv=1)
    by, dby = hermite_basis(y, ymin, ymax), hermite_basis(y, ymin, ymax, deriv=1)
    return np.array([bx @ F @ by, dbx @ F @ by, bx @ F @ dby, dbx @ F @ dby])


class QuadTreeNode:
    """Stores a quadtree cell with splitting point and Hermite coefficients of symlog(f, value_linthresh)."""

    def __init__(self, bounds, input_space="linear", value_linthresh=1.0):
        self.bounds = bounds
        self.input_space = input_space
        self.value_linthresh = value_linthresh
        self.split_point = None
        self.coefficients = None
        self.children = {}
        self.is_leaf = False

    def evaluate(self, x, y, assume_internal_coords=False):
        """Navigate tree and evaluate polynomial at (x, y)."""
        if not assume_internal_coords and self.input_space == "log10":
            x = np.log10(x)
            y = np.log10(y)

        if self.is_leaf:
            if self.coefficients is None:
                raise ValueError("Leaf node has no coefficients")
            g = polyinterp.evaluate_hermite(self.coefficients, x, y, self.bounds)
            return float(inverse_symlog(g, self.value_linthresh))

        x_mid, y_mid = self.split_point

        if x >= x_mid:
            child_key = 'NE' if y >= y_mid else 'SE'
        else:
            child_key = 'NW' if y >= y_mid else 'SW'

        return self.children[child_key].evaluate(x, y, assume_internal_coords=True)


class Leaf(NamedTuple):
    """Leaf during the build; (i, j) indexes it among the 2**level x 2**level cells."""
    node: QuadTreeNode
    level: int
    i: int
    j: int


def _corner_keys(leaf, max_depth):
    """Corners as exact integer coordinates on the finest (2**max_depth) grid, in CORNERS order."""
    s = 2 ** (max_depth - leaf.level)
    return [((leaf.i + a) * s, (leaf.j + b) * s) for a, b in CORNERS]


def assign_hermite_coefficients(leaves, spline, max_depth):
    """Set every leaf's Hermite matrix so g and its gradient are continuous across all cell edges.

    Neighbours share corner data, which makes equal-size neighbours C1. A corner lying inside a
    coarser leaf's edge (hanging node) takes its data from that leaf's patch instead of the spline,
    so the finer cells reproduce the coarse edge's cubic and normal derivative exactly.
    """
    leaves = sorted(leaves, key=lambda leaf: leaf.level)

    position = {}
    for leaf in leaves:
        xmin, xmax, ymin, ymax = leaf.node.bounds
        for (a, b), key in zip(CORNERS, _corner_keys(leaf, max_depth)):
            position[key] = ((xmin, xmax)[a], (ymin, ymax)[b])

    vlines, hlines = defaultdict(list), defaultdict(list)
    for X, Y in position:
        vlines[X].append(Y)
        hlines[Y].append(X)
    for line in (*vlines.values(), *hlines.values()):
        line.sort()

    def strictly_between(line, lo, hi):
        return line[bisect_right(line, lo):bisect_left(line, hi)]

    hanging_on = []
    for leaf in leaves:
        (X0, Y0), _, _, (X1, Y1) = _corner_keys(leaf, max_depth)
        keys = [(X, Y) for X in (X0, X1) for Y in strictly_between(vlines[X], Y0, Y1)]
        keys += [(X, Y) for Y in (Y0, Y1) for X in strictly_between(hlines[Y], X0, X1)]
        hanging_on.append(keys)

    hanging = {key for keys in hanging_on for key in keys}
    free = [key for key in position if key not in hanging]
    x, y = np.array([position[key] for key in free]).T
    spline_data = np.column_stack([spline.ev(x, y, dx=dx, dy=dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))])
    data = dict(zip(free, spline_data))

    # A hanging node only depends on strictly coarser leaves, so coarse-to-fine order resolves all.
    for leaf, keys in zip(leaves, hanging_on):
        leaf.node.coefficients = hermite_matrix([data[key] for key in _corner_keys(leaf, max_depth)])
        for key in keys:
            data[key] = patch_derivatives(leaf.node.coefficients, leaf.node.bounds, *position[key])


def leaf_error(node, x_coords, y_coords, table_symlog):
    """Max relative symlog error of the leaf's Hermite patch over the table points in the cell."""
    xmin, xmax, ymin, ymax = node.bounds
    ix = slice(np.searchsorted(x_coords, xmin, "left"), np.searchsorted(x_coords, xmax, "right"))
    iy = slice(np.searchsorted(y_coords, ymin, "left"), np.searchsorted(y_coords, ymax, "right"))
    true_vals = table_symlog[ix, iy]

    # Prevent the QT from over-refining where there is no data:
    if true_vals.size == 0:
        raise ValueError("No points in cell for error evaluation.")

    approx = hermite_basis(x_coords[ix], xmin, xmax) @ node.coefficients @ hermite_basis(y_coords[iy], ymin, ymax).T
    approx = symlog(inverse_symlog(approx, node.value_linthresh))
    return np.max(np.abs(approx - true_vals) / np.abs(true_vals))


def _subdivide(node):
    """Make node internal with four leaf children (QUADRANTS order); returns [(child, a, b)].

    Loading reuses this, so reconstructed bounds are bit-identical to the built ones.
    """
    xmin, xmax, ymin, ymax = node.bounds
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2
    node.split_point = (x_mid, y_mid)
    node.is_leaf = False
    node.coefficients = None

    children = []
    for key, (a, b) in QUADRANTS:
        bounds = ((xmin, x_mid), (x_mid, xmax))[a] + ((ymin, y_mid), (y_mid, ymax))[b]
        child = QuadTreeNode(bounds, input_space=node.input_space, value_linthresh=node.value_linthresh)
        child.is_leaf = True
        node.children[key] = child
        children.append((child, a, b))
    return children


def _split(leaf):
    return [Leaf(child, leaf.level + 1, 2 * leaf.i + a, 2 * leaf.j + b) for child, a, b in _subdivide(leaf.node)]


def preorder(node):
    """Yield node and its descendants depth-first, children in QUADRANTS order."""
    yield node
    for key, _ in QUADRANTS:
        if key in node.children:
            yield from preorder(node.children[key])


def build_quadtree(x_coords, y_coords, table_values, error_threshold, max_depth, value_linthresh):
    """Adaptively refine until every leaf's C1 Hermite patch meets the error threshold.

    table_values has shape (len(x_coords), len(y_coords)). Leaves interpolate
    g = symlog(f, value_linthresh); corner values and derivatives come from a global bicubic
    interpolating spline of g. Hanging-node constraints mean refining one cell can change its
    neighbours' patches, so all leaves are re-checked after every pass.
    """
    spline = RectBivariateSpline(x_coords, y_coords, symlog(table_values, value_linthresh), kx=3, ky=3, s=0)
    table_symlog = symlog(table_values)

    bounds = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))
    root = QuadTreeNode(bounds, input_space="log10", value_linthresh=value_linthresh)
    root.is_leaf = True
    leaves = [Leaf(root, 0, 0, 0)]

    for refinement_pass in count(1):
        assign_hermite_coefficients(leaves, spline, max_depth)

        try:
            errors = [leaf_error(leaf.node, x_coords, y_coords, table_symlog) for leaf in leaves]
        except ValueError:
            print("\n" + "+" * 60)
            print("No points from input table fell in this cell!\nConsider lowering max depth or error threshold.\nExiting program...")
            sys.exit("+" * 60)

        refine = [error > error_threshold and leaf.level < max_depth for leaf, error in zip(leaves, errors)]
        print(f"Pass {refinement_pass}: {len(leaves)} leaves, refining {sum(refine)}")
        if not any(refine):
            break

        leaves = [child for leaf, r in zip(leaves, refine) for child in (_split(leaf) if r else [leaf])]

    unresolved = [error for error in errors if not error <= error_threshold]
    if unresolved:
        print(f"MAX DEPTH REACHED | {len(unresolved)} leaves above threshold | worst error={max(unresolved):.3e}")

    return root


def save_quadtree(quadtree_root, filepath):
    """Save quadtree to HDF5 as flat arrays.

    is_leaf: one flag per node in preorder (children in QUADRANTS order). Every internal node
    has exactly four children, so these flags alone define the tree; bounds and split points
    are recomputed from the root bounds on load.
    coefficients: the leaves' 4x4 Hermite matrices, in the same preorder.
    """
    filepath = os.path.expandvars(filepath)
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    nodes = list(preorder(quadtree_root))
    with h5py.File(filepath, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["interpolant"] = INTERPOLANT
        f.attrs["coord_space"] = quadtree_root.input_space
        f.attrs["value_linthresh"] = quadtree_root.value_linthresh
        f.attrs["bounds"] = np.asarray(quadtree_root.bounds, dtype=np.float64)
        f.create_dataset("is_leaf", data=np.array([node.is_leaf for node in nodes], dtype=np.uint8))
        f.create_dataset("coefficients", data=np.array([node.coefficients for node in nodes if node.is_leaf]))

    print(f"Saved quadtree to: {filepath}")


def load_quadtree(filepath):
    """Load quadtree from HDF5 file."""
    with h5py.File(filepath, "r") as data:
        stored = (data.attrs.get("format"), data.attrs.get("interpolant", "cubic_lagrange"))
        if stored != (FORMAT, INTERPOLANT):
            raise ValueError(f"{filepath} is {stored}; this code reads {(FORMAT, INTERPOLANT)}. Rebuild the tree.")
        is_leaf = iter(data["is_leaf"][:].astype(bool))
        coefficients = iter(data["coefficients"][:])
        root = QuadTreeNode(tuple(map(float, data.attrs["bounds"])), input_space=str(data.attrs["coord_space"]),
                            value_linthresh=float(data.attrs["value_linthresh"]))

    def fill(node):
        if next(is_leaf):
            node.is_leaf = True
            node.coefficients = next(coefficients)
        else:
            for child, _, _ in _subdivide(node):
                fill(child)

    fill(root)
    return root


if __name__ == "__main__":
    x_coords, y_coords, _, global_values = load_hdf5_table(hdf5_file)
    table_values = global_values.reshape(len(y_coords), len(x_coords)).T
    train_resolution = len(x_coords)

    print("Starting Adaptive Quadtree Decomposition...")
    quadtree_root = build_quadtree(x_coords, y_coords, table_values, error_threshold, max_depth, value_linthresh)

    quadtree_file = f"/mnt/home/brileyjo/research/quadtree/tables/quadtree-{max_depth}-{error_threshold}-{train_resolution}-hermite.h5"
    save_quadtree(quadtree_root, quadtree_file)
