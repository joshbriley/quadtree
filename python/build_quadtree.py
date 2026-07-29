"""Adaptive quadtree builder for tabulated HDF5 data."""

import os
import sys
import time

import h5py
import numpy as np
from cpp import polyinterp  # HPCC
from scipy.interpolate import RegularGridInterpolator

start_time = time.perf_counter()

# Config parameters
hdf5_file = "../hdf5_data/training_data.hdf5"
error_threshold = 1e-4
# ppc = 4 # Minimum desired points per cell
max_depth = 10  # Maximum depth of quadtree

# with h5py.File(hdf5_file, "r") as f:
#     ds = f["den"][:] # Assuming den, temp, and f are all the same dimension
#     ds_size = ds.size

# max_depth = int((np.log(ds_size/ppc))/np.log(4)) # For training_data.hdf5 this is 10 
# print(f"Max depth calculated from training data: {max_depth}")
# print(f"Error threshold: {error_threshold}\nMax depth: {max_depth}")

def symlog(x, linthresh=1.0, base=10.0):
    x = np.asarray(x)
    return np.sign(x) * np.log1p(np.abs(x) / linthresh) / np.log(base)


def inverse_symlog(y, linthresh=1.0, base=10.0):
    y = np.asarray(y)
    return np.sign(y) * linthresh * np.expm1(np.abs(y) * np.log(base))


# Load HDF5 table
def load_hdf5_table(file_path):
    with h5py.File(file_path, "r") as f:
        x_raw = f["den"][:]
        y_raw = f["temp"][:]
        values = f["Table_Values"]["f"][:]

    x = np.log10(x_raw)
    y = np.log10(y_raw)
    x_coords = x[0, :]
    y_coords = y[:, 0]
    global_values = values.ravel()
    global_points = np.column_stack((x.ravel(), y.ravel()))

    return x_coords, y_coords, global_points, global_values


def evaluate_quad_error(
    xmin, xmax, ymin, ymax,
    global_points, global_values,
    x_coords, y_coords,
    table_interp,
    rel_floor=1e-6,
):
    """Compute error of local polynomial approximation on a cell."""
    nodes = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    x_nodes = xmin + (xmax - xmin) * nodes
    y_nodes = ymin + (ymax - ymin) * nodes
    xx, yy = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    corner_vals = table_interp(np.column_stack((xx.ravel(), yy.ravel()))).reshape(4, 4)

    mask = (
        (global_points[:, 0] >= xmin) & (global_points[:, 0] <= xmax) &
        (global_points[:, 1] >= ymin) & (global_points[:, 1] <= ymax)
    )

    training_pts = global_points[mask]
    training_vals = global_values[mask]

    if training_pts.size == 0:
        raise ValueError("No points in cell for error evaluation.")

    interp_vals = np.array(
        [polyinterp.evaluate_polynomial(corner_vals, x, y, (xmin, xmax, ymin, ymax)) for x, y in training_pts]
    )

    training_vals_symlog = symlog(training_vals)
    err = np.abs(symlog(interp_vals) - training_vals_symlog) / (np.abs(training_vals_symlog) + 1e-12)
    linf_norm = np.max(err)

    return linf_norm, corner_vals

class QuadTreeNode:
    """Stores a quadtree cell with splitting point and polynomial coefficients."""
    
    def __init__(self, bounds, input_space="linear"):
        self.bounds = bounds
        self.input_space = input_space
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
            return polyinterp.evaluate_polynomial(self.coefficients, x, y, self.bounds)

        x_mid, y_mid = self.split_point

        if x >= x_mid:
            child_key = 'NE' if y >= y_mid else 'SE'
        else:
            child_key = 'NW' if y >= y_mid else 'SW'

        return self.children[child_key].evaluate(x, y, assume_internal_coords=True)
    
    def get_leaf_cells(self):
        """Return list of all leaf cells (xmin, xmax, ymin, ymax, depth)."""
        if self.is_leaf:
            depth = self._compute_depth()
            xmin, xmax, ymin, ymax = self.bounds
            return [(xmin, xmax, ymin, ymax, depth)]

        cells = []
        for child in self.children.values():
            cells.extend(child.get_leaf_cells())
        return cells
    
    def _compute_depth(self):
        """Compute depth of this node in the tree."""
        depth = 0
        node = self
        while node.split_point is not None:
            depth += 1
            node = next(iter(node.children.values()))
        return depth
    
    def _flatten_to_list(self, nodes_list, node_index_map):
        """Flatten tree to a list for storage."""
        node_id = len(nodes_list)
        node_index_map[id(self)] = node_id

        child_indices = {key: None for key in ['NE', 'NW', 'SE', 'SW'] if key in self.children}

        nodes_list.append(
            {
                'bounds': self.bounds,
                'is_leaf': self.is_leaf,
                'split_point': self.split_point,
                'coefficients': self.coefficients,
                'child_indices': child_indices,
            }
        )

        for key, child in self.children.items():
            child_indices[key] = child._flatten_to_list(nodes_list, node_index_map)

        return node_id


def _save_quadtree_node(group, node):
    """Store one node as a group and recurse into its children."""
    group.attrs["bounds"] = np.asarray(node.bounds, dtype=np.float32)
    group.attrs["is_leaf"] = bool(node.is_leaf)

    if node.split_point is not None:
        group.attrs["split_point"] = np.asarray(node.split_point, dtype=np.float32)

    if node.coefficients is not None:
        group.create_dataset("coefficients", data=node.coefficients)

    for child_key in ("NE", "NW", "SE", "SW"):
        child = node.children.get(child_key)
        if child is not None:
            child_group = group.create_group(child_key)
            _save_quadtree_node(child_group, child)


def _load_quadtree_node(group, coord_space):
    """Rebuild one node from a group and recurse into its children."""
    node = QuadTreeNode(tuple(group.attrs["bounds"]), input_space=coord_space)
    node.is_leaf = bool(group.attrs.get("is_leaf", False))

    split_point = group.attrs.get("split_point")
    node.split_point = tuple(split_point) if split_point is not None else None

    if "coefficients" in group:
        node.coefficients = group["coefficients"][:]

    for child_key in ("NE", "NW", "SE", "SW"):
        if child_key in group:
            node.children[child_key] = _load_quadtree_node(group[child_key], coord_space)

    return node

def build_quadtree(
    xmin,
    xmax,
    ymin,
    ymax,
    error_threshold,
    max_depth,
    global_points,
    global_values,
    x_coords,
    y_coords,
    table_interp,
    depth=0,
):
    """Recursively build adaptive quadtree based on interpolation error."""

    node = QuadTreeNode((xmin, xmax, ymin, ymax), input_space="log10")

    try:
        error, coeffs = evaluate_quad_error(
            xmin,
            xmax,
            ymin,
            ymax,
            global_points,
            global_values,
            x_coords,
            y_coords,
            table_interp,
        )
    except ValueError:
        print("\n" + "+" * 60)
        print("No points from input table fell in this cell!\nConsider lowering max depth or error threshold.\nExiting program...")
        sys.exit("+" * 60)

    if error <= error_threshold:
        node.is_leaf = True
        node.coefficients = coeffs
        return node

    if depth >= max_depth:
        print(f"MAX DEPTH REACHED | depth={depth} | error={error:.3e}")
        node.is_leaf = True
        node.coefficients = coeffs
        return node

    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2
    node.split_point = (x_mid, y_mid)

    quadrants = {
        'NE': (x_mid, xmax, y_mid, ymax),
        'NW': (xmin, x_mid, y_mid, ymax),
        'SE': (x_mid, xmax, ymin, y_mid),
        'SW': (xmin, x_mid, ymin, y_mid),
    }

    for key, (x1, x2, y1, y2) in quadrants.items():
        node.children[key] = build_quadtree(
            x1,
            x2,
            y1,
            y2,
            error_threshold,
            max_depth,
            global_points,
            global_values,
            x_coords,
            y_coords,
            table_interp,
            depth + 1,
        )

    return node


def save_quadtree(quadtree_root, filepath):
    """Save quadtree to HDF5 file using recursive groups."""
    filepath = os.path.expandvars(filepath)
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        f.attrs["format"] = "quadtree_v1"
        f.attrs["coord_space"] = "log10"
        _save_quadtree_node(f, quadtree_root)

    print(f"Saved quadtree to: {filepath}")


def load_quadtree(filepath):
    """Load quadtree from HDF5 file."""
    with h5py.File(filepath, "r") as data:
        coord_space = str(data.attrs.get("coord_space", "linear"))
        return _load_quadtree_node(data, coord_space)


if __name__ == "__main__":
    x_coords, y_coords, global_points, global_values = load_hdf5_table(hdf5_file)
    table_values = global_values.reshape(len(y_coords), len(x_coords)).T
    table_interp = RegularGridInterpolator(
        (x_coords, y_coords),
        table_values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    domain_xmin, domain_xmax = float(x_coords.min()), float(x_coords.max())
    domain_ymin, domain_ymax = float(y_coords.min()), float(y_coords.max())
    train_resolution = len(x_coords)

    print("Starting Adaptive Quadtree Decomposition...")
    quadtree_root = build_quadtree(
        domain_xmin,
        domain_xmax,
        domain_ymin,
        domain_ymax,
        error_threshold,
        max_depth,
        global_points,
        global_values,
        x_coords,
        y_coords, 
        table_interp,
    )

    quadtree_file = f"/mnt/home/brileyjo/research/quadtree/tables/quadtree-{max_depth}-{error_threshold}-{train_resolution}-symlog-error.h5"
    save_quadtree(quadtree_root, quadtree_file)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
