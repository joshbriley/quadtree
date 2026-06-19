import numpy as np
import h5py
import sys
from scipy.interpolate import LinearNDInterpolator

"""
Adaptive quadtree builder for tabulated (curvilinear) HDF5 data.

Supports:
- irregular / warped grids
- scattered interpolation
- fast local polynomial evaluation
"""

# --------------------------------------------
# Configuration
# --------------------------------------------
SOURCE_TABLE_FILE = "training_data_den0.00625_temp0.00625.hdf5"
ERROR_THRESHOLD = 1e-1
MAX_DEPTH = 7

# fixed nodes for cubic interpolation
NODES = np.array([0.0, 1/3, 2/3, 1.0])


# --------------------------------------------
# Loader (HDF5 / scattered data)
# --------------------------------------------
def load_hdf5_table(file_path):
    with h5py.File(file_path, "r") as f:
        print("Available fields:", list(f["Table_Values"].keys()))

        X = f["den"][:]         # (Nx, Ny)
        Y = f["temp"][:]        # (Nx, Ny)
        F = f["Table_Values"]["f"][:]  # target function

    global_points = np.column_stack([X.ravel(), Y.ravel()])
    global_values = F.ravel()

    print("Building scattered interpolator...")
    source_interpolator = LinearNDInterpolator(global_points, global_values)

    return global_points, global_values, source_interpolator


# --------------------------------------------
# Polynomial evaluation (tensor-product Lagrange)
# --------------------------------------------
def lagrange_basis(t, nodes):
    L = np.ones(4)
    for i in range(4):
        for j in range(4):
            if i != j:
                L[i] *= (t - nodes[j]) / (nodes[i] - nodes[j])
    return L


def evaluate_polynomial(vals, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds

    tx = (x - xmin) / (xmax - xmin)
    ty = (y - ymin) / (ymax - ymin)

    Lx = lagrange_basis(tx, NODES)
    Ly = lagrange_basis(ty, NODES)

    return float(Lx @ vals @ Ly)


# --------------------------------------------
# Error estimator
# --------------------------------------------
def evaluate_quad_error(
    xmin, xmax, ymin, ymax,
    global_points, global_values,
    source_interpolator
):
    num_pts = 4

    x_corner = np.linspace(xmin, xmax, num_pts)
    y_corner = np.linspace(ymin, ymax, num_pts)

    x_mesh, y_mesh = np.meshgrid(x_corner, y_corner, indexing="ij")
    corner_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    # interpolate from scattered data
    corner_vals = source_interpolator(corner_points)

    # handle NaNs (outside convex hull)
    nan_mask = np.isnan(corner_vals)
    if np.any(nan_mask):
        corner_vals[nan_mask] = np.mean(global_values)

    corner_vals = corner_vals.reshape(num_pts, num_pts)

    # filter points inside cell
    gx = global_points[:, 0]
    gy = global_points[:, 1]

    mask = (
        (gx >= xmin) & (gx <= xmax) &
        (gy >= ymin) & (gy <= ymax)
    )

    testing_pts = global_points[mask]
    testing_vals = global_values[mask]

    if testing_pts.shape[0] == 0:
        raise ValueError("No points in cell.")

    # evaluate polynomial
    interp_vals = np.empty(len(testing_pts))
    for i, (x, y) in enumerate(testing_pts):
        interp_vals[i] = evaluate_polynomial(
            corner_vals, x, y, (xmin, xmax, ymin, ymax)
        )

    err = np.abs(interp_vals - testing_vals)
    return np.max(err), corner_vals


# --------------------------------------------
# Quadtree node
# --------------------------------------------
class QuadTreeNode:
    def __init__(self, bounds):
        self.bounds = bounds
        self.split_point = None
        self.values = None  # nodal values (NOT coefficients)
        self.children = {}
        self.is_leaf = False

    def evaluate(self, x, y):
        if self.is_leaf:
            if self.values is None:
                raise ValueError("Leaf missing values")
            return evaluate_polynomial(self.values, x, y, self.bounds)

        x_mid, y_mid = self.split_point

        if x >= x_mid:
            key = "NE" if y >= y_mid else "SE"
        else:
            key = "NW" if y >= y_mid else "SW"

        return self.children[key].evaluate(x, y)

    def get_leaf_cells(self):
        if self.is_leaf:
            xmin, xmax, ymin, ymax = self.bounds
            return [(xmin, xmax, ymin, ymax)]

        cells = []
        for child in self.children.values():
            cells.extend(child.get_leaf_cells())
        return cells

    def _flatten_to_list(self, nodes_list, node_index_map):
        node_id = len(nodes_list)
        node_index_map[id(self)] = node_id

        child_indices = {}

        for key in ['NE', 'NW', 'SE', 'SW']:
            if key in self.children:
                child_indices[key] = None

        nodes_list.append({
            'bounds': self.bounds,
            'is_leaf': self.is_leaf,
            'split_point': self.split_point,
            'values': self.values,
            'child_indices': child_indices,
        })

        for key, child in self.children.items():
            child_indices[key] = child._flatten_to_list(nodes_list, node_index_map)

        return node_id


# --------------------------------------------
# Quadtree builder
# --------------------------------------------
def build_quadtree(
    xmin, xmax, ymin, ymax,
    error_threshold, max_depth,
    global_points, global_values,
    source_interpolator,
    depth=0
):
    node = QuadTreeNode((xmin, xmax, ymin, ymax))

    try:
        error, vals = evaluate_quad_error(
            xmin, xmax, ymin, ymax,
            global_points, global_values,
            source_interpolator
        )
    except ValueError:
        # No data in this cell → stop refinement
        node.is_leaf = True

        # still assign values using sampling (fallback)
        num_pts = 4
        x_corner = np.linspace(xmin, xmax, num_pts)
        y_corner = np.linspace(ymin, ymax, num_pts)

        x_mesh, y_mesh = np.meshgrid(x_corner, y_corner, indexing="ij")
        corner_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

        vals = source_interpolator(corner_points)

        # handle NaNs
        nan_mask = np.isnan(vals)
        if np.any(nan_mask):
            vals[nan_mask] = np.mean(global_values)

        node.values = vals.reshape(4, 4)

    return node


    if error <= error_threshold or depth >= max_depth:
        node.is_leaf = True
        node.values = vals
        return node

    x_mid = 0.5 * (xmin + xmax)
    y_mid = 0.5 * (ymin + ymax)
    node.split_point = (x_mid, y_mid)

    quads = {
        "NE": (x_mid, xmax, y_mid, ymax),
        "NW": (xmin, x_mid, y_mid, ymax),
        "SE": (x_mid, xmax, ymin, y_mid),
        "SW": (xmin, x_mid, ymin, y_mid),
    }

    for key, (x1, x2, y1, y2) in quads.items():
        node.children[key] = build_quadtree(
            x1, x2, y1, y2,
            error_threshold, max_depth,
            global_points, global_values,
            source_interpolator,
            depth + 1
        )

    return node


# --------------------------------------------
# Save / load
# --------------------------------------------
def save_quadtree(root, filepath):
    nodes_list = []
    root._flatten_to_list(nodes_list, {})

    bounds = []
    is_leaf = []
    split_points = []
    child_indices = []

    values_dict = {}

    for i, node in enumerate(nodes_list):
        bounds.append(node["bounds"])
        is_leaf.append(node["is_leaf"])

        split_points.append(
            node["split_point"] if node["split_point"] else [0, 0]
        )

        children = node["child_indices"]
        child_indices.append([
            children.get("NE", -1),
            children.get("NW", -1),
            children.get("SE", -1),
            children.get("SW", -1),
        ])

        if node["values"] is not None:
            values_dict[f"vals_{i}"] = node["values"]

    np.savez_compressed(
        filepath,
        bounds=np.array(bounds, dtype=np.float64),
        is_leaf=np.array(is_leaf, dtype=bool),
        split_points=np.array(split_points, dtype=np.float64),
        child_indices=np.array(child_indices, dtype=int),
        **values_dict
    )

    print(f"Saved quadtree to {filepath}")

def load_quadtree(filepath):
    """
    Load quadtree from NPZ and reconstruct full tree.
    """
    data = np.load(filepath, allow_pickle=True)

    bounds = data["bounds"]
    is_leaf = data["is_leaf"]
    split_points = data["split_points"]
    child_indices = data["child_indices"]

    # --------------------------------------------------
    # Step 1: Create all nodes
    # --------------------------------------------------
    nodes = []
    for i in range(len(bounds)):
        node = QuadTreeNode(tuple(bounds[i]))
        node.is_leaf = bool(is_leaf[i])

        if not node.is_leaf:
            node.split_point = tuple(split_points[i])

        # restore 4x4 values
        key = f"vals_{i}"
        if key in data:
            node.values = data[key]

        nodes.append(node)

    # --------------------------------------------------
    # Step 2: Reconnect children
    # --------------------------------------------------
    for i, (ne, nw, se, sw) in enumerate(child_indices):
        if ne >= 0:
            nodes[i].children["NE"] = nodes[ne]
        if nw >= 0:
            nodes[i].children["NW"] = nodes[nw]
        if se >= 0:
            nodes[i].children["SE"] = nodes[se]
        if sw >= 0:
            nodes[i].children["SW"] = nodes[sw]

    # Fix invalid nodes
    for node in nodes:
        if not node.children:
            node.is_leaf = True

    return nodes[0] if nodes else None


# --------------------------------------------
# Main
# --------------------------------------------
if __name__ == "__main__":
    global_points, global_values, source_interpolator = load_hdf5_table(
        SOURCE_TABLE_FILE
    )

    xmin, xmax = global_points[:, 0].min(), global_points[:, 0].max()
    ymin, ymax = global_points[:, 1].min(), global_points[:, 1].max()

    print("Building quadtree...")

    root = build_quadtree(
        xmin, xmax, ymin, ymax,
        ERROR_THRESHOLD, MAX_DEPTH,
        global_points, global_values,
        source_interpolator
    )

    outfile = f"tables/quadtree-{MAX_DEPTH}-{ERROR_THRESHOLD}.npz"
    save_quadtree(root, outfile)
