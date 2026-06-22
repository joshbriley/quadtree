import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import polyinterp 
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
import h5py


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
SOURCE_TABLE_FILE = "training_data.hdf5"
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

def evaluate_quad_error(
    xmin, xmax, ymin, ymax,
    global_points, global_values,
    source_interpolator
):
    """
    Compute error of local polynomial approximation on a cell.
    """

    # --------------------------------------------------
    # Step 1: build corner grid (4x4)
    # --------------------------------------------------
    num_of_points = 4
    x_corner = np.linspace(xmin, xmax, num_of_points)
    y_corner = np.linspace(ymin, ymax, num_of_points)

    x_mesh, y_mesh = np.meshgrid(x_corner, y_corner, indexing="ij")
    corner_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    # Ground truth at grid nodes (this is fine to keep)
    corner_vals = source_interpolator(corner_points).reshape(num_of_points, num_of_points)

    # --------------------------------------------------
    # Step 2: find testing points inside cell
    # --------------------------------------------------
    x_coords = global_points[:, 0]
    y_coords = global_points[:, 1]

    mask = (
        (x_coords >= xmin) & (x_coords <= xmax) &
        (y_coords >= ymin) & (y_coords <= ymax)
    )

    testing_pts = global_points[mask]
    testing_vals = global_values[mask]

    if testing_pts.shape[0] == 0:
        raise ValueError("No points found in cell for error evaluation.")

    # --------------------------------------------------
    # Step 3: evaluate polynomial (FIXED PART)
    # --------------------------------------------------
    interp_vals = np.array([
        polyinterp.evaluate_polynomial(corner_vals, x, y, (xmin, xmax, ymin, ymax))
        for x, y in testing_pts
    ])

    # --------------------------------------------------
    # Step 4: compute error
    # --------------------------------------------------
    err = np.abs(interp_vals - testing_vals)
    linf_norm = np.max(err)

    return linf_norm, corner_vals

class QuadTreeNode:
    """Stores a quadtree cell with splitting point and polynomial coefficients."""
    
    def __init__(self, bounds):
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        self.split_point = None  # (x_mid, y_mid)
        self.coefficients = None  # Polynomial coeffs if leaf
        self.children = {}  # {'NE': Node, 'NW': Node, 'SE': Node, 'SW': Node}
        self.is_leaf = False
    
    def evaluate(self, x, y):
        """Navigate tree and evaluate polynomial at (x, y)."""
        if self.is_leaf:
            if self.coefficients is None:
                raise ValueError("Leaf node has no coefficients")
            return polyinterp.evaluate_polynomial(self.coefficients, x, y, self.bounds)
        
        # Navigate to correct child quadrant
        xmin, xmax, ymin, ymax = self.bounds
        x_mid, y_mid = self.split_point
        
        if x >= x_mid:
            child_key = 'NE' if y >= y_mid else 'SE'
        else:
            child_key = 'NW' if y >= y_mid else 'SW'
        
        return self.children[child_key].evaluate(x, y)
    
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
            node = list(node.children.values())[0]
        return depth
    
    def _flatten_to_list(self, nodes_list, node_index_map):
        """Flatten tree to a list for storage."""
        node_id = len(nodes_list)
        node_index_map[id(self)] = node_id
        
        child_indices = {}
        for key in ['NE', 'NW', 'SE', 'SW']:
            if key in self.children:
                child_indices[key] = None  # Placeholder, will fill after recursion
        
        nodes_list.append({
            'bounds': self.bounds,
            'is_leaf': self.is_leaf,
            'split_point': self.split_point,
            'coefficients': self.coefficients,
            'child_indices': child_indices
        })
        
        for key, child in self.children.items():
            child_indices[key] = child._flatten_to_list(nodes_list, node_index_map)
        
        return node_id

# Quadtree Builder -- Must be able to export the quadtree structure and the interpolation polynomials at each leaf node for in-situ surrogate evaluation.
def build_quadtree(
    xmin,
    xmax,
    ymin,
    ymax,
    error_threshold,
    max_depth,
    global_points,
    global_valuesuess,
    source_interpolator,
    depth=0,
):
    """Recursively build adaptive quadtree based on interpolation error."""
    
    node = QuadTreeNode((xmin, xmax, ymin, ymax))
    
    # Evaluate error for this cell
    try:
        error, coeffs = evaluate_quad_error(
            xmin,
            xmax,
            ymin,
            ymax,
            global_points,
            global_valuesuess,
            source_interpolator,
        )
    except ValueError:
        # No table samples fell inside this cell.        
        print("\n" + "+" * 60)
        print("No points from input table fell in this cell!\nConsider lowering max depth or error threshold, or increase training resolution. Exiting program...")
        sys.exit("+" * 60)
    
    # Check stopping criteria
    if error <= error_threshold or depth >= max_depth:
        node.is_leaf = True
        node.coefficients = coeffs
        return node
    
    # Split cell into 4 quadrants
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2
    node.split_point = (x_mid, y_mid)
    
    quadrants = {
        'NE': (x_mid, xmax, y_mid, ymax),
        'NW': (xmin, x_mid, y_mid, ymax),
        'SE': (x_mid, xmax, ymin, y_mid),
        'SW': (xmin, x_mid, ymin, y_mid)
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
            global_valuesuess,
            source_interpolator,
            depth + 1,
        )
    
    return node

def save_quadtree(quadtree_root, filepath):
    """Save quadtree to compressed NPZ file using flattened structure."""
    nodes_list = []
    quadtree_root._flatten_to_list(nodes_list, {})
    
    # Flatten node data into numpy arrays
    bounds_list = []
    is_leaf_list = []
    split_points_list = []
    child_indices_list = []
    
    for node_data in nodes_list:
        bounds_list.append(node_data['bounds'])
        is_leaf_list.append(node_data['is_leaf'])
        split_points_list.append(node_data['split_point'] if node_data['split_point'] else [0, 0])
        
        # Encode children as 4-tuple (NE, NW, SE, SW)
        child_indices = node_data['child_indices']
        indices = [
            child_indices.get('NE', -1),
            child_indices.get('NW', -1),
            child_indices.get('SE', -1),
            child_indices.get('SW', -1)
        ]
        child_indices_list.append(indices)
    
    # Collect function values (some nodes have None)
    values_dict = {}
    for i, node_data in enumerate(nodes_list):
        if node_data['coefficients'] is not None:
            values_dict[f'vals_{i}'] = node_data['coefficients']
    
    # Save as NPZ with compression
    np.savez_compressed(
        filepath,
        format='quadtree_v1',
        num_nodes=np.array([len(nodes_list)]),
        bounds=np.array(bounds_list, dtype=np.float32),
        is_leaf=np.array(is_leaf_list, dtype=bool),
        split_points=np.array(split_points_list, dtype=np.float32),
        child_indices=np.array(child_indices_list, dtype=np.int32),
        **values_dict
    )
    print(f"Saved quadtree to: {filepath}")

def load_quadtree(filepath):
    """Load quadtree from compressed NPZ file."""
    data = np.load(filepath, allow_pickle=False)
    
    if data['format'] != 'quadtree_v1':
        raise ValueError("Invalid quadtree file format")
    
    num_nodes = int(data['num_nodes'][0])
    bounds_list = data['bounds']
    is_leaf_list = data['is_leaf']
    split_points_list = data['split_points']
    child_indices_list = data['child_indices']
    
    # Reconstruct nodes
    nodes_list = []
    for i in range(num_nodes):
        node_data = {
            'bounds': tuple(bounds_list[i]),
            'is_leaf': bool(is_leaf_list[i]),
            'split_point': tuple(split_points_list[i]) if not is_leaf_list[i] else None,
            'coefficients': data[f'vals_{i}'] if f'vals_{i}' in data else None,
            'child_indices': {}
        }
        nodes_list.append(node_data)
    
    # Reconstruct tree using flattened list
    nodes = []
    for node_data in nodes_list:
        node = QuadTreeNode(node_data['bounds'])
        node.is_leaf = node_data['is_leaf']
        node.split_point = node_data['split_point']
        node.coefficients = node_data['coefficients']
        nodes.append(node)
    
    # Link children
    for i, child_indices in enumerate(child_indices_list):
        ne_idx, nw_idx, se_idx, sw_idx = child_indices
        if ne_idx >= 0:
            nodes[i].children['NE'] = nodes[ne_idx]
        if nw_idx >= 0:
            nodes[i].children['NW'] = nodes[nw_idx]
        if se_idx >= 0:
            nodes[i].children['SE'] = nodes[se_idx]
        if sw_idx >= 0:
            nodes[i].children['SW'] = nodes[sw_idx]
    
    # print(f"Loaded quadtree from: {filepath}")
    return nodes[0] if nodes else None

if __name__ == "__main__":
    x_coords, y_coords, global_points, global_values, source_interpolator = load_hdf5_table(SOURCE_TABLE_FILE)
    domain_xmin, domain_xmax = float(x_coords.min()), float(x_coords.max())
    domain_ymin, domain_ymax = float(y_coords.min()), float(y_coords.max())
    train_resolution = len(x_coords)

    print("Starting Adaptive Quadtree Decomposition from Uniform Table...")
    quadtree_root = build_quadtree(
        domain_xmin,
        domain_xmax,
        domain_ymin,
        domain_ymax,
        ERROR_THRESHOLD,
        MAX_DEPTH,
        global_points,
        global_values,
        source_interpolator,
    )

    quadtree_file = f"tables/quadtree-{MAX_DEPTH}-{ERROR_THRESHOLD}-{train_resolution}.npz"
    save_quadtree(quadtree_root, quadtree_file)
    
