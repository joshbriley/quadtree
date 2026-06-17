import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import RegularGridInterpolator

"""
Build an adaptive quadtree from a uniform data table, without assuming access
to the underlying analytical function that produced the table.
"""

SOURCE_TABLE_FILE = "tables/uniform_grid_func_evals/uniform_evaluations-128.csv"
ERROR_THRESHOLD = 1e-1
MAX_DEPTH = 7

_DEFAULT_SOURCE_INTERPOLATOR = None


def load_uniform_table(csv_path):
    df = pd.read_csv(csv_path)
    if not {"X", "Y", "F"}.issubset(df.columns):
        raise ValueError("Uniform table must contain X, Y, and F columns.")

    x_coords = np.sort(df["X"].unique())
    y_coords = np.sort(df["Y"].unique())
    pivot = df.pivot(index="X", columns="Y", values="F").sort_index().sort_index(axis=1)

    if pivot.shape != (len(x_coords), len(y_coords)):
        raise ValueError("Uniform table is missing grid points or is not rectangular.")

    grid_values = pivot.to_numpy(dtype=float)
    global_points = df[["X", "Y"]].to_numpy(dtype=float)
    global_values = df["F"].to_numpy(dtype=float)

    interp_method = "cubic" if len(x_coords) >= 4 and len(y_coords) >= 4 else "linear"
    source_interpolator = RegularGridInterpolator(
        (x_coords, y_coords),
        grid_values,
        method=interp_method,
        bounds_error=False,
        fill_value=None,
    )

    global _DEFAULT_SOURCE_INTERPOLATOR
    _DEFAULT_SOURCE_INTERPOLATOR = source_interpolator

    return x_coords, y_coords, global_points, global_values, source_interpolator


def evaluate_quad_error(xmin, xmax, ymin, ymax, global_points, global_values, source_interpolator):
    num_of_points = 4
    x_corner = np.linspace(xmin, xmax, num_of_points)
    y_corner = np.linspace(ymin, ymax, num_of_points)

    x_mesh, y_mesh = np.meshgrid(x_corner, y_corner, indexing="ij")
    corner_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    corner_vals = source_interpolator(corner_points).reshape(num_of_points, num_of_points)

    interp_func = RegularGridInterpolator((x_corner, y_corner), corner_vals, method="cubic")

    x_coords = global_points[:, 0]
    y_coords = global_points[:, 1]
    mask = (x_coords >= xmin) & (x_coords <= xmax) & (y_coords >= ymin) & (y_coords <= ymax)

    testing_pts = global_points[mask]
    testing_vals = global_values[mask]
    if testing_pts.shape[0] == 0:
        raise ValueError("No points found in cell for error evaluation.")

    interp_vals = interp_func(testing_pts)
    err = np.abs(interp_vals - testing_vals)
    linfy_norm = np.max(err)
    return linfy_norm, corner_vals

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
            return evaluate_polynomial(self.coefficients, x, y, self.bounds)
        
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
    
    @staticmethod
    def _unflatten_from_list(nodes_list):
        """Reconstruct tree from flattened list."""
        if not nodes_list:
            return None
        
        nodes = []
        for node_data in nodes_list:
            node = QuadTreeNode(node_data['bounds'])
            node.is_leaf = node_data['is_leaf']
            node.split_point = node_data['split_point']
            node.coefficients = node_data['coefficients']
            node._child_indices = node_data.get('child_indices', {})
            nodes.append(node)
        
        for i, node_data in enumerate(nodes_list):
            for key, child_idx in node_data['child_indices'].items():
                if child_idx is not None:
                    nodes[i].children[key] = nodes[child_idx]
        
        return nodes[0] if nodes else None

def evaluate_polynomial(vals, x, y, bounds):
    """Evaluate cubic spline polynomial at (x, y) within cell bounds."""
    xmin, xmax, ymin, ymax = bounds
    
    # Reconstruct the interpolator with original function values at grid corners
    x_corner = np.linspace(xmin, xmax, 4)
    y_corner = np.linspace(ymin, ymax, 4)
    
    # vals contains the function values at the 4x4 grid points
    interp = RegularGridInterpolator((x_corner, y_corner), vals.reshape(4, 4), method='cubic')
    return float(interp([[x, y]])[0])

# Quadtree Builder -- Must be able to export the quadtree structure and the interpolation polynomials at each leaf node for in-situ surrogate evaluation.
def build_quadtree(
    xmin,
    xmax,
    ymin,
    ymax,
    error_threshold,
    max_depth,
    global_points,
    global_vals,
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
            global_vals,
            source_interpolator,
        )
    except ValueError:
        # No table samples fell inside this cell. Warn and fall back to the
        # source interpolator at the cell corners so the leaf remains usable.
        warnings.warn(
            (
                "No points found in cell "
                f"({xmin}, {xmax}, {ymin}, {ymax}); using corner values "
                "from the source interpolator."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        x_corner = np.linspace(xmin, xmax, 4)
        y_corner = np.linspace(ymin, ymax, 4)
        x_mesh, y_mesh = np.meshgrid(x_corner, y_corner, indexing="ij")
        corner_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
        node.is_leaf = True
        node.coefficients = source_interpolator(corner_points).reshape(4, 4)
        return node
    
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
            global_vals,
            source_interpolator,
            depth + 1,
        )
    
    return node

def save_quadtree(quadtree_root, filepath):
    """Save quadtree to compressed NPZ file using flattened structure."""
    nodes_list = []
    quadtree_root._flatten_to_list(nodes_list, {})
    
    # Store metadata and node data separately for efficiency
    metadata = {
        'format': 'quadtree_v1',
        'num_nodes': len(nodes_list)
    }
    
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
    x_coords, y_coords, global_points, global_vals, source_interpolator = load_uniform_table(SOURCE_TABLE_FILE)
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
        global_vals,
        source_interpolator,
    )

    # boxes = quadtree_root.get_leaf_cells()

    # x_bg = np.linspace(domain_xmin, domain_xmax, 400)
    # y_bg = np.linspace(domain_ymin, domain_ymax, 400)
    # X_bg, Y_bg = np.meshgrid(x_bg, y_bg, indexing="ij")

    # test_func = lambda x, y: np.tanh(x * y)
    # Z_bg = test_func(X_bg, Y_bg)

    # fig, ax = plt.subplots(figsize=(10, 8))
    # pc = ax.pcolormesh(X_bg, Y_bg, Z_bg, cmap="viridis", shading="auto", alpha=0.75)
    # fig.colorbar(pc, ax=ax, label="table value")

    # for xmin, xmax, ymin, ymax, depth in boxes:
    #     x_box = [xmin, xmax, xmax, xmin, xmin]
    #     y_box = [ymin, ymin, ymax, ymax, ymin]
    #     linewidth = max(0.5, 2.5 - 0.5 * depth)
    #     ax.plot(x_box, y_box, color="red", linewidth=linewidth, alpha=0.8)

    # ax.set_title(
    #     f"Adaptive Quadtree from Uniform Table\n"
    #     f"Threshold={ERROR_THRESHOLD}, Max Depth={MAX_DEPTH}, Leaves={len(boxes)}"
    # )
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_xlim(domain_xmin, domain_xmax)
    # ax.set_ylim(domain_ymin, domain_ymax)
    # ax.grid(False)
    # plt.savefig("figs/quadtree_grid.png", dpi=300)

    quadtree_file = f"tables/quadtree-{MAX_DEPTH}-{ERROR_THRESHOLD}-{train_resolution}.npz"
    save_quadtree(quadtree_root, quadtree_file)
    
