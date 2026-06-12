import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CloughTocher2DInterpolator


# Function we are trying to emulate
def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

# Compute Training error
def evaluate_quad_error(xmin, xmax, ymin, ymax, global_points):

    # Evaluate training error
    num_of_points = 3 # minimum of 2
    x_corner = np.linspace(xmin, xmax, num_of_points)
    y_corner = np.linspace(ymin, ymax, num_of_points)

    # X_c, Y_c = np.meshgrid(x_grid, y_grid, indexing='ij')
    x_mesh, y_mesh = np.meshgrid(x_corner, y_corner, indexing='ij')
    corner_vals = test_func(x_mesh, y_mesh)
    points_input = np.column_stack((x_mesh.flatten(), y_mesh.flatten()))
    values_input = corner_vals.flatten()

    # interp_func = RegularGridInterpolator((x_corner, y_corner), corner_vals, method='cubic')
    interp_func = CloughTocher2DInterpolator(points_input, values_input) 

    # Predict values for the internal global training points
    x_coords = global_points[:, 0]
    y_coords = global_points[:, 1]

    # Create a boolean mask for points inside the cell
    mask = (x_coords >= xmin) & (x_coords <= xmax) & (y_coords >= ymin) & (y_coords <= ymax)

    # Filter the global points for only points in the current cell
    quad_pts = global_points[mask]
    if quad_pts.shape[0] == 0:
        raise ValueError(
            f"No global training points found in the current cell!\n"
            f"Cell Boundaries:\n"
            f"  X: [{xmin}, {xmax}]\n"
            f"  Y: [{ymin}, {ymax}]\n"
            f"Hint: The max depth might be too deep or the error threshold too small for your global "
            f"TRAIN_RESOLUTION, causing cells to become smaller than the grid spacing."
        )
    interp_vals = interp_func(quad_pts)
    quad_true_vals = test_func(quad_pts[:, 0], quad_pts[:, 1])
    
    # Compute Absolute Error Norm
    err = np.abs(interp_vals - quad_true_vals)
    linfy_norm = np.max(err)
    return linfy_norm 

def build_quadtree(xmin, xmax, ymin, ymax, threshold, max_depth, global_points, global_vals, fig_index, current_depth=0, leaf_boxes=None):
    if leaf_boxes is None:
        leaf_boxes = []

    global quad_error 
    # Calculate error based on the global grid subset
    quad_error = evaluate_quad_error(xmin, xmax, ymin, ymax, global_points)

    # Base Cases
    if quad_error <= threshold or current_depth >= max_depth:
        leaf_boxes.append((xmin, xmax, ymin, ymax, current_depth))
        # print(f"Quad error: {quad_error}\nCurrent depth: {current_depth}")
        return leaf_boxes

    # Recursive Step: Split into 4 children
    x_mid = (xmin + xmax) / 2.0
    y_mid = (ymin + ymax) / 2.0

    children = [
        (x_mid, xmax,  y_mid, ymax),  # Quad I
        (xmin,  x_mid,  y_mid, ymax),  # Quad II
        (xmin,  x_mid,  ymin,  y_mid),  # Quad III
        (x_mid, xmax,  ymin,  y_mid)   # Quad IV
    ]

    for c_xmin, c_xmax, c_ymin, c_ymax in children:
        build_quadtree(c_xmin, c_xmax, c_ymin, c_ymax, threshold, max_depth, 
                       global_points, global_vals, fig_index, current_depth + 1, leaf_boxes)
    return leaf_boxes

## === Execute Algorithm === ##
DOMAIN_XMIN, DOMAIN_XMAX = -2.0, 2.0
DOMAIN_YMIN, DOMAIN_YMAX = -2.0, 2.0
ERROR_THRESHOLD = 1e-2
MAX_DEPTH = 7 

# Define the Global Uniform Training Grid
TRAIN_RESOLUTION = 500 
x_train_global = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, TRAIN_RESOLUTION)
y_train_global = np.linspace(DOMAIN_YMIN, DOMAIN_YMAX, TRAIN_RESOLUTION)
X_train_g, Y_train_g = np.meshgrid(x_train_global, y_train_global, indexing='ij')

global_points = np.vstack([X_train_g.flatten(), Y_train_g.flatten()]).T
global_vals = test_func(global_points[:, 0], global_points[:, 1])
fig_index = 0

print("Starting Adaptive Quadtree Decomposition...")
boxes = build_quadtree(DOMAIN_XMIN, DOMAIN_XMAX, DOMAIN_YMIN, DOMAIN_YMAX, 
                       ERROR_THRESHOLD, MAX_DEPTH, global_points, global_vals, fig_index)

# --- Generate High-Res Background Heatmap ---
x_bg = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, 400)
y_bg = np.linspace(DOMAIN_YMIN, DOMAIN_YMAX, 400)
X_bg, Y_bg = np.meshgrid(x_bg, y_bg, indexing='ij')
Z_bg = test_func(X_bg, Y_bg)

# --- Plotting Layout ---
fig, ax = plt.subplots(figsize=(10, 8))

# Plot underlying function data as a background heatmap
pc = ax.pcolormesh(X_bg, Y_bg, Z_bg, cmap='viridis', shading='auto', alpha=0.75)
fig.colorbar(pc, ax=ax, label='f(x, y) Value')

# Outline each leaf node bounding box
for xmin, xmax, ymin, ymax, depth in boxes:
    x_box = [xmin, xmax, xmax, xmin, xmin]
    y_box = [ymin, ymin, ymax, ymax, ymin]
    
    # Dynamically thin lines for deeper levels so the plot remains clean
    linewidth = max(0.5, 2.5 - 0.5 * depth)
    ax.plot(x_box, y_box, color='red', linewidth=linewidth, alpha=0.8)

ax.set_title(f'Adaptive Quadtree Grid (Global Training Grid Method)\nThreshold={ERROR_THRESHOLD}, Max Depth={MAX_DEPTH}, Leaves={len(boxes)}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(DOMAIN_XMIN, DOMAIN_XMAX)
ax.set_ylim(DOMAIN_YMIN, DOMAIN_YMAX)
ax.grid(False)
plt.show()

# Extract and save the unique node corners 
unique_corners = set()

for xmin, xmax, ymin, ymax, depth in boxes:
    unique_corners.add((xmin, ymin)) # Bottom-Left
    unique_corners.add((xmax, ymin)) # Bottom-Right
    unique_corners.add((xmin, ymax)) # Top-Left
    unique_corners.add((xmax, ymax)) # Top-Right

# Convert the unique coordinates into a structured NumPy array
nodes_array = np.array(list(unique_corners))

# Compute the function evaluation at each unique mesh node corner
node_vals = test_func(nodes_array[:, 0], nodes_array[:, 1])

# Construct the uniform-equivalent Dataframe
df_points = pd.DataFrame({
    'X': nodes_array[:, 0],
    'Y': nodes_array[:, 1],
    'F': node_vals
})

# Save the final table to your directory
csv_dir = f"tables/quadtree_points_CT-{MAX_DEPTH}-{ERROR_THRESHOLD}-{TRAIN_RESOLUTION}.csv"
df_points.to_csv(csv_dir, index=False)

print(f"Saved quadtree vertices table to: {csv_dir}")
