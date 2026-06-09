import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator

def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

def evaluate_quad_error(xmin, xmax, ymin, ymax, global_points, global_vals):
    """
    Evaluates error using ONLY the pre-existing global uniform training grid 
    points that fall inside this specific box boundary (matching paper).
    """
    # Mask out only the global points that sit inside this quad
    mask = (global_points[:, 0] >= xmin) & (global_points[:, 0] <= xmax) & \
           (global_points[:, 1] >= ymin) & (global_points[:, 1] <= ymax)
    
    quad_pts = global_points[mask]
    quad_true_vals = global_vals[mask]
    
    # If a box is so small that no global grid points fall inside it, stop splitting.
    if len(quad_pts) == 0:
        return 0.0

    # Local Interpolator using ONLY the 4 corners of the current box
    x_corners = np.array([xmin, xmax])
    y_corners = np.array([ymin, ymax])
    X_c, Y_c = np.meshgrid(x_corners, y_corners, indexing='ij')
    corner_vals = test_func(X_c, Y_c)

    interp_func = RegularGridInterpolator((x_corners, y_corners), corner_vals, method='linear')
    
    # Predict values for the internal global training points
    interp_vals = interp_func(quad_pts)
    
    # Compute Absolute Error Norm
    err = np.abs(interp_vals - quad_true_vals)
    linfy_norm = np.max(err)
    return linfy_norm 

def build_quadtree(xmin, xmax, ymin, ymax, threshold, max_depth, global_points, global_vals, current_depth=0, leaf_boxes=None):
    if leaf_boxes is None:
        leaf_boxes = []

    # Calculate error based on the global grid subset
    quad_error = evaluate_quad_error(xmin, xmax, ymin, ymax, global_points, global_vals)

    # Base Cases
    if quad_error <= threshold or current_depth >= max_depth:
        leaf_boxes.append((xmin, xmax, ymin, ymax, current_depth))
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
                       global_points, global_vals, current_depth + 1, leaf_boxes)

    return leaf_boxes

## === Execute Algorithm === ##
DOMAIN_XMIN, DOMAIN_XMAX = -2.0, 2.0
DOMAIN_YMIN, DOMAIN_YMAX = -2.0, 2.0
ERROR_THRESHOLD = 0.001
MAX_DEPTH = 15 

# Define the Global Uniform Training Grid (128x128 baseline)
TRAIN_RESOLUTION = 256 
x_train_global = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, TRAIN_RESOLUTION)
y_train_global = np.linspace(DOMAIN_YMIN, DOMAIN_YMAX, TRAIN_RESOLUTION)
X_train_g, Y_train_g = np.meshgrid(x_train_global, y_train_global, indexing='ij')

global_points = np.vstack([X_train_g.flatten(), Y_train_g.flatten()]).T
global_vals = test_func(global_points[:, 0], global_points[:, 1])

print("Starting Adaptive Quadtree Decomposition...")
boxes = build_quadtree(DOMAIN_XMIN, DOMAIN_XMAX, DOMAIN_YMIN, DOMAIN_YMAX, 
                       ERROR_THRESHOLD, MAX_DEPTH, global_points, global_vals)

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

# Safely save the plot, building the directories if missing
os.makedirs("figs/quadtree_grid", exist_ok=True)
fig_filename = f"figs/quadtree_grid/quadtree_boxes_{MAX_DEPTH}_{ERROR_THRESHOLD}.png"
plt.savefig(fig_filename, bbox_inches='tight')
plt.show()

# --- Save Structural Tables ---
# 1. Keep saving your bounding box metadata layout file
df_boxes = pd.DataFrame(boxes, columns=['X_min', 'X_max', 'Y_min', 'Y_max', 'Depth'])
boxes_csv = f"tables/quadtree_grid/quadtree_boxes_{MAX_DEPTH}_{ERROR_THRESHOLD}.csv"
df_boxes.to_csv(boxes_csv, index=False)
print(f"Saved structural index layout containing {len(boxes)} leaves.")


# 2. Extract and save the unique node corners (Matches the Uniform Grid table structure)
unique_corners = set()

for xmin, xmax, ymin, ymax, depth in boxes:
    # A 2D cell has exactly 4 corner vertices
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
    'X_Coordinate': nodes_array[:, 0],
    'Y_Coordinate': nodes_array[:, 1],
    'Function_Value': node_vals
})

# Save the final table to your directory
points_csv = f"tables/quadtree_grid/quadtree_points_{MAX_DEPTH}_{ERROR_THRESHOLD}.csv"
df_points.to_csv(points_csv, index=False)

print(f"Saved uniform-equivalent quadtree points table to: {points_csv}")
print(f"Total unique emulation data points stored in memory: {len(df_points):,}")
