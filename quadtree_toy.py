import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def test_func(x,y):
    result = np.sin(x*y) + 1.0/(1.0 + np.exp(-100*(x - y)))
    return result

def test_func(x, y):
    # Math fix to avoid overflow/underflow warnings
    exponent = -100 * (x - y)
    clipped_exponent = np.clip(exponent, -700, 700)
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(clipped_exponent))

def evaluate_quad_error(xmin, xmax, ymin, ymax):
    """
    Builds a local interpolator and computes the L-infinity norm 
    of the relative error within a specific bounding box.
    """
    # 1. Local "Training" Grid (Discretized Points)
    # Using 10x10 points locally for speed/precision inside each quad
    x_train = np.linspace(xmin, xmax, 10)
    y_train = np.linspace(ymin, ymax, 10)
    X_train, Y_train = np.meshgrid(x_train, y_train, indexing='ij')
    func_grid = test_func(X_train, Y_train)
    
    # 2. Local Interpolator
    interp_func = RegularGridInterpolator((x_train, y_train), func_grid, method='linear')
    
    # 3. Dense local test grid to find the max error between training nodes
    x_test = np.linspace(xmin, xmax, 25)
    y_test = np.linspace(ymin, ymax, 25)
    X_test, Y_test = np.meshgrid(x_test, y_test, indexing='ij')
    
    query_points = np.vstack([X_test.flatten(), Y_test.flatten()]).T
    true_vals = test_func(query_points[:, 0], query_points[:, 1])
    interp_vals = interp_func(query_points)
    
    # 4. Compute Relative Error safely
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.abs((interp_vals - true_vals) / true_vals)
        rel_err = np.nan_to_num(rel_err, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Return the L-infinity norm (max absolute relative error)
    return np.max(rel_err)

def build_quadtree(xmin, xmax, ymin, ymax, threshold, max_depth, current_depth=0, leaf_boxes=None):
    """
    Recursively splits the domain into quads until the error threshold
    is satisfied or max_depth is reached. Tracks and returns leaf nodes.
    """
    # Initialize the tracking list on the very first call
    if leaf_boxes is None:
        leaf_boxes = []

    # Calculate the error in the current bounding box
    error = evaluate_quad_error(xmin, xmax, ymin, ymax)

    indent = "  " * current_depth
    print(f"{indent}Depth {current_depth} | Region: X[{xmin:.2f}, {xmax:.2f}], Y[{ymin:.2f}, {ymax:.2f}] -> Max Error: {error:.4f}")

    # Base Case 1: Error is acceptable
    if error <= threshold:
        print(f"{indent}>> Stop: Error threshold met.")
        leaf_boxes.append((xmin, xmax, ymin, ymax, current_depth))
        return leaf_boxes

    # Base Case 2: Max depth reached
    if current_depth >= max_depth:
        print(f"{indent}>> Stop: Reached maximum depth limit.")
        leaf_boxes.append((xmin, xmax, ymin, ymax, current_depth))
        return leaf_boxes

    # Recursive Step: Split into 4 smaller quadrants
    print(f"{indent}>> Error exceeds threshold! Splitting region...")
    x_mid = (xmin + xmax) / 2.0
    y_mid = (ymin + ymax) / 2.0

    # Define boundaries for the 4 children
    children = [
        ("Quad I (Top-Right)",    x_mid, xmax,  y_mid, ymax),
        ("Quad II (Top-Left)",    xmin,  x_mid,  y_mid, ymax),
        ("Quad III (Bottom-Left)", xmin,  x_mid,  ymin,  y_mid),
        ("Quad IV (Bottom-Right)", x_mid, xmax,  ymin,  y_mid)
    ]

    # Recursively process each child, passing the same leaf_boxes list down
    for name, c_xmin, c_xmax, c_ymin, c_ymax in children:
        build_quadtree(c_xmin, c_xmax, c_ymin, c_ymax, threshold, max_depth, current_depth + 1, leaf_boxes)

    # CRITICAL: This passes the finalized coordinates back up to the top level execution
    return leaf_boxes

# --- Execute Algorithm ---
# Initial global bounding box domain
DOMAIN_XMIN, DOMAIN_XMAX = -2.0, 2.0
DOMAIN_YMIN, DOMAIN_YMAX = -2.0, 2.0

# --- 1. Run Quadtree Algorithm ---
DOMAIN_MIN, DOMAIN_MAX = -2.0, 2.0
ERROR_THRESHOLD = 0.01
MAX_DEPTH = 6 

print("Starting Adaptive Quadtree Decomposition...")

# Make sure the variable 'boxes' is explicitly defined here:
boxes = build_quadtree(DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX, ERROR_THRESHOLD, MAX_DEPTH)

# --- 2. Generate Background Heatmap Data ---
# High resolution grid just for a beautiful background image
x_bg = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, 400)
y_bg = np.linspace(DOMAIN_YMIN, DOMAIN_YMAX, 400)
X_bg, Y_bg = np.meshgrid(x_bg, y_bg, indexing='ij')
Z_bg = test_func(X_bg, Y_bg)

# --- 3. Plotting ---
plt.figure(figsize=(10, 8))

# Plot underlying function data as a background heatmap
plt.pcolormesh(X_bg, Y_bg, Z_bg, cmap='viridis', shading='auto', alpha=0.75)
plt.colorbar(label='f(x, y) Value')

# Iterate through every saved box and outline it on the plot
for xmin, xmax, ymin, ymax, depth in boxes:
    # Build coordinates for a closed box path: (xmin, ymin) -> (xmax, ymin) -> (xmax, ymax) -> (xmin, ymax) -> back to start
    x_box = [xmin, xmax, xmax, xmin, xmin]
    y_box = [ymin, ymin, ymax, ymax, ymin]

    # Optional styling: make deeper, smaller quads have thinner lines
    linewidth = max(0.5, 2.5 - 0.5 * depth)

    # Draw the boundary line
    plt.plot(x_box, y_box, color='red', linewidth=linewidth, alpha=0.8)

plt.title(f'Adaptive Quadtree Grid\nThreshold={ERROR_THRESHOLD}, Max Depth={MAX_DEPTH}, Leaves={len(boxes)}')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(DOMAIN_MIN, DOMAIN_MAX)
plt.ylim(DOMAIN_MIN, DOMAIN_MAX)
plt.grid(False) # Turn off standard grid so it doesn't clash with quadtree lines
plt.savefig('quadtree_boxes.png')
plt.show()

# 1. Define the grid axes
x_pts = np.linspace(-2, 2, 100)
y_pts = np.linspace(-2, 2, 100)

# 2. Create a 2D grid for the function evaluation
X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')

# Evaluate the 2D function
func_grid = np.sin(X*Y) + 1.0/(1.0 + np.exp(-100*(X - Y)))

# # 4. Define arbitrary points where you want to interpolate
# # Format: [[x1, y1], [x2, y2], ...]
# query_points = np.array([
#     [1.3, -1.5],
#     [0.0, 0.0],
#     [-0.1, 0.8]
# ])

# # 5. Compute the interpolated values
# interpolated_values = interp_func(query_points)

# # Output results
# for pt, val in zip(query_points, interpolated_values):
#     print(f"Interpolated value at (x={pt[0]:.2f}, y={pt[1]:.2f}): {val:.4f}")

# # Compare to true values
# rel_err = abs((interpolated_values - test_func(query_points[:, 0], query_points[:, 1])) / test_func(query_points[:, 0], query_points[:, 1]))
# print(f"Relative errors: {rel_err}")

fig = plt.figure(figsize=(14, 6))

## ------------------------- ##
# --- Plot 1: 2D Heatmap ---
ax1 = fig.add_subplot(1, 2, 1)
# Create the heatmap
c2d = ax1.pcolormesh(X, Y, func_grid, cmap='viridis', shading='auto')
fig.colorbar(c2d, ax=ax1, label='f(x, y) value')

# # Scatter the query points on top
# ax1.scatter(query_points[:, 0], query_points[:, 1], color='red', edgecolor='black',
#             s=100, zorder=5, label='Query Points')

ax1.set_title('2D Heatmap of Test Function')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
# ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Plot 2: 3D Surface Plot ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# Create the 3D surface
surf = ax2.plot_surface(X, Y, func_grid, cmap='viridis', edgecolor='none', alpha=0.8)

# # Calculate true values for query points to plot them in 3D space
# true_vals = test_func(query_points[:, 0], query_points[:, 1])
# ax2.scatter(query_points[:, 0], query_points[:, 1], true_vals, 
#             color='red', edgecolor='black', s=100, label='Query Points', zorder=10)

ax2.set_title('3D Surface of Test Function')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('f(x, y)')

# Adjust camera angle for a great view of the sharp drop-off
ax2.view_init(elev=30, azim=-45)

plt.tight_layout()
plt.savefig('2d&3d_plot-true_sol.png')
## ------------------------- ##

# 7. Flatten the 2D grid matrices into 1D columns for the table
x_flat = X.flatten()
y_flat = Y.flatten()
func_flat = func_grid.flatten()

# 8. Create a structured Pandas DataFrame for the entire grid landscape
df_grid_evaluations = pd.DataFrame({
    'X_Coordinate': x_flat,
    'Y_Coordinate': y_flat,
    'Function_Value': func_flat
})

# 9. Save to a CSV file
output_filename = 'grid_evaluations.csv'
df_grid_evaluations.to_csv(output_filename, index=False)

print(f"\nSuccessfully saved all {len(df_grid_evaluations):,} grid points to '{output_filename}'")
