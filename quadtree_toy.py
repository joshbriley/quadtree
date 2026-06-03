import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

def evaluate_quad_error(xmin, xmax, ymin, ymax):
    """
    Builds a local interpolator and computes the L-infinity norm 
    of the relative error within a specific bounding box.
    """
    # 1. Local "Training" Grid (Discretized Points)
    # Using 10x10 points locally inside each quad
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
    
    # 4. Compute Error
    err = np.abs((interp_vals - true_vals))
        
    # Return the L-infinity norm (max absolute relative error)
    return np.max(err)

def build_quadtree(xmin, xmax, ymin, ymax, threshold, max_depth, current_depth=0, leaf_boxes=None):
    """
    Recursively splits the domain into quads until the error threshold
    is satisfied or max_depth is reached. Tracks and returns leaf nodes.
    """
    # Initialize the tracking list
    if leaf_boxes is None:
        leaf_boxes = []

    # Calculate the error in the current bounding box using the above function
    quad_error = evaluate_quad_error(xmin, xmax, ymin, ymax)

    # Base Case 1: Error is acceptable
    if quad_error <= threshold:
        leaf_boxes.append((xmin, xmax, ymin, ymax, current_depth))
        return leaf_boxes

    # Base Case 2: Max depth reached
    if current_depth >= max_depth:
        leaf_boxes.append((xmin, xmax, ymin, ymax, current_depth))
        return leaf_boxes

    # Recursive Step: Split into 4 smaller quadrants
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

    return leaf_boxes

## === Execute Algorithm === ##
# 1. Run Quadtree Algorithm
DOMAIN_XMIN, DOMAIN_XMAX = -2.0, 2.0
DOMAIN_YMIN, DOMAIN_YMAX = -2.0, 2.0
ERROR_THRESHOLD = 0.1
MAX_DEPTH = 4 

print("Starting Adaptive Quadtree Decomposition...")
boxes = build_quadtree(DOMAIN_XMIN, DOMAIN_XMAX, DOMAIN_YMIN, DOMAIN_YMAX, ERROR_THRESHOLD, MAX_DEPTH)

# 2. Generate High-Res Background Heatmap Data
# High resolution grid just for a comparison background image
x_bg = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, 400)
y_bg = np.linspace(DOMAIN_YMIN, DOMAIN_YMAX, 400)
X_bg, Y_bg = np.meshgrid(x_bg, y_bg, indexing='ij')
Z_bg = test_func(X_bg, Y_bg)

# 3. Plotting
plt.figure(figsize=(10, 8))

# Plot underlying function data as a background heatmap
plt.pcolormesh(X_bg, Y_bg, Z_bg, cmap='viridis', shading='auto', alpha=0.75)
plt.colorbar(label='f(x, y) Value')

# Iterate through every saved box and outline it on the plot
for xmin, xmax, ymin, ymax, depth in boxes:
    # Build coordinates for a closed box path: (xmin, ymin) -> (xmax, ymin) -> (xmax, ymax) -> (xmin, ymax) -> back to start
    x_box = [xmin, xmax, xmax, xmin, xmin]
    y_box = [ymin, ymin, ymax, ymax, ymin]

    # Styling
    linewidth = max(0.5, 2.5 - 0.5 * depth)

    # Draw the boundary line
    plt.plot(x_box, y_box, color='red', linewidth=linewidth, alpha=0.8)

plt.title(f'Adaptive Quadtree Grid\nThreshold={ERROR_THRESHOLD}, Max Depth={MAX_DEPTH}, Leaves={len(boxes)}')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(DOMAIN_XMIN, DOMAIN_XMAX)
plt.ylim(DOMAIN_YMIN, DOMAIN_YMAX)
plt.grid(False) 
plt.savefig('quadtree_boxes.png')
plt.show()

# Save boxes to a csv
df_boxes = pd.DataFrame(boxes, columns=['X_min', 'X_max', 'Y_min', 'Y_max', 'Depth'])
output_filename = 'quadtree_boxes.csv'
df_boxes.to_csv(output_filename, index=False)

