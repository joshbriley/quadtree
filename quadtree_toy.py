import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def test_func(x,y):
    result = np.sin(x*y) + 1.0/(1.0 + np.exp(-100*(x - y)))
    return result

# 1. Define the grid axes
x_pts = np.linspace(-5, 5, 100)
y_pts = np.linspace(-5, 5, 100)

# 2. Create a 2D grid for the function evaluation
X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')

# Evaluate the 2D function
# Note: Using X and Y (2D grids) instead of x and y (1D arrays)
func_grid = np.sin(X*Y) + 1.0/(1.0 + np.exp(-100*(X - Y)))

# 3. Set up the linear interpolator
interp_func = RegularGridInterpolator((x_pts, y_pts), func_grid, method='linear')

# 4. Define arbitrary points where you want to interpolate
# Format: [[x1, y1], [x2, y2], ...]
query_points = np.array([
    [2.3, -1.5],
    [0.0, 0.0],
    [-4.1, 3.8]
])

# 5. Compute the interpolated values
interpolated_values = interp_func(query_points)

# Output results
for pt, val in zip(query_points, interpolated_values):
    print(f"Interpolated value at (x={pt[0]:.2f}, y={pt[1]:.2f}): {val:.4f}")

# Compare to true values
# query_points[:, 0] grabs all rows of the first column (X)
# query_points[:, 1] grabs all rows of the second column (Y)

rel_err = abs((interpolated_values - test_func(query_points[:, 0], query_points[:, 1])) / test_func(query_points[:, 0], query_points[:, 1]))
print(abs((interpolated_values[0] - test_func(2.3,-1.5))/test_func(2.3,-1.5)))
print(rel_err)

# --- START OF PLOTING --- #
# Initialize a 3D plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the data points (Scatter) and connect them (Line)
# # We map the color of the points to the Pressure (P) using a colormap ('viridis')
# sc = ax.scatter(x, y, func, c=func, cmap='viridis', s=50, edgecolors='black', label='State Points')
# ax.plot(x, y, func, color='gray', linestyle='--', alpha=0.7)

# # Add a color bar to show the pressure scale
# cbar = fig.colorbar(sc, ax=ax, pad=0.1)
# cbar.set_label('y', rotation=270, labelpad=15)

# # Label the axes and add a title
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x,y)')
# ax.set_title('A (hopefully) smooth and rapid function', fontsize=14, fontweight='bold')

# # Adjust the camera viewing angle for better visibility
# ax.view_init(elev=20, azim=-45)

# # Show the plot
# plt.tight_layout()
# plt.show()
# # --- END OF PLOTING --- #
