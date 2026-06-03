# Saves a 2D and 3D plot of the test functiona and saves a csv file of evaluations on the test function.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define resolution
resolution = 128 
x_pts = np.linspace(-2, 2, resolution)
y_pts = np.linspace(-2, 2, resolution)

# 2. Create a 2D grid for the function evaluation
X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')

# Evaluate the 2D function
func_grid = np.sin(X*Y) + 1.0/(1.0 + np.exp(-100*(X - Y)))

## ------------------------- ##
# --- Plot 1: 2D Heatmap ---
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1)

# Create the heatmap
c2d = ax1.pcolormesh(X, Y, func_grid, cmap='viridis', shading='auto')
fig.colorbar(c2d, ax=ax1, label='f(x, y) value')
ax1.set_title('2D Heatmap of Test Function')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Plot 2: 3D Surface Plot ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Y, func_grid, cmap='viridis', edgecolor='none', alpha=0.8)
ax2.set_title('3D Surface of Test Function')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('f(x, y)')

# Adjust camera angle
ax2.view_init(elev=30, azim=-45)
plt.tight_layout()
filename = f"figs/2d&3d_uniform-{resolution}.png"
plt.savefig(filename)
print(f"---\nSaved plot to '{filename}'\n---")
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
output_filename = f'tables/uniform_evaluations-{resolution}.csv'
df_grid_evaluations.to_csv(output_filename, index=False)

print(f"Saved all {len(df_grid_evaluations):,} grid points to '{output_filename}'\n---")
