# Saves a 2D and 3D plot of the test function and saves an HDF5 file of evaluations on the test function.

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Define resolution
resolution = 256
x_pts = np.linspace(-2, 2, resolution)
y_pts = np.linspace(-2, 2, resolution)

# Create a 2D grid for the function evaluation
X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')

# # Evaluate the 2D function
# func_grid = np.sin(X*Y) + 1.0/(1.0 + np.exp(-100*(X - Y)))

# Use a 2-dimensional tanh as a smoother function 
func_grid = np.tanh(X*Y)

# Save to an HDF5 file matching the expected structure (den, temp, Table_Values/f)
output_filename = f'tables/uniform_grid_func_evals/uniform_evaluations-{resolution}.hdf5'
with h5py.File(output_filename, 'w') as f:
    f.create_dataset('den', data=X)
    f.create_dataset('temp', data=Y)
    grp = f.create_group('Table_Values')
    grp.create_dataset('f', data=func_grid)

print(f"Saved {resolution}x{resolution} = {resolution**2:,} grid points to '{output_filename}'\n---")

# ## ----------- PLOTTING ------------- ## #
# # --- Plot 1: 2D Heatmap ---
# fig = plt.figure(figsize=(14, 6))
# ax1 = fig.add_subplot(1, 2, 1)

# # Create the heatmap
# c2d = ax1.pcolormesh(X, Y, func_grid, cmap='viridis', shading='auto')
# fig.colorbar(c2d, ax=ax1, label='f(x, y) value')
# ax1.set_title('2D Heatmap of Test Function')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.grid(True, linestyle='--', alpha=0.5)

# # --- Plot 2: 3D Surface Plot ---
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax2.plot_surface(X, Y, func_grid, cmap='viridis', edgecolor='none', alpha=0.8)
# ax2.set_title('3D Surface of Test Function')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('f(x, y)')

# # Adjust camera angle
# ax2.view_init(elev=30, azim=-45)
# plt.tight_layout()
# filename = f"figs/2d&3d_uniform-{resolution}.png"
# plt.savefig(filename)
# print(f"---\nSaved plot to '{filename}'\n---")
# ## ------------------------- ##

