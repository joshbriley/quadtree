import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

def calculate_norms_from_csv(csv_filename, linf_norms, i):
    # 1. Load the flat table
    df = pd.read_csv(csv_filename)
    
    # 2. Extract the unique X and Y axis coordinate points
    x_pts = np.unique(df['X_Coordinate'].values)
    y_pts = np.unique(df['Y_Coordinate'].values)
    
    res_x = len(x_pts)
    res_y = len(y_pts)
    print(f"Uniform grid resolution: {res_x}x{res_y} ({len(df):,} total rows)")
    
    # 3. Reshape the flat function column back into a 2D matrix
    func_grid_2d = df['Function_Value'].values.reshape(res_x, res_y)
    
    # 4. Reconstruct the Uniform Grid Interpolator
    uniform_interp = RegularGridInterpolator((x_pts, y_pts), func_grid_2d, method='linear')
    
    # 5. Create a Dense Disjoint Test Set
    # I chose 200x200 points slightly offset so we evaluate *between* the table nodes.
    x_test = np.linspace(-1.99, 1.99, 200)
    y_test = np.linspace(-1.99, 1.99, 200)
    X_test, Y_test = np.meshgrid(x_test, y_test, indexing='ij')
    test_points = np.vstack([X_test.flatten(), Y_test.flatten()]).T
    
    # 6. Compute exact true values vs interpolated values at test points
    true_vals = test_func(test_points[:, 0], test_points[:, 1])
    interp_vals = uniform_interp(test_points)
    
    # 7. Calculate Pointwise Absolute Error
    rel_errors = np.abs((interp_vals - true_vals))
        
    # 8. Compute global error norms
    l1_norm = np.mean(rel_errors)       
    l2_norm = np.sqrt(np.mean(rel_errors**2))             
    linf_norm = np.max(rel_errors)                       
    
    # Display Results
    print("================ ERROR NORMS ================")
    print(f"L1 Norm:    {l1_norm:.6f}")
    print(f"L2 Norm:    {l2_norm:.6f}")
    print(f"L_inf Norm: {linf_norm:.6f}")
    print("=============================================\n")

    # norms[:,i] = [l1_norm, l2_norm, linf_norm] 
    linf_norms[i] = linf_norm

    return l1_norm, l2_norm, linf_norm, linf_norms

# --- Execute Script ---
if __name__ == "__main__":
    # Point this to the CSV file you want to evaluate
    resolutions = [16, 32, 64, 128]
    # norms = np.empty((3,4))
    linf_norms = np.empty(4)
    table_sizes = np.empty(4)
    i = 0 
    for res in resolutions:
        target_csv = f'tables/uniform_grid_func_evals/uniform_evaluations-{res}.csv'
        table_sizes[i] = os.path.getsize(target_csv)/1024
        calculate_norms_from_csv(target_csv, linf_norms, i)
        i = i + 1
        print(f"L_inf norms:\n{linf_norms}")
        print(f"\nTable Sizes:\n{table_sizes}")

    # plt.plot(resolutions, norms[0,:], color='green', marker='*', label='L_1 Norm')
    # plt.plot(resolutions, norms[1,:], color='blue', marker='*', label='L_2 Norm')
    # plt.plot(resolutions, norms[2,:], color='red', marker='*', label='L_inf Norm')
    
    # plt.title('Uniform Grid: Resolution vs Error')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig('figs/uniform_grid_vs_resolution.png')
    # plt.show()

    # Got these values manually. 
    table_sizes_quadtree = [5.880859375, 35.373046875, 318.1103515625, 998.3125]
    linf_norms_quadtree = [0.15232588506142, 0.126142845944937, 0.0490495835072506, 0.0490495835072506]
    plt.plot(table_sizes, linf_norms, 'go--', label='Uniform Grid')
    plt.plot(table_sizes_quadtree, linf_norms_quadtree, 'bo--', label='Quadtree')
    plt.title('Comparing Memory Size & Error: Uniform Grid and Quadtree')
    plt.xlabel('Table Size (KBytes)')
    plt.ylabel('L_inf Norm Absolute Error')
    plt.legend()
    plt.savefig('figs/L-inf-norm_vs_size.png')
    plt.show()
