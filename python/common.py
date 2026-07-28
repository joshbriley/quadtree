import numpy as np
import h5py

def symlog(x, linthresh=1.0, base=10.0):
    x = np.asarray(x)
    return np.sign(x) * np.log1p(np.abs(x) / linthresh) / np.log(base)

def inverse_symlog(y, linthresh=1.0, base=10.0):
    y = np.asarray(y)
    return np.sign(y) * linthresh * np.expm1(np.abs(y) * np.log(base))

def load_hdf5_table(file_path):
     
    with h5py.File(file_path, "r") as f:
        X_raw = f["den"][:]
        Y_raw = f["temp"][:]
        F = f["Table_Values"]["f"][:]

    # Build tree in log10-space to better resolve multi-decade behavior.
    X = np.log10(X_raw)
    Y = np.log10(Y_raw)
    # F = symlog(F_raw)

    # den varies along columns, temp varies along rows
    x_coords = X[0, :]
    y_coords = Y[:, 0]
    global_values = F.ravel()
    global_points = np.column_stack((X.ravel(), Y.ravel()))

    return x_coords, y_coords, global_points, global_values