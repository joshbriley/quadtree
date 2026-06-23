# Overview

This branch is the same as the `hdf5` branch, but with with the intention of verifying the method is working properly. I removed all log10 transformations since the function (`f = tanh(x*y)`) only spans ~[-1, 1] and does not have multi-decade behavior.

- I built a uniform table of evaluations of the function `f` on a 128x128 and 256x256 grid, and saved it to an HDF5 file.
- I built a quadtree from the 128x128 table, and saved it to a `.npz` file.
- I wrote a script to evaluate the quadtree at 100,000 random points, and compare the results to the 256x256 table. The error is calculated as the absolute difference between the quadtree evaluation and the reference evaluation.

## Tree parameters
Max depth: 6
Error threshold: 1e-4
Resolution of training table: 128x128
Quadtree file: `tables/quadtree-6-0.0001-128.npz`
Quadtree size: 968.79 kB

## Results
+-- Quadtree Relative Error --+
| L1   : 1.525e-04            |
| L2   : 2.116e-04            |
| Linf : 1.227e-03            |
| Size : 968.79 kB            |
+-----------------------------+

Resolution of reference table: 256x256
Number of points sampled: 100,000

These results are similar to what is seen on the `main` branch when the true function is used to evaluate the error, which suggests that the quadtree is working properly.

