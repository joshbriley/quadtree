#!/bin/bash --login
#SBATCH --job-name=quadtree-10-1e-2-new
#SBATCH --time=23:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=brileyjo@msu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/quadtree_10_1e-2_new.out

ml purge
module load Miniforge3/25.11.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quadtree

python -m python.build_quadtree
