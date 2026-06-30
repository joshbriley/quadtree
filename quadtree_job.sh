#!/bin/bash --login
#SBATCH --job-name=quadtree1e-2
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=brileyjo@msu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/quadtree_10_1e-2.out

ml purge
module load Miniforge3/25.11.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quadtree

python -m python.build_quadtree
