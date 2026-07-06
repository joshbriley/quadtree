#!/bin/bash --login
#SBATCH --job-name=quadtree-7-1e-5-log
#SBATCH --time=9:59:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=brileyjo@msu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/quadtree_7_1e-5_log.out

ml purge
module load Miniforge3/25.11.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quadtree

python -m python.build_quadtree
