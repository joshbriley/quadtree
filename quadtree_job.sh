#!/bin/bash --login
#SBATCH --job-name=quadtree
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=brileyjo@msu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/quadtree_%j.out
#SBATCH --error=logs/quadtree_%j.err

ml purge
module load Miniforge3/25.11.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quadtree

python build_quadtree.py
