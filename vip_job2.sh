#!/bin/bash
#SBATCH --job-name=diffusion_job          # Job name
#SBATCH --output=diffusion_output2.log     # Output log file
#SBATCH --error=diffusion_error.log       # Error log file
#SBATCH --partition=ice-gpu                   # Partition (change based on your cluster)
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --mem-per-cpu=128G               # Increase memory (try 32G, 64G, or higher)
#SBATCH --cpus-per-task=8                 # Request 4 CPU cores
#SBATCH --time=01:00:00                   # Time limit (1 hour)
#SBATCH --mail-type=END,FAIL              # Email on job end or fail
#SBATCH --mail-user=ychauhan9@gatech.edu  # Replace with your email

# Load necessary modules (modify if needed)
module load anaconda3
source activate my_env  # Replace with your Conda environment

# Run the Python script
accelerate launch bf_test.py --num_processes=1
free -m