#!/bin/bash
#SBATCH --job-name=cpu_job          # Job name
#SBATCH --output=output_%j.log       # Output log file (%j = job ID)
#SBATCH --error=error_%j.log         # Error log file
#SBATCH --time=02:00:00              # Time limit (HH:MM:SS)
#SBATCH --partition=ice-cpu    # Specify a CPU-only partition (if required)
#SBATCH --ntasks=1                   # Number of tasks (1 process)
#SBATCH --cpus-per-task=8            # Request 4 CPU cores
#SBATCH --mem=64G
#SBATCH --mail-user=ychauhan9@gatech.edu  # Replace with your email

# Load necessary modules (modify if needed)
module load anaconda3
source activate my_env  # Replace with your Conda environment

# Run the Python script
python bf_test.py