#!/bin/bash
#SBATCH --job-name=Eman_Thesis_NNtraining
#SBATCH --output=NNresults.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=regular

# Load necessary modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/11.7.0

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to the project directory
cd /home3/s4317394/pytorch

# Run the Python script
python3.11 main.py

# Transfer results back (if necessary)
mkdir -p /home3/s4317394/pytorch/results
cp *.pth /home3/s4317394/pytorch/results
cp pytorch_job.out /home3/s4317394/pytorch/results