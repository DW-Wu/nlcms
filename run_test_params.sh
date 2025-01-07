#!/bin/bash
#SBATCH -o params_%j_%N.out
#SBATCH -p cpu
#SBATCH -J fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 6-00:00:00

# Load python environment
module load anaconda; source ~/.bashrc; source bin/activate

python3 test_params.py -s