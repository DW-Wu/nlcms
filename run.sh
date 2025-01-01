#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2-00:00:00

module load conda

# Ring to tactoid
python3 nlc_cont.py cont0.json cont1.json -i score48.npy -N=47 --num-steps=10 --refine-steps=3 --maxiter=6000 --eta=2e-4 --tol=1e-6 -sf -o out/sc2tc
# Tactoid to whatever
python3 nlc_cont.py cont1.json cont0.json -i tactoid48.npy -N=47 --num-steps=10 --refine-steps=3 --maxiter=6000 --eta=2e-4 --tol=1e-6 -sf -o out/tc2w
