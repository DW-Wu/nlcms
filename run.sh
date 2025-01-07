#!/bin/bash
#SBATCH -o continuation_%j_%N.out
#SBATCH -p cpu
#SBATCH -J fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2-00:00:00

# Load python environment
module load anaconda; source ~/.bashrc; source bin/activate

# Ring to tactoid
echo ring lam+
python3 nlc_cont.py cont0.json cont1.json -i ring48.npy -N=47 --num-steps=10 --refine-steps=3 --maxiter=12000 --eta=2e-4 --tol=1e-6 -sf -o out/rn2tc
# Split-core to tactoid
echo score lam+
python3 nlc_cont.py cont0.json cont1.json -i score48.npy -N=47 --num-steps=10 --refine-steps=3 --maxiter=12000 --eta=2e-4 --tol=1e-6 -sf -o out/sc2tc
# Tactoid to whatever
echo tac lam-
python3 nlc_cont.py cont1.json cont0.json -i tactoid48.npy -N=47 --num-steps=10 --refine-steps=3 --maxiter=12000 --eta=2e-4 --tol=1e-6 -sf -o out/tc2w
