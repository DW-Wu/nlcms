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

# Radial to ring
echo radial A-
python3 nlc_cont.py cont0.json cont1.json -i radial64.npy -N=63 \
    --num-steps=10 --refine-steps=1 \
    --algorithm=newton --maxiter=50 --eta=0.5 --tol=1e-5 \
    -f -o out/rd2rn
# Ring to radial
echo ring A+
python3 nlc_cont.py cont1.json cont0.json -i ring64.npy -N=63 \
    --num-steps=10 --refine-steps=2 \
    --algorithm=newton --maxiter=50 --eta=0.5 --tol=1e-5 \
    -f -o out/rn2rd
# Radial to tactoid
echo radial lam+
python3 nlc_cont.py cont2.json cont3.json -i radial64.npy -N=63 \
    --num-steps=10 --refine-steps=0 \
    --algorithm=newton --maxiter=15 --eta=0.2 --tol=1e-5 \
    -f -o out/rd2tc
# Tactoid to radial
echo tactoid lam-
python3 nlc_cont.py cont3.json cont2.json -i tactoid64.npy -N=63 \
    --num-steps=10 --refine-steps=2 \
    --algorithm=newton --maxiter=15 --eta=0.2 --tol=1e-5 \
    -f -o out/tc2rd

