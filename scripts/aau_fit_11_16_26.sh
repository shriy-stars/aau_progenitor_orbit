#!/bin/bash
# Submit from the folder containing aau_fit_11_16_26.py:
#   mkdir -p logs mcmc_results      # logs/ must exist BEFORE you submit
#   sbatch aau_fit_11_16_26.sh

#SBATCH --job-name=aau_mcmc
#SBATCH --output=logs/%x_%j.out          # print output (empty now; results go to summary.rtf)
#SBATCH --error=logs/%x_%j.err           # tracebacks and warnings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40               # emcee can use at most nwalkers/2 = 40 cores
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=YOUR_PARTITION       # <-- EDIT: check available partitions with `sinfo`
#SBATCH --requeue

# --- load your environment (EDIT for your cluster) ---
# module load python
# source activate your_env

export OMP_NUM_THREADS=1                 # one thread per worker; the Pool provides the parallelism
export MPLBACKEND=Agg

cd "$SLURM_SUBMIT_DIR"                   # ../code, ../data, ../potential_files are relative paths
python -u aau_fit_11_16_26.py