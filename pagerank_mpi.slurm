#!/bin/bash
# =============================================================================
# SLURM job script — Distributed PageRank with MPI
# Cluster : SeaWulf (Stony Brook University HPC)
# =============================================================================
#SBATCH -J pagerank_mpi
#SBATCH -o pagerank_mpi.%j.out
#SBATCH -e pagerank_mpi.%j.err
#SBATCH -N 1                       # 1 node
#SBATCH --ntasks=5                 # 5 MPI ranks (one per input file partition)
#SBATCH --time=00:20:00            # wall-clock limit (job ran in ~8-10 min)
#SBATCH --partition=debug-28core

set -euo pipefail

# Load required modules
module purge
module load python/3.9.7
module load mpich

# Prevent Hydra launcher conflicts on SeaWulf
export HYDRA_LAUNCHER=fork

# Activate virtual environment
source ~/mpi_env_mpich/bin/activate

# Sanity checks
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "MPI ranks   : $SLURM_NTASKS"
echo "Start time  : $(date)"
echo "Python      : $(which python) — $(python -V 2>&1)"
echo "mpiexec     : $(which mpiexec)"
python - <<'PY'
from mpi4py import MPI
print("mpi4py OK — MPI vendor:", MPI.get_vendor())
PY
echo "--------------------------------------"

# Paths
INPUT="/gpfs/projects/AMS598/projects2025_data/project2_data"
OUTDIR="/gpfs/projects/AMS598/class2025/Kakkera_Poojitha/project2_outputs"
mkdir -p "$OUTDIR"

# Run
mpiexec -launcher fork -n 5 python -u pagerank_mpi.py \
    --input  "$INPUT"  \
    --outdir "$OUTDIR" \
    --beta   0.9       \
    --iters  4

echo "--------------------------------------"
echo "End time    : $(date)"
