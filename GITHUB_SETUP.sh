# GitHub Setup Guide
# Distributed PageRank with MPI
# =============================================

# ── STEP 1: Initialize the repo locally ───────────────────────────────────────

cd distributed-pagerank/

git init
git add .
git commit -m "Initial commit: distributed PageRank with mpi4py on SeaWulf HPC"

# ── STEP 2: Create the GitHub repo ────────────────────────────────────────────
# Option A — GitHub CLI (fastest):

gh repo create distributed-pagerank \
    --public \
    --description "Distributed PageRank on ~10M-node web graph using mpi4py and SLURM on SeaWulf HPC" \
    --push \
    --source=.

# Option B — Manual:
# 1. Go to https://github.com/new
# 2. Name:        distributed-pagerank
# 3. Description: Distributed PageRank on ~10M-node web graph using mpi4py and SLURM on SeaWulf HPC
# 4. Public, no README (you already have one)
# 5. Then run:

git remote add origin https://github.com/YOUR_USERNAME/distributed-pagerank.git
git branch -M main
git push -u origin main

# ── STEP 3: Add topics ────────────────────────────────────────────────────────
# GitHub repo → gear ⚙ next to "About" → add topics:
#   pagerank  mpi  hpc  graph-algorithms  distributed-computing
#   python  mpi4py  parallel-computing  big-data  slurm

# ── STEP 4: Pin to your GitHub profile ────────────────────────────────────────
# GitHub profile → "Customize your pins" → select this repo
