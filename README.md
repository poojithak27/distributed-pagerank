# Distributed PageRank with MPI

Parallel implementation of the **PageRank algorithm** using the taxation model across a directed web graph (~10M nodes, ~10M edges), executed on the [SeaWulf HPC cluster](https://it.stonybrook.edu/services/high-performance-computing) at Stony Brook University with **5 MPI ranks** via `mpi4py`.

---

## Overview

PageRank is computed iteratively using the taxation (random-surfer) model. The graph is stored as edge-list files partitioned across disk; each MPI rank loads a subset of files, builds a local adjacency list, and contributes to the global PageRank vector via `MPI.Allreduce` at every iteration. Dangling nodes (out-degree 0) are handled explicitly so the probability mass stays normalized.

---

## Repository Structure

```
distributed-pagerank/
├── pagerank_mpi.py          # Main distributed PageRank solver
├── pagerank_mpi.slurm       # SLURM job script for SeaWulf HPC
├── requirements.txt         # Python dependencies
├── results/
│   └── top10_pagerank.txt   # Final top-10 ranked nodes
└── README.md
```

---

## Algorithm

### PageRank Formula (Taxation Model)

```
PR^(t+1)(v) = (1-β)/N  +  β * Σ_{u∈In(v)} PR^(t)(u) / outdeg(u)  +  β * D^(t) / N
```

where:
- `β = 0.9` is the damping factor
- `N` is the total number of distinct nodes
- `D^(t)` is the total PageRank mass held by dangling nodes at iteration `t`

### Execution Phases

| Phase | Description |
|-------|-------------|
| **1. File loading** | Each rank reads its assigned files and builds a local adjacency list |
| **2. Node indexing** | Rank 0 aggregates all node IDs into a global index; broadcast to all ranks |
| **3. Out-degree computation** | Local out-degrees merged via `Allreduce` into a global array |
| **4. Iteration** | 4 rounds of the PageRank update; contributions merged via `Allreduce` per round |
| **5. Output** | Rank 0 writes per-iteration CSVs and final top-10 files |

### File Distribution

Files are assigned round-robin across ranks:
```
file i → rank (i mod num_processes)
```
An alternative `--hash_mod_source` flag shards by `hash(source_node) % size` for single large files.

---

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--beta`  | 0.9   | Damping factor |
| `--iters` | 4     | Number of iterations |
| MPI ranks | 5     | One per input file partition |
| Nodes     | ~10M  | Distinct nodes in the graph |
| Edges     | ~10M  | Total directed edges |

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

Requires an MPI installation (MPICH or OpenMPI) and `mpi4py`.

### Data

Edge-list files live at (SeaWulf):
```
/gpfs/projects/AMS598/projects2025_data/project2_data/
```
Each line is a pair `u v` or `u, v` representing a directed edge u → v. Data is not included in this repo due to size.

---

## Running

### On SeaWulf HPC (SLURM)

```bash
sbatch pagerank_mpi.slurm
```

Monitor with:
```bash
squeue -j <JOBID>
sacct  -j <JOBID>
```

### Locally (any MPI install)

```bash
mpirun -np 5 python3 pagerank_mpi.py \
    --input  /path/to/edge/files \
    --outdir ./outputs \
    --beta   0.9 \
    --iters  4
```

---

## Output Files

| File | Description |
|------|-------------|
| `pagerank_iter_1.csv` … `pagerank_iter_4.csv` | All ~10M node scores after each iteration |
| `top10_pagerank.txt` | Top-10 nodes by final PageRank score |
| `top10_pagerank.json` | Same results in JSON format |

---

## Results

Runtime on SeaWulf: **~8–10 minutes** (1 node, 5 MPI ranks, debug-28core partition).

### Top 10 Nodes by PageRank

| Rank | Node ID | PageRank Score |
|------|---------|---------------|
| 1 | 2817335 | 2.186196e-05 |
| 2 | 9423027 | 2.162894e-05 |
| 3 | 3465571 | 2.133396e-05 |
| 4 | 5132918 | 2.100575e-05 |
| 5 | 6813130 | 2.087437e-05 |
| 6 | 8065384 | 2.071829e-05 |
| 7 | 1460753 | 2.070437e-05 |
| 8 | 4159429 | 1.829233e-05 |
| 9 | 7495117 | 1.584945e-05 |
| 10 | 878533 | 1.547294e-05 |

### Validation

- Sum of all PageRank values ≈ 1.0 (correctly normalized at each iteration)
- No negative or NaN values found in any output
- Rankings stabilized after iteration 3, indicating near-convergence
- Each CSV contained ~10 million rows

### Why scores are ~10⁻⁵

With N ≈ 10 million nodes, a uniform distribution gives each node `1/N ≈ 10⁻⁷`. The top nodes here score ~200× higher than average, reflecting genuine hub structure in the web graph.

---

## Key Design Decisions

- **No file I/O between ranks** — all communication via `MPI.Allreduce`
- **Dangling node handling** — mass from zero-out-degree nodes is redistributed uniformly at each iteration, preserving normalization
- **Round-robin file assignment** — works for any number of MPI processes
- **Re-normalization** — PageRank vector is normalized after each iteration to prevent floating-point drift
- **`HYDRA_LAUNCHER=fork`** — required to avoid launcher conflicts on SeaWulf

---

## Resume / LinkedIn Bullet Points

**1. Distributed Graph Algorithm on HPC**
Implemented the PageRank taxation model across a ~10M-node web graph using 5 MPI ranks on SeaWulf, distributing edge-list files round-robin and merging contributions via `MPI.Allreduce` at each iteration.
Correctly handled dangling nodes to maintain probability mass normalization, producing validated output where all scores summed to 1.0 with no NaN values.

**2. Large-Scale Parallel Data Processing**
Processed ~10 million edges across partitioned input files with no intermediate file I/O between processes — all inter-rank communication done through collective MPI operations.
Achieved convergence in 4 iterations (~8–10 min runtime), with top-ranked nodes stabilizing after iteration 3 and results written as both CSV and JSON for reproducibility.

**3. End-to-End HPC Pipeline with SLURM**
Built a complete parallel pipeline from edge-list ingestion to ranked output, scheduled via SLURM on a 28-core debug partition with environment validation and structured logging baked in.
Designed the system to scale to any number of MPI processes using hash-based or round-robin sharding, making the implementation adaptable beyond the 5-rank assignment configuration.

---

## Skills & Tools

`Python` · `mpi4py` · `NumPy` · `PageRank` · `MPI` · `SLURM` · `HPC` · `Graph Algorithms` · `Distributed Computing` · `Big Data`

---

## Author

Poojitha K — AMS 598, Stony Brook University, Spring 2025
