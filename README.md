# Distributed PageRank with MPI + REST API

A distributed graph processing pipeline that computes PageRank across a **~10M node, ~10M edge** web graph using 5 parallel MPI workers on an HPC cluster, with results served via a **FastAPI REST API**.

---

## System Architecture

```
Edge-list files (partitioned)
        ↓
Distributed MPI computation (5 workers, SeaWulf HPC)
  - Round-robin file sharding across ranks
  - Local adjacency list per worker
  - Contributions merged via MPI.Allreduce each iteration
        ↓
PageRank scores for ~10M nodes (4 iterations, β=0.9)
        ↓
FastAPI REST API — query scores by node, get top-K rankings
```

---

## API Endpoints

Start the server:
```bash
pip install fastapi uvicorn
python -m uvicorn api:app --reload
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/top?k=10` | Top-K nodes by PageRank score |
| GET | `/pagerank/{node_id}` | Score for a specific node |

Interactive docs available at `http://127.0.0.1:8000/docs`

Example response for `/top?k=3`:
```json
{
  "top_k": 3,
  "results": [
    {"rank": 1, "node_id": 2817335, "pr": 2.186196e-05},
    {"rank": 2, "node_id": 9423027, "pr": 2.162894e-05},
    {"rank": 3, "node_id": 3465571, "pr": 2.133396e-05}
  ]
}
```

---

## Repository Structure

```
distributed-pagerank/
├── pagerank_mpi.py        # Distributed PageRank solver (MPI)
├── api.py                 # FastAPI REST API
├── pagerank_mpi.slurm     # SLURM job script for SeaWulf HPC
├── requirements.txt       # Python dependencies
├── results/
│   └── top10_pagerank.txt # Final top-10 ranked nodes
└── README.md
```

---

## Algorithm

### PageRank Formula (Taxation Model)

```
PR^(t+1)(v) = (1-β)/N  +  β * Σ_{u∈In(v)} PR^(t)(u) / outdeg(u)  +  β * D^(t) / N
```

where:
- `β = 0.9` — damping factor
- `N` — total distinct nodes
- `D^(t)` — PageRank mass held by dangling nodes (out-degree 0) at iteration `t`

### Execution Phases

| Phase | Description |
|-------|-------------|
| 1. File loading | Each rank reads its assigned files, builds a local adjacency list |
| 2. Node indexing | Rank 0 aggregates all node IDs into a global index, broadcasts to all ranks |
| 3. Out-degree computation | Local out-degrees merged via `Allreduce` into a global array |
| 4. Iteration | 4 rounds of PageRank update, contributions merged via `Allreduce` per round |
| 5. Output | Rank 0 writes per-iteration CSVs and final top-10 files |

### File Sharding

Round-robin by default:
```
file i → rank (i mod num_processes)
```
Alternative: `--hash_mod_source` shards by `hash(source_node) % size` for single large files.

---

## Setup

### Dependencies
```bash
pip install -r requirements.txt
```
Requires an MPI installation (MPICH or OpenMPI) for the distributed solver.

### Data

Edge-list files (SeaWulf path):
```
/gpfs/projects/AMS598/projects2025_data/project2_data/
```
Each line is a pair `u v` or `u, v` representing a directed edge u → v. Data not included due to size.

---

## Running

### Distributed solver on SeaWulf HPC
```bash
sbatch pagerank_mpi.slurm
```

Monitor:
```bash
squeue -j <JOBID>
sacct  -j <JOBID>
```

### Locally
```bash
mpirun -np 5 python3 pagerank_mpi.py \
    --input  /path/to/edge/files \
    --outdir ./results \
    --beta   0.9 \
    --iters  4
```

---

## Results

Runtime on SeaWulf: **~8–10 minutes** (1 node, 5 MPI ranks, debug-28core partition)

### Top 10 Nodes by PageRank

| Rank | Node ID | PageRank Score |
|------|---------|----------------|
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
- No negative or NaN values in any output
- Rankings stabilized after iteration 3, indicating near-convergence
- Each iteration CSV contained ~10 million rows

Top nodes score ~200× above the uniform baseline (`1/N ≈ 10⁻⁷`), reflecting genuine hub structure in the graph.

---

## Key Design Decisions

- **No file I/O between ranks** — all inter-process communication via `MPI.Allreduce`
- **Dangling node handling** — probability mass from zero-out-degree nodes redistributed uniformly each iteration
- **Round-robin sharding** — works for any number of MPI processes, no hardcoded rank count
- **Re-normalization** — PageRank vector normalized after each iteration to prevent floating-point drift
- **REST API layer** — FastAPI serves precomputed results with automatic OpenAPI docs at `/docs`

---

## Tech Stack

`Python` · `FastAPI` · `mpi4py` · `NumPy` · `MPI` · `SLURM` · `HPC` · `Distributed Computing` · `REST API` · `Graph Algorithms`

---

## Author

Poojitha K — AMS 598, Stony Brook University, Spring 2025
