#!/usr/bin/env python3
"""
Distributed PageRank with mpi4py.

Implements the taxation model over a directed graph stored as edge-list files:

    PR^(t+1)(v) = (1-β)/N  +  β * Σ_{u∈In(v)} PR^(t)(u)/outdeg(u)  +  β * D^(t)/N

where D^(t) is the total PageRank mass held by dangling nodes (out-degree 0).

Features
--------
- Works with any number of MPI ranks (assignment uses 5)
- Handles dangling nodes explicitly
- Round-robin file distribution OR hash-mod-source sharding
- Writes pagerank_iter_{1..K}.csv and top-10 files to --outdir

Usage
-----
    mpirun -np 5 python3 pagerank_mpi.py \\
        --input  /path/to/edge/files \\
        --outdir /path/to/outputs \\
        --beta   0.9 \\
        --iters  4
"""

from mpi4py import MPI
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Regex: matches "u v", "u, v", "(u, v)" — tolerates negative IDs
# ---------------------------------------------------------------------------
EDGE_RE = re.compile(r"(-?\d+)[,\s]+(-?\d+)")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(msg: str, rank: int = 0) -> None:
    """Print only from rank 0."""
    if rank == 0:
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Distributed PageRank (mpi4py)")
    ap.add_argument("--input",  required=True,
                    help="Edge-list file or directory of edge-list files")
    ap.add_argument("--outdir", required=True,
                    help="Directory for output CSV and text files")
    ap.add_argument("--beta",   type=float, default=0.9,
                    help="Damping factor β (default: 0.9)")
    ap.add_argument("--iters",  type=int,   default=4,
                    help="Number of PageRank iterations (default: 4)")
    ap.add_argument("--hash_mod_source", action="store_true",
                    help="Shard by hash(source) %% size instead of round-robin files")
    return ap.parse_args()


def discover_files(path: str) -> list:
    """Return sorted list of input files (handles single file or directory)."""
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and not f.startswith(".")
        )
    return [path]


def parse_edge(line: str):
    """Return (u, v) int pair, or None if the line has no valid edge."""
    m = EDGE_RE.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    log(f"[INFO] Starting PageRank on {size} MPI ranks", rank)

    args = parse_args()

    # Rank 0 discovers files, broadcasts to all
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        files = discover_files(args.input)
        if not files:
            print("No input files found.", file=sys.stderr)
            comm.Abort(1)
    else:
        files = None
    files = comm.bcast(files, root=0)
    log(f"[INFO] Discovered {len(files)} input file(s)", rank)

    beta  = args.beta
    iters = args.iters

    # ------------------------------------------------------------------
    # Phase 1: Build local adjacency list
    # ------------------------------------------------------------------
    local_adj   = defaultdict(list)   # source -> [destinations]
    local_nodes = set()

    # File assignment: round-robin unless --hash_mod_source
    assigned_files = (
        files
        if args.hash_mod_source
        else [f for i, f in enumerate(files) if i % size == rank]
    )

    read_edges = 0
    for fpath in assigned_files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                edge = parse_edge(line)
                if edge is None:
                    continue
                u, v = edge
                if args.hash_mod_source and (hash(u) % size) != rank:
                    continue
                local_adj[u].append(v)
                local_nodes.add(u)
                local_nodes.add(v)
                read_edges += 1

    total_edges = comm.allreduce(read_edges, op=MPI.SUM)
    log(f"[INFO] Total edges read: {total_edges:,}", rank)

    # ------------------------------------------------------------------
    # Phase 2: Build global node index (rank 0 aggregates, then broadcasts)
    # ------------------------------------------------------------------
    local_ids = np.array(sorted(local_nodes), dtype=np.int64)
    all_ids   = comm.allgather(local_ids)

    if rank == 0:
        seen       = set()
        global_ids = []
        for arr in all_ids:
            for x in arr:
                if x not in seen:
                    seen.add(x)
                    global_ids.append(int(x))
        global_ids = np.array(global_ids, dtype=np.int64)
    else:
        global_ids = None
    global_ids = comm.bcast(global_ids, root=0)

    N      = int(global_ids.shape[0])
    id2idx = {int(nid): i for i, nid in enumerate(global_ids)}
    log(f"[INFO] Total distinct nodes: {N:,}", rank)

    # ------------------------------------------------------------------
    # Phase 3: Compute global out-degrees
    # ------------------------------------------------------------------
    local_outdeg = np.zeros(N, dtype=np.int64)
    for u, vs in local_adj.items():
        local_outdeg[id2idx[u]] = len(vs)
    global_outdeg = np.zeros_like(local_outdeg)
    comm.Allreduce(local_outdeg, global_outdeg, op=MPI.SUM)

    # Each rank owns sources where hash(u) % size == rank
    owned_sources = [u for u in local_adj if (hash(u) % size) == rank]

    # ------------------------------------------------------------------
    # Phase 4: PageRank iterations
    # ------------------------------------------------------------------
    pr        = np.full(N, 1.0 / max(N, 1), dtype=np.float64)
    teleport  = (1.0 - beta) / max(N, 1)

    for it in range(1, iters + 1):
        local_contrib  = np.zeros(N, dtype=np.float64)
        local_dangling = 0.0

        for u in owned_sources:
            ui  = id2idx[u]
            deg = global_outdeg[ui]
            if deg == 0:
                local_dangling += pr[ui]
            else:
                share = pr[ui] / deg
                for v in local_adj[u]:
                    local_contrib[id2idx[v]] += beta * share

        # Merge contributions across all ranks
        global_contrib = np.zeros_like(local_contrib)
        comm.Allreduce(local_contrib, global_contrib, op=MPI.SUM)

        total_dangling = comm.allreduce(local_dangling, op=MPI.SUM)

        # Full PageRank update
        new_pr  = global_contrib + beta * (total_dangling / max(N, 1)) + teleport

        # Re-normalize to ensure sum = 1
        s = np.sum(new_pr)
        if s > 0:
            new_pr /= s
        pr = new_pr

        # Write per-iteration CSV (rank 0 only)
        if rank == 0:
            out_path = os.path.join(args.outdir, f"pagerank_iter_{it}.csv")
            with open(out_path, "w") as fout:
                fout.write("node_id,pr\n")
                for nid, val in zip(global_ids, pr):
                    fout.write(f"{nid},{val:.12e}\n")
            log(f"[INFO] Wrote pagerank_iter_{it}.csv  (sum={pr.sum():.6f})", rank)

    # ------------------------------------------------------------------
    # Phase 5: Write top-10 output (rank 0 only)
    # ------------------------------------------------------------------
    if rank == 0:
        k = min(10, N)
        if N > 0:
            idx_topk  = np.argpartition(-pr, k - 1)[:k]
            idx_topk  = idx_topk[np.argsort(-pr[idx_topk])]
            top_nodes = [(int(global_ids[i]), float(pr[i])) for i in idx_topk]
        else:
            top_nodes = []

        # Plain-text output
        txt_path = os.path.join(args.outdir, "top10_pagerank.txt")
        with open(txt_path, "w") as fout:
            fout.write(f"Top {k} webpages by PageRank (beta={beta}, iters={iters})\n")
            for i, (nid, score) in enumerate(top_nodes, 1):
                fout.write(f"{i:2d}. node={nid:>10d}  pr={score:.12e}\n")

        # JSON output for reproducibility
        json_path = os.path.join(args.outdir, "top10_pagerank.json")
        with open(json_path, "w") as jf:
            json.dump(
                [{"rank": i + 1, "node_id": nid, "pr": sc}
                 for i, (nid, sc) in enumerate(top_nodes)],
                jf,
                indent=2,
            )

        log("[INFO] Wrote top10_pagerank.txt and top10_pagerank.json", rank)
        log("[INFO] Done.", rank)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
