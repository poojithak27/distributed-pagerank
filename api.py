from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="PageRank API",
    description="Query PageRank scores from a 10M-node distributed graph computation"
)

def load_results():
    nodes = []
    with open("results/top10_pagerank.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Top") or line.startswith("Validation") or line.startswith("-"):
                continue
            try:
                # parse lines like: " 1. node=   2817335  pr=2.186196173309e-05"
                node_part = line.split("node=")[1].split()[0].strip()
                pr_part = line.split("pr=")[1].strip()
                node_id = int(node_part)
                pr = float(pr_part)
                nodes.append({"rank": len(nodes)+1, "node_id": node_id, "pr": pr})
            except:
                continue
    return nodes

results = load_results()
node_map = {r["node_id"]: r for r in results}

@app.get("/")
def root():
    return {"message": "PageRank API is running"}

@app.get("/top")
def get_top(k: int = 10):
    return {"top_k": k, "results": results[:k]}

@app.get("/pagerank/{node_id}")
def get_node(node_id: int):
    if node_id not in node_map:
        raise HTTPException(status_code=404, detail="Node not found")
    return node_map[node_id]
