from __future__ import annotations

import networkx as nx

from backend.store import GraphStore


def pagerank_topk(store: GraphStore, node_ids: list[str], k: int) -> tuple[list[str], str]:
    sg = store.subgraph(node_ids)
    g = nx.DiGraph()
    for n in sg["nodes"]:
        g.add_node(n["id"])
    for e in sg["edges"]:
        g.add_edge(e["src"], e["dst"])
    if g.number_of_nodes() == 0:
        return [], "0 nodes"
    und = g.to_undirected()
    pr = nx.pagerank(und) if und.number_of_edges() else {n: 1.0 for n in und.nodes}
    ranked = sorted(pr, key=pr.get, reverse=True)[:k]
    log = f"{len(node_ids)} nodes -> top {len(ranked)}"
    return ranked, log


def mst_skeleton(store: GraphStore, node_ids: list[str]) -> tuple[dict, str]:
    sg = store.subgraph(node_ids)
    ug = nx.Graph()
    for n in sg["nodes"]:
        ug.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in sg["edges"]:
        if e["src"] == e["dst"]:
            continue
        if not ug.has_edge(e["src"], e["dst"]):
            ug.add_edge(e["src"], e["dst"], rel=e.get("rel", ""))
    n_cycles = max(0, ug.number_of_edges() - max(0, ug.number_of_nodes() - nx.number_connected_components(ug)))
    if ug.number_of_edges() == 0:
        return sg, "0 cycles removed"
    mst = nx.minimum_spanning_tree(ug)
    nodes = [{"id": n, **ug.nodes[n]} for n in mst.nodes]
    edges = [{"src": u, "dst": v, "rel": d.get("rel", "")} for u, v, d in mst.edges(data=True)]
    removed = n_cycles
    return {"nodes": nodes, "edges": edges}, f"{removed} cycles removed"


def louvain_communities(store: GraphStore) -> dict:
    return store.communities()


def deferred_fetch(store: GraphStore, winning_ids: list[str]) -> list[dict]:
    """Fetch raw text only for winning nodes and their adjacent chunks."""
    want = set(winning_ids)
    nb = store.neighbors(list(winning_ids), hops=1)
    for n in nb["nodes"]:
        d = store.g.nodes.get(n, {}) if hasattr(store, "g") else {}
        if d.get("ntype") == "chunk":
            want.add(n)
    texts = store.get_text(list(want))
    out = []
    for i, t in texts.items():
        if not t:
            continue
        meta = {}
        if hasattr(store, "g") and i in store.g:
            meta = store.g.nodes[i]
        if meta.get("ntype") != "chunk" and len(t) < 80:
            continue
        out.append(
            {
                "id": i,
                "text": t,
                "source": meta.get("source", ""),
                "ntype": meta.get("ntype", ""),
            }
        )
    return out
