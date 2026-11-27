"""Deterministic network metrics export. Spec §3.5, defect #9.

Notebook 01 previously wrote only net.save_graph() HTML and one savefig() PNG, so
the claim 'computed metrics fed to Stage 1' had nothing to load. This produces a
versioned artifact whose hash enters the run manifest.

Determinism requirements: sorted keys, fixed float precision, seeded Louvain, and
no timestamps. Any of those missing and the manifest hash changes run to run.
"""
import json
import sys
import warnings
from pathlib import Path

import networkx as nx

SCHEMA_VERSION = 1
PRECISION = 6
LOUVAIN_SEED = 20260821

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "networks"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _round(x: float) -> float:
    return round(float(x), PRECISION)


def compute_metrics(graph: nx.Graph, corpus: str, source: str | None = None,
                     n_hadiths: int | None = None) -> dict:
    """Per-node centralities + Louvain community, plus graph-level stats.

    `source`/`n_hadiths` record which input built the graph (the 500-item sample,
    not the full raw corpus -- see build_and_export) so a reader of the artifact
    knows what it covers. Always present, even when the caller omits them, so the
    schema is stable for every consumer of this file.
    """
    provenance = {"source": source, "n_hadiths": n_hadiths}
    if graph.number_of_nodes() == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "corpus": corpus,
            "provenance": provenance,
            "graph": {"n_nodes": 0, "n_edges": 0, "modularity": 0.0,
                      "n_components": 0, "component_sizes": []},
            "nodes": [],
        }

    if graph.is_directed():
        graph = graph.to_undirected()

    import community as community_louvain  # python-louvain

    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, seed=LOUVAIN_SEED)
    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        warnings.warn(
            f"eigenvector_centrality did not converge on {corpus!r} graph "
            f"({graph.number_of_nodes()} nodes); setting all eigenvector values to 0.0",
            RuntimeWarning,
            stacklevel=2,
        )
        eigenvector = {n: 0.0 for n in graph}

    partition = community_louvain.best_partition(graph, random_state=LOUVAIN_SEED)
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)

    nodes = [
        {
            "id": str(n),
            "degree": _round(degree[n]),
            "betweenness": _round(betweenness[n]),
            "eigenvector": _round(eigenvector[n]),
            "community": int(partition[n]),
        }
        for n in sorted(graph.nodes(), key=str)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "corpus": corpus,
        "provenance": provenance,
        "graph": {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "modularity": _round(community_louvain.modularity(partition, graph)),
            "n_components": len(components),
            "component_sizes": components,
        },
        "nodes": nodes,
    }


def write_metrics(metrics: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def export(graph: nx.Graph, corpus: str) -> Path:
    """Convenience entry point for notebook 01."""
    return write_metrics(compute_metrics(graph, corpus), OUT_DIR / f"{corpus}_metrics.json")


def build_and_export(corpus: str) -> Path:
    """CLI entry point: build the isnad graph and write its metrics.

    Built from poc/data/<corpus>_sample.json -- the 500-item sample notebook 01
    already drew (seed 42) and cleaned -- not the raw corpus. The raw files nest
    hadiths under a different shape per corpus (bukhari: chapters/sub_chapters,
    kafi: chapters/hadiths) and re-sampling them here would be a second, possibly
    divergent, re-derivation of the sample the notebook's numbers were checked
    against. The sample size and file are recorded in `provenance` so a reader of
    the metrics knows the graph covers 500 hadiths, not the full corpus.
    """
    from hermeneutics.isnad_graph import build_isnad_graph, load_sample

    hadiths = load_sample(corpus)
    graph = build_isnad_graph(hadiths, corpus)
    metrics = compute_metrics(graph, corpus, source=f"{corpus}_sample.json",
                               n_hadiths=len(hadiths))
    return write_metrics(metrics, OUT_DIR / f"{corpus}_metrics.json")


def build_parser():
    """Separate from main so the accepted corpora are testable, as in run_inquiry.py."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the isnad network for a corpus and export its metrics. Spec §3.5.")
    # muslim: al-Nawawi's Minhaj commentates Muslim, so an inquiry under his method
    # file needs Muslim's isnad metrics. build_and_export already handles it -- only
    # this list stood between the function and the command line.
    parser.add_argument("corpus", choices=["bukhari", "kafi", "muslim"])
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    path = build_and_export(args.corpus)
    print(f"{args.corpus}: wrote {path}")


if __name__ == "__main__":
    main()
