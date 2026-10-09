from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx

FEATURE_TO_DECISIONS = {
    "asset_criticality": ["D3", "D4", "D5"],
    "failed_logins": ["D1", "D2", "D5"],
    "login_hour": ["D1"],
    "geo_anomaly": ["D1", "D4", "D5"],
    "threat_intel_score": ["D2", "D3", "D4"],
    "previous_alerts": ["D2", "D3"],
    "source_ip": ["D1"],
}

DECISION_DEPENDENCIES = {
    "D2": ["D4", "D5"],
}


def build_provenance_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for feature, decisions in FEATURE_TO_DECISIONS.items():
        graph.add_node(feature, kind="feature")
        for decision in decisions:
            graph.add_node(decision, kind="decision")
            graph.add_edge(feature, decision)
        for decision in decisions:
            graph.add_node(f"{decision}-output", kind="output")
            graph.add_edge(decision, f"{decision}-output")
    for decision, downstream_decisions in DECISION_DEPENDENCIES.items():
        for downstream in downstream_decisions:
            graph.add_edge(decision, downstream)
    return graph


def downstream_decisions_for_feature(feature: str) -> List[str]:
    return FEATURE_TO_DECISIONS.get(feature, [])


def query_graph(feature: str, graph: nx.DiGraph | None = None) -> List[str]:
    graph = graph or build_provenance_graph()
    if feature not in graph:
        return []
    reachable: Set[str] = set()
    for node in nx.descendants(graph, feature):
        if graph.nodes[node].get("kind") == "decision":
            reachable.add(node)
    return sorted(reachable)


def graph_payload_for_features(features: List[str] | None = None) -> Dict[str, List[Dict[str, str]]]:
    graph = build_provenance_graph()
    selected_features = features if features is not None else list(FEATURE_TO_DECISIONS)
    included = {feature for feature in selected_features if feature in FEATURE_TO_DECISIONS}
    for feature in list(included):
        included.update(nx.descendants(graph, feature))
    nodes = [
        {"id": node, "kind": attrs["kind"]}
        for node, attrs in graph.nodes(data=True)
        if node in included
    ]
    edges = [
        {"source": source, "target": target}
        for source, target in graph.edges()
        if source in included and target in included
    ]
    return {"nodes": nodes, "edges": edges}


def provenance_paths_for_features(features: List[str] | None = None) -> List[Dict[str, object]]:
    graph = build_provenance_graph()
    selected_features = features if features is not None else list(FEATURE_TO_DECISIONS)
    paths: List[Dict[str, object]] = []
    for feature in selected_features:
        if feature not in graph or graph.nodes[feature].get("kind") != "feature":
            continue
        decisions = sorted(
            node
            for node in nx.descendants(graph, feature)
            if graph.nodes[node].get("kind") == "decision"
        )
        for decision in decisions:
            paths.extend(
                {
                    "feature": feature,
                    "decision_id": decision,
                    "path": path,
                }
                for path in nx.all_simple_paths(graph, feature, decision)
            )
    return paths
