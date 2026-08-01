from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from .schemas import ModelGraph, SimulationSettings


@dataclass
class SimulationResult:
    nodes: List[str]
    total_impact: np.ndarray
    final_state: np.ndarray
    path: List[np.ndarray]
    risk_score: float
    stability_score: float
    spectral_radius: float
    effective_radius: float
    warnings: List[str]


def graph_index(graph: ModelGraph) -> Tuple[List[str], Dict[str, int]]:
    nodes = [node.id for node in graph.nodes]
    return nodes, {node_id: idx for idx, node_id in enumerate(nodes)}


def propagation_matrix(
    graph: ModelGraph,
    edge_noise: np.ndarray | None = None,
) -> np.ndarray:
    nodes, index = graph_index(graph)
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)

    for edge_idx, edge in enumerate(graph.edges):
        multiplier = 1.0 if edge_noise is None else float(edge_noise[edge_idx])
        # W[target, source]: shock at source propagates to target.
        matrix[index[edge.target], index[edge.source]] += (
            edge.weight * edge.confidence * multiplier
        )
    return matrix


def spectral_radius(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    values = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(values))) if len(values) else 0.0


def node_loss(value: float, target: float, direction: str) -> float:
    if direction == "minimize":
        return max(value - target, 0.0)
    if direction == "maximize":
        return max(target - value, 0.0)
    return abs(value - target)


def calculate_risk(graph: ModelGraph, impact: np.ndarray) -> float:
    weighted_losses = []
    total_weight = 0.0

    for idx, node in enumerate(graph.nodes):
        if node.kind == "decision":
            continue
        value = node.baseline + float(impact[idx])
        weighted_losses.append(
            node.risk_weight * node_loss(value, node.target, node.direction)
        )
        total_weight += node.risk_weight

    return float(sum(weighted_losses) / max(total_weight, 1e-12))


def calculate_stability(path: Iterable[np.ndarray]) -> float:
    states = list(path)
    if len(states) < 2:
        return 100.0
    increments = np.diff(np.vstack(states), axis=0)
    volatility = float(np.mean(np.abs(increments)))
    return float(100.0 / (1.0 + 10.0 * volatility))


def simulate(
    graph: ModelGraph,
    shocks: Dict[str, float],
    decisions: Dict[str, float],
    settings: SimulationSettings,
    edge_noise: np.ndarray | None = None,
    shock_noise: np.ndarray | None = None,
) -> SimulationResult:
    nodes, index = graph_index(graph)
    W = propagation_matrix(graph, edge_noise=edge_noise)
    rho = spectral_radius(W)
    effective = settings.alpha * rho

    warnings: List[str] = []
    alpha = settings.alpha
    if effective >= 0.98:
        alpha = 0.98 / max(rho, 1e-12)
        warnings.append(
            "Propagation damping was reduced automatically because the "
            "effective spectral radius was unstable."
        )
        effective = alpha * rho

    current = np.zeros(len(nodes), dtype=float)
    for node_id, value in shocks.items():
        if node_id in index:
            current[index[node_id]] += float(value)

    if shock_noise is not None:
        current = current + shock_noise

    # Decisions are injected through decision nodes and then propagate via W.
    for node_id, value in decisions.items():
        if node_id in index:
            current[index[node_id]] += float(value)

    total = current.copy()
    path = [current.copy()]

    for _ in range(settings.steps):
        propagated = alpha * (W @ current)
        current = np.tanh(propagated) if settings.nonlinear else propagated
        total += current
        path.append(current.copy())

    risk = calculate_risk(graph, total)
    stability = calculate_stability(path)

    return SimulationResult(
        nodes=nodes,
        total_impact=total,
        final_state=current,
        path=path,
        risk_score=risk,
        stability_score=stability,
        spectral_radius=rho,
        effective_radius=effective,
        warnings=warnings,
    )


def monte_carlo(
    graph: ModelGraph,
    shocks: Dict[str, float],
    decisions: Dict[str, float],
    settings: SimulationSettings,
) -> Dict[str, object]:
    rng = np.random.default_rng(settings.seed)
    nodes, _ = graph_index(graph)
    impacts: List[np.ndarray] = []
    risks: List[float] = []
    stability: List[float] = []

    edge_count = len(graph.edges)
    uncertainty_vector = np.array(
        [node.uncertainty for node in graph.nodes], dtype=float
    )

    for _ in range(settings.monte_carlo_runs):
        edge_noise = rng.normal(1.0, 0.08, size=edge_count)
        shock_noise = rng.normal(0.0, uncertainty_vector)
        result = simulate(
            graph=graph,
            shocks=shocks,
            decisions=decisions,
            settings=settings,
            edge_noise=edge_noise,
            shock_noise=shock_noise,
        )
        impacts.append(result.total_impact)
        risks.append(result.risk_score)
        stability.append(result.stability_score)

    impact_array = np.vstack(impacts)
    risk_array = np.array(risks, dtype=float)
    stability_array = np.array(stability, dtype=float)

    return {
        "nodes": nodes,
        "impact_mean": impact_array.mean(axis=0).tolist(),
        "impact_p05": np.quantile(impact_array, 0.05, axis=0).tolist(),
        "impact_p95": np.quantile(impact_array, 0.95, axis=0).tolist(),
        "risk_mean": float(risk_array.mean()),
        "risk_p05": float(np.quantile(risk_array, 0.05)),
        "risk_p95": float(np.quantile(risk_array, 0.95)),
        "risk_probability_above_mean": float(
            np.mean(risk_array > risk_array.mean())
        ),
        "stability_mean": float(stability_array.mean()),
    }


def centrality(graph: ModelGraph) -> Dict[str, Dict[str, float]]:
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.id)
    for edge in graph.edges:
        G.add_edge(
            edge.source,
            edge.target,
            weight=abs(edge.weight * edge.confidence),
        )

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank = nx.pagerank(G, weight="weight")

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {node: 0.0 for node in G.nodes}

    result: Dict[str, Dict[str, float]] = {}
    for node in G.nodes:
        result[node] = {
            "degree": float(degree.get(node, 0.0)),
            "betweenness": float(betweenness.get(node, 0.0)),
            "pagerank": float(pagerank.get(node, 0.0)),
            "eigenvector": float(eigenvector.get(node, 0.0)),
        }
    return result


def strongest_paths(
    graph: ModelGraph,
    source_nodes: List[str],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    G = nx.DiGraph()
    for edge in graph.edges:
        strength = abs(edge.weight * edge.confidence)
        if strength <= 0:
            continue
        # Convert strength to additive distance.
        distance = -np.log(max(strength, 1e-9))
        G.add_edge(
            edge.source,
            edge.target,
            distance=float(distance),
            signed_weight=float(edge.weight * edge.confidence),
        )

    found: List[Dict[str, object]] = []
    for source in source_nodes:
        if source not in G:
            continue
        for target in G.nodes:
            if target == source:
                continue
            try:
                path = nx.shortest_path(G, source, target, weight="distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            product = 1.0
            for a, b in zip(path[:-1], path[1:]):
                product *= G[a][b]["signed_weight"]

            found.append(
                {
                    "source": source,
                    "target": target,
                    "path": path,
                    "path_weight": float(product),
                }
            )

    found.sort(key=lambda item: abs(item["path_weight"]), reverse=True)
    return found[:top_k]
