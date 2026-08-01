import json
from pathlib import Path

from app.engine import centrality, simulate
from app.schemas import ModelGraph, SimulationSettings


def load_graph():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "data" / "sample_graph.json").read_text())
    return ModelGraph(**payload)


def test_simulation_runs():
    graph = load_graph()
    result = simulate(
        graph=graph,
        shocks={"energy_cost": 0.25},
        decisions={"supplier_diversification": 0.15},
        settings=SimulationSettings(steps=5, monte_carlo_runs=20),
    )
    assert len(result.total_impact) == len(graph.nodes)
    assert result.risk_score >= 0
    assert 0 <= result.stability_score <= 100


def test_centrality_contains_all_nodes():
    graph = load_graph()
    metrics = centrality(graph)
    assert set(metrics) == {node.id for node in graph.nodes}
