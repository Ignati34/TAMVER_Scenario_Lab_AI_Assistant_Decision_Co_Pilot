from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .engine import centrality, monte_carlo, simulate, strongest_paths
from .schemas import (
    AssistantRequest,
    ModelGraph,
    RecommendationRequest,
    SimulationSettings,
)


@dataclass
class ObjectiveProfile:
    risk_weight: float = 1.0
    stability_weight: float = 0.25
    cost_weight: float = 0.10
    robustness_weight: float = 0.35


def infer_objective(command: str) -> ObjectiveProfile:
    text = command.lower()
    profile = ObjectiveProfile()

    if any(word in text for word in ["стабиль", "устойчив", "resilien"]):
        profile.stability_weight = 0.70
    if any(word in text for word in ["бюджет", "дешев", "затрат", "cost"]):
        profile.cost_weight = 0.60
    if any(word in text for word in ["стресс", "худш", "worst", "robust"]):
        profile.robustness_weight = 0.80
    if any(word in text for word in ["рост", "growth", "доход", "выручк"]):
        # Growth remains represented by node-level maximize targets.
        profile.risk_weight = 0.85
        profile.stability_weight = 0.20

    return profile


def decision_cost(graph: ModelGraph, decisions: Dict[str, float]) -> float:
    costs = {node.id: node.decision_cost for node in graph.nodes}
    return float(
        sum(abs(value) * costs.get(node_id, 1.0)
            for node_id, value in decisions.items())
    )


def candidate_decision(
    graph: ModelGraph,
    rng: np.random.Generator,
) -> Dict[str, float]:
    proposal: Dict[str, float] = {}
    for node in graph.nodes:
        if node.kind != "decision":
            continue

        if rng.random() < 0.30:
            value = 0.0
        else:
            value = rng.uniform(node.decision_min, node.decision_max)

        proposal[node.id] = float(value)
    return proposal


def score_candidate(
    graph: ModelGraph,
    shocks: Dict[str, float],
    decisions: Dict[str, float],
    settings: SimulationSettings,
    objective: ObjectiveProfile,
    budget: Optional[float],
) -> Dict[str, object]:
    base = simulate(graph, shocks, decisions, settings)
    cost = decision_cost(graph, decisions)

    if budget is not None and cost > budget:
        return {
            "score": float("inf"),
            "risk": base.risk_score,
            "stability": base.stability_score,
            "cost": cost,
            "budget_violation": True,
            "impact": base.total_impact.tolist(),
        }

    # Small MC sample for candidate screening.
    screening_settings = settings.model_copy(
        update={"monte_carlo_runs": min(80, settings.monte_carlo_runs)}
    )
    mc = monte_carlo(graph, shocks, decisions, screening_settings)
    robust_risk = float(mc["risk_p95"])

    score = (
        objective.risk_weight * base.risk_score
        + objective.robustness_weight * robust_risk
        + objective.cost_weight * cost
        - objective.stability_weight * (base.stability_score / 100.0)
    )

    return {
        "score": float(score),
        "risk": float(base.risk_score),
        "robust_risk_p95": robust_risk,
        "stability": float(base.stability_score),
        "cost": float(cost),
        "budget_violation": False,
        "impact": base.total_impact.tolist(),
        "warnings": base.warnings,
    }


def explain_recommendation(
    graph: ModelGraph,
    decisions: Dict[str, float],
    evaluation: Dict[str, object],
) -> Dict[str, object]:
    active = [
        node_id
        for node_id, value in decisions.items()
        if abs(value) > 1e-9
    ]

    paths = strongest_paths(graph, active, top_k=5)

    node_map = {node.id: node.label for node in graph.nodes}
    decision_lines = [
        {
            "node_id": node_id,
            "node": node_map.get(node_id, node_id),
            "value": round(value, 4),
            "direction": "усилить" if value > 0 else "сократить",
        }
        for node_id, value in sorted(
            decisions.items(), key=lambda item: abs(item[1]), reverse=True
        )
        if abs(value) > 1e-9
    ][:5]

    return {
        "summary": (
            f"Решение снижает расчетный риск до "
            f"{evaluation['risk']:.4f}; стресс-риск P95 — "
            f"{evaluation['robust_risk_p95']:.4f}; "
            f"ожидаемая устойчивость — {evaluation['stability']:.1f}/100."
        ),
        "actions": decision_lines,
        "causal_paths": paths,
        "cost": evaluation["cost"],
    }


def recommend(request: RecommendationRequest) -> Dict[str, object]:
    rng = np.random.default_rng(request.settings.seed)
    objective = infer_objective(request.command)

    baseline_decisions = {
        node.id: 0.0 for node in request.graph.nodes if node.kind == "decision"
    }
    baseline = score_candidate(
        graph=request.graph,
        shocks=request.shocks,
        decisions=baseline_decisions,
        settings=request.settings,
        objective=objective,
        budget=request.budget,
    )

    pool: List[Dict[str, object]] = []
    for _ in range(request.candidates):
        decisions = candidate_decision(request.graph, rng)
        evaluation = score_candidate(
            graph=request.graph,
            shocks=request.shocks,
            decisions=decisions,
            settings=request.settings,
            objective=objective,
            budget=request.budget,
        )
        if np.isfinite(evaluation["score"]):
            pool.append(
                {
                    "decisions": decisions,
                    "evaluation": evaluation,
                }
            )

    pool.sort(key=lambda item: item["evaluation"]["score"])
    top = pool[: request.top_k]

    enriched = []
    for rank, item in enumerate(top, start=1):
        enriched.append(
            {
                "rank": rank,
                "decisions": item["decisions"],
                "evaluation": item["evaluation"],
                "explanation": explain_recommendation(
                    request.graph,
                    item["decisions"],
                    item["evaluation"],
                ),
            }
        )

    return {
        "command": request.command,
        "objective": objective.__dict__,
        "baseline": baseline,
        "recommendations": enriched,
        "centrality": centrality(request.graph),
    }


def assistant_response(request: AssistantRequest) -> Dict[str, object]:
    text = request.message.lower()

    if any(word in text for word in ["узл", "central", "критическ"]):
        metrics = centrality(request.graph)
        ranked = sorted(
            metrics.items(),
            key=lambda item: (
                item[1]["pagerank"]
                + item[1]["betweenness"]
                + item[1]["eigenvector"]
            ),
            reverse=True,
        )[:5]
        return {
            "intent": "centrality",
            "message": "Критические узлы определены по совокупной сетевой важности.",
            "data": [
                {"node": node, **values} for node, values in ranked
            ],
        }

    if any(word in text for word in ["стресс", "stress", "монте", "monte"]):
        zero_decisions = {
            node.id: 0.0 for node in request.graph.nodes if node.kind == "decision"
        }
        mc = monte_carlo(
            request.graph,
            request.shocks,
            zero_decisions,
            request.settings,
        )
        return {
            "intent": "stress_test",
            "message": (
                f"Стресс-тест завершен. Средний риск: {mc['risk_mean']:.4f}; "
                f"P95: {mc['risk_p95']:.4f}; "
                f"устойчивость: {mc['stability_mean']:.1f}/100."
            ),
            "data": mc,
        }

    recommendation = recommend(
        RecommendationRequest(
            graph=request.graph,
            command=request.message,
            shocks=request.shocks,
            settings=request.settings,
            candidates=200,
            top_k=3,
            budget=request.budget,
        )
    )
    best = recommendation["recommendations"][0] if recommendation["recommendations"] else None

    return {
        "intent": "recommendation",
        "message": (
            best["explanation"]["summary"]
            if best
            else "Допустимое решение в заданных ограничениях не найдено."
        ),
        "data": recommendation,
    }
