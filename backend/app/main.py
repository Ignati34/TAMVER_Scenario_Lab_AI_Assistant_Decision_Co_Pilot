from fastapi import FastAPI

from .agent import assistant_response, recommend
from .engine import centrality, monte_carlo, simulate
from .schemas import AssistantRequest, RecommendationRequest, SimulationRequest

app = FastAPI(
    title="TAMVER Scenario Lab AI",
    version="0.1.0",
    description="Strategic Decision Intelligence Platform prototype",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/simulate")
def run_simulation(request: SimulationRequest):
    deterministic = simulate(
        graph=request.graph,
        shocks=request.shocks,
        decisions=request.decisions,
        settings=request.settings,
    )
    mc = monte_carlo(
        graph=request.graph,
        shocks=request.shocks,
        decisions=request.decisions,
        settings=request.settings,
    )

    return {
        "deterministic": {
            "nodes": deterministic.nodes,
            "total_impact": deterministic.total_impact.tolist(),
            "final_state": deterministic.final_state.tolist(),
            "risk_score": deterministic.risk_score,
            "stability_score": deterministic.stability_score,
            "spectral_radius": deterministic.spectral_radius,
            "effective_radius": deterministic.effective_radius,
            "warnings": deterministic.warnings,
        },
        "monte_carlo": mc,
        "centrality": centrality(request.graph),
    }


@app.post("/agent/recommend")
def run_recommendation(request: RecommendationRequest):
    return recommend(request)


@app.post("/assistant/respond")
def run_assistant(request: AssistantRequest):
    return assistant_response(request)
