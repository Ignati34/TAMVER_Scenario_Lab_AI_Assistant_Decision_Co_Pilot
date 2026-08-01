from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


NodeKind = Literal["system", "decision"]
ObjectiveDirection = Literal["minimize", "maximize", "target"]


class Node(BaseModel):
    id: str
    label: str
    kind: NodeKind = "system"
    baseline: float = 0.0
    target: float = 0.0
    risk_weight: float = Field(default=1.0, ge=0.0)
    direction: ObjectiveDirection = "target"
    uncertainty: float = Field(default=0.05, ge=0.0)
    decision_min: float = -0.30
    decision_max: float = 0.30
    decision_cost: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.decision_min > self.decision_max:
            raise ValueError("decision_min must be <= decision_max")
        return self


class Edge(BaseModel):
    source: str
    target: str
    weight: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ModelGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

    @model_validator(mode="after")
    def validate_references(self):
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("Node IDs must be unique")
        for edge in self.edges:
            if edge.source not in ids or edge.target not in ids:
                raise ValueError(
                    f"Unknown edge endpoint: {edge.source} -> {edge.target}"
                )
        return self


class SimulationSettings(BaseModel):
    alpha: float = Field(default=0.65, gt=0.0, le=1.0)
    steps: int = Field(default=8, ge=1, le=100)
    nonlinear: bool = True
    monte_carlo_runs: int = Field(default=300, ge=20, le=5000)
    seed: int = 42


class SimulationRequest(BaseModel):
    graph: ModelGraph
    shocks: Dict[str, float] = Field(default_factory=dict)
    decisions: Dict[str, float] = Field(default_factory=dict)
    settings: SimulationSettings = Field(default_factory=SimulationSettings)


class RecommendationRequest(BaseModel):
    graph: ModelGraph
    command: str = "Снизить системный риск"
    shocks: Dict[str, float] = Field(default_factory=dict)
    settings: SimulationSettings = Field(default_factory=SimulationSettings)
    candidates: int = Field(default=250, ge=20, le=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    budget: Optional[float] = Field(default=None, ge=0.0)


class AssistantRequest(BaseModel):
    graph: ModelGraph
    message: str
    shocks: Dict[str, float] = Field(default_factory=dict)
    settings: SimulationSettings = Field(default_factory=SimulationSettings)
    budget: Optional[float] = Field(default=None, ge=0.0)
