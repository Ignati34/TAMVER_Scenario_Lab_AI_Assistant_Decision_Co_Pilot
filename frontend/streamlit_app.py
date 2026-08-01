from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_GRAPH = Path(__file__).resolve().parents[1] / "data" / "sample_graph.json"

st.set_page_config(
    page_title="TAMVER Scenario Lab",
    page_icon="🧭",
    layout="wide",
)

st.title("TAMVER Strategic Decision Intelligence Platform")
st.caption("Scenario Lab · AIR propagation · Monte Carlo · Decision Agent")


@st.cache_data
def load_default_graph():
    return json.loads(DEFAULT_GRAPH.read_text(encoding="utf-8"))


def api_post(path: str, payload: dict):
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_graph(graph: dict, impact_map: dict | None = None):
    network = Network(
        height="620px",
        width="100%",
        directed=True,
        bgcolor="#0E1117",
        font_color="white",
    )

    for node in graph["nodes"]:
        impact = 0.0 if impact_map is None else impact_map.get(node["id"], 0.0)
        size = 24 + min(abs(impact) * 45, 35)
        color = "#4CAF50"
        if impact > 0.15:
            color = "#FF5252"
        elif impact > 0.05:
            color = "#FFC107"
        elif impact < -0.05:
            color = "#42A5F5"

        shape = "diamond" if node["kind"] == "decision" else "dot"
        network.add_node(
            node["id"],
            label=node["label"],
            title=f"Impact: {impact:.4f}",
            color=color,
            shape=shape,
            size=size,
        )

    for edge in graph["edges"]:
        network.add_edge(
            edge["source"],
            edge["target"],
            value=abs(edge["weight"]) * 5,
            title=f"weight={edge['weight']}; confidence={edge.get('confidence', 1)}",
            color="#FF8A80" if edge["weight"] < 0 else "#90CAF9",
            arrows="to",
        )

    html_path = Path("/tmp/tamver_graph.html")
    network.write_html(str(html_path), notebook=False)
    components.html(
        html_path.read_text(encoding="utf-8"),
        height=640,
        scrolling=True,
    )


if "graph" not in st.session_state:
    st.session_state.graph = load_default_graph()
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header("Настройки модели")
    alpha = st.slider("Затухание α", 0.10, 0.95, 0.65, 0.05)
    steps = st.slider("Глубина распространения", 1, 20, 8)
    runs = st.slider("Monte Carlo", 50, 1500, 300, 50)
    seed = st.number_input("Seed", value=42, step=1)
    budget_enabled = st.checkbox("Ограничить бюджет")
    budget = (
        st.number_input("Максимальный бюджет", min_value=0.0, value=0.50, step=0.05)
        if budget_enabled
        else None
    )

    st.divider()
    st.header("Исходный шок")
    shocks = {}
    system_nodes = [
        node for node in st.session_state.graph["nodes"]
        if node["kind"] == "system"
    ]
    selected_shock = st.selectbox(
        "Узел шока",
        [node["id"] for node in system_nodes],
        format_func=lambda node_id: next(
            n["label"] for n in system_nodes if n["id"] == node_id
        ),
    )
    shock_value = st.slider("Величина шока", -0.50, 0.50, 0.20, 0.01)
    shocks[selected_shock] = shock_value

settings = {
    "alpha": alpha,
    "steps": steps,
    "nonlinear": True,
    "monte_carlo_runs": runs,
    "seed": int(seed),
}

tab_graph, tab_lab, tab_agent, tab_data = st.tabs(
    ["Карта системы", "Scenario Lab", "AI-ассистент", "Данные модели"]
)

with tab_graph:
    impact_map = None
    if st.session_state.last_result:
        det = st.session_state.last_result.get("deterministic", {})
        impact_map = dict(
            zip(det.get("nodes", []), det.get("total_impact", []))
        )
    render_graph(st.session_state.graph, impact_map)

with tab_lab:
    st.subheader("Ручной сценарий")
    decision_nodes = [
        node for node in st.session_state.graph["nodes"]
        if node["kind"] == "decision"
    ]
    decisions = {}
    decision_columns = st.columns(min(3, max(1, len(decision_nodes))))
    for idx, node in enumerate(decision_nodes):
        with decision_columns[idx % len(decision_columns)]:
            decisions[node["id"]] = st.slider(
                node["label"],
                float(node["decision_min"]),
                float(node["decision_max"]),
                0.0,
                0.01,
                key=f"decision_{node['id']}",
            )

    if st.button("Запустить сценарий", type="primary"):
        payload = {
            "graph": st.session_state.graph,
            "shocks": shocks,
            "decisions": decisions,
            "settings": settings,
        }
        try:
            st.session_state.last_result = api_post("/simulate", payload)
        except Exception as exc:
            st.error(f"Ошибка API: {exc}")

    if st.session_state.last_result:
        result = st.session_state.last_result
        det = result["deterministic"]
        mc = result["monte_carlo"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk score", f"{det['risk_score']:.4f}")
        c2.metric("Stability", f"{det['stability_score']:.1f}/100")
        c3.metric("Stress risk P95", f"{mc['risk_p95']:.4f}")
        c4.metric("Effective radius", f"{det['effective_radius']:.3f}")

        impacts = pd.DataFrame(
            {
                "node": det["nodes"],
                "impact": det["total_impact"],
                "p05": mc["impact_p05"],
                "p95": mc["impact_p95"],
            }
        ).sort_values("impact", key=lambda s: s.abs(), ascending=False)
        st.dataframe(impacts, use_container_width=True)

        centrality_df = (
            pd.DataFrame.from_dict(result["centrality"], orient="index")
            .reset_index(names="node")
            .sort_values("pagerank", ascending=False)
        )
        st.subheader("Network centrality")
        st.dataframe(centrality_df, use_container_width=True)

with tab_agent:
    st.subheader("Живой Decision Co-Pilot")
    st.caption(
        "Примеры: «Снизь риск при бюджете 0.4», "
        "«Покажи критические узлы», «Проведи стресс-тест»."
    )

    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input("Опиши цель или вопрос к системе")
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload = {
            "graph": st.session_state.graph,
            "message": prompt,
            "shocks": shocks,
            "settings": settings,
            "budget": budget,
        }

        with st.chat_message("assistant"):
            try:
                response = api_post("/assistant/respond", payload)
                st.markdown(response["message"])

                if response["intent"] == "recommendation":
                    recs = response["data"]["recommendations"]
                    if recs:
                        best = recs[0]
                        actions = best["explanation"]["actions"]
                        if actions:
                            st.markdown("**Предлагаемые действия:**")
                            st.dataframe(pd.DataFrame(actions), use_container_width=True)

                        paths = best["explanation"]["causal_paths"]
                        if paths:
                            st.markdown("**Ключевые цепочки влияния:**")
                            st.dataframe(pd.DataFrame(paths), use_container_width=True)

                        comparison = []
                        for rec in recs:
                            evaluation = rec["evaluation"]
                            comparison.append(
                                {
                                    "rank": rec["rank"],
                                    "risk": evaluation["risk"],
                                    "stress_risk_p95": evaluation["robust_risk_p95"],
                                    "stability": evaluation["stability"],
                                    "cost": evaluation["cost"],
                                }
                            )
                        st.markdown("**Сравнение вариантов:**")
                        st.dataframe(pd.DataFrame(comparison), use_container_width=True)
                elif response["intent"] == "centrality":
                    st.dataframe(pd.DataFrame(response["data"]), use_container_width=True)
                elif response["intent"] == "stress_test":
                    st.json(response["data"])

                st.session_state.history.append(
                    {"role": "assistant", "content": response["message"]}
                )
            except Exception as exc:
                message = f"Ошибка API: {exc}"
                st.error(message)
                st.session_state.history.append(
                    {"role": "assistant", "content": message}
                )

with tab_data:
    st.subheader("Редактор JSON-модели")
    edited = st.text_area(
        "Graph JSON",
        value=json.dumps(st.session_state.graph, ensure_ascii=False, indent=2),
        height=600,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Применить JSON"):
            try:
                st.session_state.graph = json.loads(edited)
                st.success("Модель обновлена")
            except json.JSONDecodeError as exc:
                st.error(f"Некорректный JSON: {exc}")
    with col_b:
        uploaded = st.file_uploader("Загрузить graph.json", type=["json"])
        if uploaded:
            st.session_state.graph = json.load(uploaded)
            st.success("Файл загружен")
