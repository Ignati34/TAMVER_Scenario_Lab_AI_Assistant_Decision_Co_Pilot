from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
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


def decision_actions_summary(
    decisions: dict,
    graph: dict,
    limit: int = 3,
) -> str:
    node_labels = {
        node["id"]: node["label"]
        for node in graph["nodes"]
        if node["kind"] == "decision"
    }
    active = sorted(
        (
            (node_id, float(value))
            for node_id, value in decisions.items()
            if abs(float(value)) > 1e-9
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    if not active:
        return "Без дополнительных действий"

    parts = []
    for node_id, value in active[:limit]:
        direction = "+" if value > 0 else ""
        parts.append(
            f"{node_labels.get(node_id, node_id)} {direction}{value:.2f}"
        )
    return "; ".join(parts)


def corridor_dataframe(
    recommendation_data: dict,
    graph: dict,
) -> pd.DataFrame:
    rows = []

    baseline = recommendation_data.get("baseline", {})
    if baseline:
        baseline_risk = float(baseline.get("risk", 0.0))
        rows.append(
            {
                "variant": "Baseline",
                "variant_type": "Baseline",
                "cost": 0.0,
                "risk_p05": float(baseline.get("risk_p05", baseline_risk)),
                "risk_mean": float(baseline.get("risk_mean", baseline_risk)),
                "risk_p95": float(
                    baseline.get("robust_risk_p95", baseline_risk)
                ),
                "stability": float(baseline.get("stability", 0.0)),
                "score": float(baseline.get("score", 0.0)),
                "actions": "Без дополнительных действий",
            }
        )

    corridor_items = recommendation_data.get("decision_corridor")
    if not corridor_items:
        corridor_items = recommendation_data.get("recommendations", [])

    for index, item in enumerate(corridor_items, start=1):
        evaluation = item.get("evaluation", {})
        deterministic_risk = float(evaluation.get("risk", 0.0))
        rows.append(
            {
                "variant": f"Вариант {index}",
                "variant_type": "Решение",
                "cost": float(evaluation.get("cost", 0.0)),
                "risk_p05": float(
                    evaluation.get("risk_p05", deterministic_risk)
                ),
                "risk_mean": float(
                    evaluation.get("risk_mean", deterministic_risk)
                ),
                "risk_p95": float(
                    evaluation.get("robust_risk_p95", deterministic_risk)
                ),
                "stability": float(evaluation.get("stability", 0.0)),
                "score": float(evaluation.get("score", 0.0)),
                "actions": decision_actions_summary(
                    item.get("decisions", {}),
                    graph,
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            subset=["cost", "risk_mean", "risk_p95", "stability"],
            keep="first",
        )
        .sort_values(["cost", "risk_p95", "risk_mean"])
        .reset_index(drop=True)
    )


def render_decision_corridor(
    recommendation_data: dict,
    graph: dict,
) -> None:
    corridor_df = corridor_dataframe(recommendation_data, graph)
    if corridor_df.empty or len(corridor_df) < 2:
        st.info(
            "Недостаточно рассчитанных вариантов. "
            "Запустите поиск решений с большим числом кандидатов."
        )
        return

    baseline_row = corridor_df[
        corridor_df["variant_type"] == "Baseline"
    ]
    if baseline_row.empty:
        baseline_p95 = float(corridor_df["risk_p95"].max())
    else:
        baseline_p95 = float(baseline_row.iloc[0]["risk_p95"])

    max_observed_cost = max(float(corridor_df["cost"].max()), 0.01)
    min_observed_stability = float(corridor_df["stability"].min())

    st.subheader("Decision Opportunity Corridor")
    st.caption(
        "Коридор показывает множество допустимых решений: "
        "P05 — благоприятная граница, среднее — ожидаемый риск, "
        "P95 — стресс-граница."
    )

    controls, visual = st.columns([1, 3])

    with controls:
        st.markdown("#### Параметры коридора")
        risk_threshold = st.number_input(
            "Максимальный стресс-риск P95",
            min_value=0.0,
            value=max(baseline_p95, 0.0),
            step=0.01,
            format="%.4f",
            key="corridor_risk_threshold",
        )
        max_cost = st.number_input(
            "Максимальная стоимость",
            min_value=0.0,
            value=max_observed_cost,
            step=max(max_observed_cost / 20, 0.01),
            format="%.3f",
            key="corridor_max_cost",
        )
        min_stability = st.slider(
            "Минимальная устойчивость",
            min_value=0.0,
            max_value=100.0,
            value=max(0.0, min(100.0, min_observed_stability)),
            step=1.0,
            key="corridor_min_stability",
        )
        show_only_safe = st.checkbox(
            "Показывать только допустимые",
            value=False,
            key="corridor_show_only_safe",
        )

    working_df = corridor_df.copy()
    working_df["zone"] = "Вне ограничений"
    safe_mask = (
        (working_df["risk_p95"] <= risk_threshold)
        & (working_df["cost"] <= max_cost)
        & (working_df["stability"] >= min_stability)
    )
    working_df.loc[safe_mask, "zone"] = "Допустимое решение"
    working_df.loc[
        working_df["variant_type"] == "Baseline",
        "zone",
    ] = "Baseline"

    display_df = working_df
    if show_only_safe:
        display_df = working_df[
            working_df["zone"].isin(["Допустимое решение", "Baseline"])
        ].copy()

    safe_solutions = working_df[
        working_df["zone"] == "Допустимое решение"
    ].copy()
    if safe_solutions.empty:
        best_solution = working_df[
            working_df["variant_type"] == "Решение"
        ].sort_values(
            ["risk_p95", "cost", "stability"],
            ascending=[True, True, False],
        ).head(1)
    else:
        best_solution = safe_solutions.sort_values(
            ["risk_p95", "cost", "stability"],
            ascending=[True, True, False],
        ).head(1)

    if not best_solution.empty:
        best_variant = best_solution.iloc[0]["variant"]
        working_df["is_best"] = working_df["variant"] == best_variant
        display_df["is_best"] = display_df["variant"] == best_variant
    else:
        working_df["is_best"] = False
        display_df["is_best"] = False

    with visual:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Вариантов", int((working_df["variant_type"] == "Решение").sum()))
        m2.metric("Допустимых", int(len(safe_solutions)))
        m3.metric(
            "Лучший P95",
            (
                f"{best_solution.iloc[0]['risk_p95']:.4f}"
                if not best_solution.empty
                else "—"
            ),
        )
        m4.metric(
            "Стоимость лучшего",
            (
                f"{best_solution.iloc[0]['cost']:.3f}"
                if not best_solution.empty
                else "—"
            ),
        )

        tooltip = [
            alt.Tooltip("variant:N", title="Вариант"),
            alt.Tooltip("zone:N", title="Статус"),
            alt.Tooltip("cost:Q", title="Стоимость", format=".3f"),
            alt.Tooltip("risk_p05:Q", title="Риск P05", format=".4f"),
            alt.Tooltip("risk_mean:Q", title="Средний риск", format=".4f"),
            alt.Tooltip("risk_p95:Q", title="Стресс-риск P95", format=".4f"),
            alt.Tooltip("stability:Q", title="Устойчивость", format=".1f"),
            alt.Tooltip("actions:N", title="Действия"),
        ]

        band = (
            alt.Chart(display_df)
            .mark_area(opacity=0.18)
            .encode(
                x=alt.X(
                    "cost:Q",
                    title="Стоимость / интенсивность решения",
                    scale=alt.Scale(zero=True),
                ),
                y=alt.Y(
                    "risk_p05:Q",
                    title="Системный риск",
                    scale=alt.Scale(zero=False),
                ),
                y2="risk_p95:Q",
                order=alt.Order("cost:Q"),
                tooltip=tooltip,
            )
        )

        mean_line = (
            alt.Chart(display_df)
            .mark_line(point=False, strokeWidth=3)
            .encode(
                x="cost:Q",
                y="risk_mean:Q",
                order=alt.Order("cost:Q"),
                tooltip=tooltip,
            )
        )

        points = (
            alt.Chart(display_df)
            .mark_circle(size=110, opacity=0.95)
            .encode(
                x="cost:Q",
                y="risk_mean:Q",
                color=alt.Color(
                    "zone:N",
                    title="Зона решения",
                    scale=alt.Scale(
                        domain=[
                            "Допустимое решение",
                            "Вне ограничений",
                            "Baseline",
                        ],
                        range=["#2E7D32", "#C62828", "#616161"],
                    ),
                ),
                tooltip=tooltip,
            )
        )

        interval_rules = (
            alt.Chart(display_df)
            .mark_rule(strokeWidth=2)
            .encode(
                x="cost:Q",
                y="risk_p05:Q",
                y2="risk_p95:Q",
                color=alt.Color(
                    "zone:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=[
                            "Допустимое решение",
                            "Вне ограничений",
                            "Baseline",
                        ],
                        range=["#2E7D32", "#C62828", "#616161"],
                    ),
                ),
                tooltip=tooltip,
            )
        )

        threshold_data = pd.DataFrame(
            {"risk_threshold": [float(risk_threshold)]}
        )
        threshold_rule = (
            alt.Chart(threshold_data)
            .mark_rule(color="#D32F2F", strokeDash=[8, 5], strokeWidth=2)
            .encode(y="risk_threshold:Q")
        )

        best_marker = (
            alt.Chart(display_df[display_df["is_best"]])
            .mark_point(
                shape="diamond",
                size=260,
                filled=True,
                color="#1565C0",
            )
            .encode(
                x="cost:Q",
                y="risk_mean:Q",
                tooltip=tooltip,
            )
        )

        chart = (
            band
            + interval_rules
            + mean_line
            + points
            + threshold_rule
            + best_marker
        ).properties(
            height=470,
            title={
                "text": "Коридор возможных решений",
                "subtitle": [
                    "Вертикальный диапазон: P05–P95",
                    "Линия: ожидаемый риск",
                    "Синий ромб: рекомендуемый вариант",
                ],
            },
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    if not best_solution.empty:
        best = best_solution.iloc[0]
        status = (
            "находится в допустимой зоне"
            if best["zone"] == "Допустимое решение"
            else "лучший из рассчитанных, но требует пересмотра ограничений"
        )
        st.success(
            f"**{best['variant']}** {status}. "
            f"P95: {best['risk_p95']:.4f}; "
            f"стоимость: {best['cost']:.3f}; "
            f"устойчивость: {best['stability']:.1f}/100. "
            f"Действия: {best['actions']}."
        )

    st.markdown("#### Таблица коридора")
    table_columns = [
        "variant",
        "zone",
        "cost",
        "risk_p05",
        "risk_mean",
        "risk_p95",
        "stability",
        "actions",
    ]
    st.dataframe(
        working_df[table_columns].sort_values(
            ["zone", "risk_p95", "cost"],
            ascending=[True, True, True],
        ),
        use_container_width=True,
        hide_index=True,
    )


if "graph" not in st.session_state:
    st.session_state.graph = load_default_graph()
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = None

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

tab_graph, tab_lab, tab_corridor, tab_agent, tab_data = st.tabs(
    [
        "Карта системы",
        "Scenario Lab",
        "Коридор решений",
        "AI-ассистент",
        "Данные модели",
    ]
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

with tab_corridor:
    st.subheader("Поиск пространства допустимых решений")
    st.caption(
        "Система генерирует варианты, выполняет Monte Carlo, "
        "удаляет доминируемые решения и строит Pareto-коридор."
    )

    corridor_command = st.text_input(
        "Цель поиска",
        value="Снизь системный риск и сохрани устойчивость",
        key="corridor_command",
    )
    candidate_count = st.slider(
        "Количество проверяемых вариантов",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
        key="corridor_candidate_count",
    )

    if st.button(
        "Рассчитать коридор решений",
        type="primary",
        key="calculate_corridor",
    ):
        payload = {
            "graph": st.session_state.graph,
            "command": corridor_command,
            "shocks": shocks,
            "settings": settings,
            "candidates": candidate_count,
            "top_k": 5,
            "budget": budget,
        }
        try:
            with st.spinner("Поиск и стресс-тестирование вариантов..."):
                st.session_state.last_recommendation = api_post(
                    "/agent/recommend",
                    payload,
                )
        except Exception as exc:
            st.error(f"Ошибка API: {exc}")

    if st.session_state.last_recommendation:
        render_decision_corridor(
            st.session_state.last_recommendation,
            st.session_state.graph,
        )
    else:
        st.info(
            "Нажмите «Рассчитать коридор решений». "
            "Результат сохранится в текущей сессии."
        )

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
                    st.session_state.last_recommendation = response["data"]
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
                                    "risk_p05": evaluation.get("risk_p05"),
                                    "risk_mean": evaluation.get(
                                        "risk_mean",
                                        evaluation["risk"],
                                    ),
                                    "stress_risk_p95": evaluation["robust_risk_p95"],
                                    "stability": evaluation["stability"],
                                    "cost": evaluation["cost"],
                                }
                            )
                        st.markdown("**Сравнение лучших вариантов:**")
                        st.dataframe(pd.DataFrame(comparison), use_container_width=True)
                        st.info(
                            "Полный Pareto-коридор доступен "
                            "во вкладке «Коридор решений»."
                        )
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
                st.session_state.last_recommendation = None
                st.success("Модель обновлена")
            except json.JSONDecodeError as exc:
                st.error(f"Некорректный JSON: {exc}")
    with col_b:
        uploaded = st.file_uploader("Загрузить graph.json", type=["json"])
        if uploaded:
            st.session_state.graph = json.load(uploaded)
            st.session_state.last_recommendation = None
            st.success("Файл загружен")
