from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
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
def load_default_graph() -> dict:
    return json.loads(DEFAULT_GRAPH.read_text(encoding="utf-8"))


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_graph(graph: dict, impact_map: dict | None = None) -> None:
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
            title=(
                f"weight={edge['weight']}; "
                f"confidence={edge.get('confidence', 1)}"
            ),
            color="#FF8A80" if edge["weight"] < 0 else "#90CAF9",
            arrows="to",
        )

    html = network.generate_html(notebook=False)
    components.html(html, height=640, scrolling=True)


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
    rows: list[dict] = []

    baseline = recommendation_data.get("baseline", {})
    if baseline:
        baseline_risk = float(baseline.get("risk", 0.0))
        rows.append(
            {
                "variant": "Baseline",
                "variant_type": "Baseline",
                "cost": 0.0,
                "risk_p05": float(
                    baseline.get("risk_p05", baseline_risk)
                ),
                "risk_mean": float(
                    baseline.get("risk_mean", baseline_risk)
                ),
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
                    evaluation.get(
                        "robust_risk_p95",
                        deterministic_risk,
                    )
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


def quality_score(
    series: pd.Series,
    *,
    higher_is_better: bool,
) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    spread = maximum - minimum

    if spread <= 1e-12:
        return pd.Series(1.0, index=series.index)

    normalized = (series - minimum) / spread
    return normalized if higher_is_better else 1.0 - normalized


def append_special_role(
    frame: pd.DataFrame,
    variant: str,
    role: str,
) -> None:
    mask = frame["variant"] == variant
    current = frame.loc[mask, "special_role"].astype(str)
    frame.loc[mask, "special_role"] = current.apply(
        lambda value: role if not value else f"{value} · {role}"
    )


def render_solution_card(
    title: str,
    row: pd.Series | None,
    *,
    note: str,
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if row is None:
            st.caption("Нет варианта, соответствующего ограничениям.")
            return

        st.metric("Стресс-риск P95", f"{row['risk_p95']:.4f}")
        st.caption(
            f"{row['variant']} · стоимость {row['cost']:.3f} · "
            f"устойчивость {row['stability']:.1f}/100"
        )
        st.caption(note)
        st.caption(f"Действия: {row['actions']}")


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

    baseline_rows = corridor_df[
        corridor_df["variant_type"] == "Baseline"
    ]
    if baseline_rows.empty:
        baseline_p95 = float(corridor_df["risk_p95"].max())
        baseline_stability = float(corridor_df["stability"].max())
        baseline = None
    else:
        baseline = baseline_rows.iloc[0]
        baseline_p95 = float(baseline["risk_p95"])
        baseline_stability = float(baseline["stability"])

    max_observed_cost = max(float(corridor_df["cost"].max()), 0.01)

    st.subheader("Decision Opportunity Corridor")
    st.caption(
        "P05 — благоприятная граница, среднее — ожидаемый риск, "
        "P95 — стресс-граница. Контрольные точки показывают baseline, "
        "минимальный риск, сбалансированный оптимум и минимальную "
        "стоимость допустимого решения."
    )

    controls, visual = st.columns([1, 3])

    with controls:
        st.markdown("#### Параметры коридора")

        default_risk_threshold = max(baseline_p95 * 0.70, 0.0)
        default_max_cost = max(max_observed_cost * 0.75, 0.01)
        default_min_stability = max(
            0.0,
            min(100.0, baseline_stability * 0.85),
        )

        risk_threshold = st.number_input(
            "Максимальный стресс-риск P95",
            min_value=0.0,
            value=float(default_risk_threshold),
            step=max(default_risk_threshold / 20, 0.005),
            format="%.4f",
            key="corridor_risk_threshold_v2",
        )
        max_cost = st.number_input(
            "Максимальная стоимость",
            min_value=0.0,
            value=float(default_max_cost),
            step=max(default_max_cost / 20, 0.01),
            format="%.3f",
            key="corridor_max_cost_v2",
        )
        min_stability = st.slider(
            "Минимальная устойчивость",
            min_value=0.0,
            max_value=100.0,
            value=float(default_min_stability),
            step=1.0,
            key="corridor_min_stability_v2",
        )

        with st.expander("Весовые коэффициенты Balanced Optimum"):
            risk_weight = st.slider(
                "Риск",
                0,
                100,
                50,
                5,
                key="balanced_risk_weight",
            )
            cost_weight = st.slider(
                "Стоимость",
                0,
                100,
                30,
                5,
                key="balanced_cost_weight",
            )
            stability_weight = st.slider(
                "Устойчивость",
                0,
                100,
                20,
                5,
                key="balanced_stability_weight",
            )
            st.caption(
                "Balanced Score нормирует показатели внутри допустимой "
                "зоны и объединяет их с заданными весами."
            )

        show_only_safe = st.checkbox(
            "Показывать только допустимые",
            value=False,
            key="corridor_show_only_safe_v2",
        )

    working_df = corridor_df.copy()
    working_df["zone"] = "Вне ограничений"

    safe_mask = (
        (working_df["risk_p95"] <= risk_threshold)
        & (working_df["cost"] <= max_cost)
        & (working_df["stability"] >= min_stability)
        & (working_df["variant_type"] == "Решение")
    )
    working_df.loc[safe_mask, "zone"] = "Допустимое решение"
    working_df.loc[
        working_df["variant_type"] == "Baseline",
        "zone",
    ] = "Baseline"

    working_df["risk_reduction"] = baseline_p95 - working_df["risk_p95"]
    working_df["risk_reduction_pct"] = (
        working_df["risk_reduction"] / max(baseline_p95, 1e-12) * 100.0
    )
    working_df["efficiency"] = 0.0
    positive_cost = working_df["cost"] > 1e-12
    working_df.loc[positive_cost, "efficiency"] = (
        working_df.loc[positive_cost, "risk_reduction"]
        / working_df.loc[positive_cost, "cost"]
    )

    all_solutions = working_df[
        working_df["variant_type"] == "Решение"
    ].copy()
    safe_solutions = working_df[
        working_df["zone"] == "Допустимое решение"
    ].copy()
    selection_pool = (
        safe_solutions.copy()
        if not safe_solutions.empty
        else all_solutions.copy()
    )

    total_weight = risk_weight + cost_weight + stability_weight
    if total_weight <= 0:
        risk_weight, cost_weight, stability_weight = 50, 30, 20
        total_weight = 100

    if not selection_pool.empty:
        risk_quality = quality_score(
            selection_pool["risk_p95"],
            higher_is_better=False,
        )
        cost_quality = quality_score(
            selection_pool["cost"],
            higher_is_better=False,
        )
        stability_quality = quality_score(
            selection_pool["stability"],
            higher_is_better=True,
        )
        selection_pool["balanced_score"] = 100.0 * (
            (risk_weight / total_weight) * risk_quality
            + (cost_weight / total_weight) * cost_quality
            + (stability_weight / total_weight) * stability_quality
        )
    else:
        selection_pool["balanced_score"] = pd.Series(dtype=float)

    working_df["balanced_score"] = float("nan")
    if not selection_pool.empty:
        score_map = selection_pool.set_index("variant")["balanced_score"]
        working_df["balanced_score"] = working_df["variant"].map(score_map)

    minimum_risk = (
        all_solutions.sort_values(
            ["risk_p95", "cost", "stability"],
            ascending=[True, True, False],
        ).head(1)
        if not all_solutions.empty
        else pd.DataFrame()
    )
    minimum_cost = (
        safe_solutions.sort_values(
            ["cost", "risk_p95", "stability"],
            ascending=[True, True, False],
        ).head(1)
        if not safe_solutions.empty
        else pd.DataFrame()
    )
    balanced_optimum = (
        selection_pool.sort_values(
            ["balanced_score", "risk_p95", "cost"],
            ascending=[False, True, True],
        ).head(1)
        if not selection_pool.empty
        else pd.DataFrame()
    )
    efficiency_champion = (
        selection_pool.sort_values(
            ["efficiency", "risk_p95", "cost"],
            ascending=[False, True, True],
        ).head(1)
        if not selection_pool.empty
        else pd.DataFrame()
    )

    working_df["special_role"] = ""
    if baseline is not None:
        append_special_role(working_df, "Baseline", "Baseline")
    if not minimum_risk.empty:
        append_special_role(
            working_df,
            str(minimum_risk.iloc[0]["variant"]),
            "Minimum Risk",
        )
    if not balanced_optimum.empty:
        append_special_role(
            working_df,
            str(balanced_optimum.iloc[0]["variant"]),
            "Balanced Optimum",
        )
    if not minimum_cost.empty:
        append_special_role(
            working_df,
            str(minimum_cost.iloc[0]["variant"]),
            "Minimum Cost",
        )

    display_df = working_df.copy()
    if show_only_safe:
        display_df = working_df[
            working_df["zone"].isin(
                ["Допустимое решение", "Baseline"]
            )
        ].copy()

    baseline_row = baseline if baseline is not None else None
    min_risk_row = (
        minimum_risk.iloc[0] if not minimum_risk.empty else None
    )
    balanced_row = (
        balanced_optimum.iloc[0]
        if not balanced_optimum.empty
        else None
    )
    min_cost_row = (
        minimum_cost.iloc[0] if not minimum_cost.empty else None
    )
    efficiency_row = (
        efficiency_champion.iloc[0]
        if not efficiency_champion.empty
        else None
    )

    with visual:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Вариантов",
            int((working_df["variant_type"] == "Решение").sum()),
        )
        m2.metric("Допустимых", int(len(safe_solutions)))
        m3.metric("Baseline P95", f"{baseline_p95:.4f}")
        m4.metric(
            "Balanced P95",
            f"{balanced_row['risk_p95']:.4f}"
            if balanced_row is not None
            else "—",
        )

        tooltip = [
            alt.Tooltip("variant:N", title="Вариант"),
            alt.Tooltip("special_role:N", title="Контрольная точка"),
            alt.Tooltip("zone:N", title="Статус"),
            alt.Tooltip("cost:Q", title="Стоимость", format=".3f"),
            alt.Tooltip("risk_p05:Q", title="Риск P05", format=".4f"),
            alt.Tooltip(
                "risk_mean:Q",
                title="Средний риск",
                format=".4f",
            ),
            alt.Tooltip(
                "risk_p95:Q",
                title="Стресс-риск P95",
                format=".4f",
            ),
            alt.Tooltip(
                "risk_reduction_pct:Q",
                title="Снижение P95",
                format=".1f",
            ),
            alt.Tooltip(
                "efficiency:Q",
                title="Эффективность",
                format=".4f",
            ),
            alt.Tooltip(
                "stability:Q",
                title="Устойчивость",
                format=".1f",
            ),
            alt.Tooltip(
                "balanced_score:Q",
                title="Balanced Score",
                format=".1f",
            ),
            alt.Tooltip("actions:N", title="Действия"),
        ]

        band = (
            alt.Chart(display_df)
            .mark_area(opacity=0.16)
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

        interval_rules = (
            alt.Chart(display_df)
            .mark_rule(strokeWidth=2)
            .encode(
                x="cost:Q",
                y="risk_p05:Q",
                y2="risk_p95:Q",
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
            .mark_circle(size=100, opacity=0.92)
            .encode(
                x="cost:Q",
                y="risk_mean:Q",
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

        threshold_rule = (
            alt.Chart(
                pd.DataFrame(
                    {"risk_threshold": [float(risk_threshold)]}
                )
            )
            .mark_rule(
                color="#D32F2F",
                strokeDash=[8, 5],
                strokeWidth=2,
            )
            .encode(y="risk_threshold:Q")
        )

        special_df = display_df[
            display_df["special_role"] != ""
        ].copy()
        special_markers = (
            alt.Chart(special_df)
            .mark_point(size=300, filled=True)
            .encode(
                x="cost:Q",
                y="risk_mean:Q",
                shape=alt.Shape(
                    "special_role:N",
                    title="Контрольные точки",
                ),
                color=alt.Color(
                    "special_role:N",
                    title="Контрольные точки",
                ),
                tooltip=tooltip,
            )
        )
        special_labels = (
            alt.Chart(special_df)
            .mark_text(
                dy=-16,
                fontSize=11,
                fontWeight="bold",
            )
            .encode(
                x="cost:Q",
                y="risk_mean:Q",
                text="special_role:N",
            )
        )

        chart = (
            band
            + interval_rules
            + mean_line
            + points
            + threshold_rule
            + special_markers
            + special_labels
        ).properties(
            height=500,
            title={
                "text": "Коридор возможных решений",
                "subtitle": [
                    "Вертикальный диапазон: P05–P95",
                    "Линия: ожидаемый риск",
                    "Контрольные точки: Baseline, Minimum Risk, "
                    "Balanced Optimum, Minimum Cost",
                ],
            },
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Контрольные варианты")
    card_columns = st.columns(4)
    with card_columns[0]:
        render_solution_card(
            "Baseline",
            baseline_row,
            note="Точка отсчёта: сценарий без дополнительных действий.",
        )
    with card_columns[1]:
        render_solution_card(
            "Minimum Risk",
            min_risk_row,
            note="Минимальный P95 среди всех рассчитанных решений.",
        )
    with card_columns[2]:
        render_solution_card(
            "Balanced Optimum",
            balanced_row,
            note=(
                f"Баланс: риск {risk_weight}%, стоимость "
                f"{cost_weight}%, устойчивость {stability_weight}%."
            ),
        )
    with card_columns[3]:
        render_solution_card(
            "Minimum Cost",
            min_cost_row,
            note="Самое дешёвое решение внутри допустимой зоны.",
        )

    if balanced_row is not None:
        status = (
            "находится в допустимой зоне"
            if balanced_row["zone"] == "Допустимое решение"
            else "выбран из всех решений, поскольку допустимая зона пуста"
        )
        st.success(
            f"**Balanced Optimum: {balanced_row['variant']}** {status}. "
            f"P95: {balanced_row['risk_p95']:.4f}; "
            f"снижение к baseline: "
            f"{balanced_row['risk_reduction_pct']:.1f}%; "
            f"стоимость: {balanced_row['cost']:.3f}; "
            f"устойчивость: {balanced_row['stability']:.1f}/100; "
            f"Balanced Score: {balanced_row['balanced_score']:.1f}. "
            f"Действия: {balanced_row['actions']}."
        )

    if efficiency_row is not None:
        st.info(
            f"**Лидер эффективности: {efficiency_row['variant']}** — "
            f"{efficiency_row['efficiency']:.4f} единицы снижения P95 "
            f"на единицу стоимости; снижение риска "
            f"{efficiency_row['risk_reduction_pct']:.1f}%."
        )

    st.markdown("#### Таблица коридора")
    table_columns = [
        "variant",
        "special_role",
        "zone",
        "cost",
        "risk_p05",
        "risk_mean",
        "risk_p95",
        "risk_reduction_pct",
        "efficiency",
        "stability",
        "balanced_score",
        "actions",
    ]
    table_df = working_df[table_columns].copy()
    table_df = table_df.rename(
        columns={
            "variant": "Вариант",
            "special_role": "Контрольная точка",
            "zone": "Зона",
            "cost": "Стоимость",
            "risk_p05": "P05",
            "risk_mean": "Средний риск",
            "risk_p95": "P95",
            "risk_reduction_pct": "Снижение P95, %",
            "efficiency": "Эффективность",
            "stability": "Устойчивость",
            "balanced_score": "Balanced Score",
            "actions": "Действия",
        }
    )
    st.dataframe(
        table_df.sort_values(
            ["Зона", "P95", "Стоимость"],
            ascending=[True, True, True],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Скачать таблицу коридора CSV",
        data=table_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="tamver_decision_corridor.csv",
        mime="text/csv",
    )

    st.caption(
        "Эффективность = (Baseline P95 − P95 решения) / стоимость. "
        "Balanced Score рассчитывается по нормированным риску, стоимости "
        "и устойчивости внутри допустимой зоны."
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
        st.number_input(
            "Максимальный бюджет",
            min_value=0.0,
            value=0.50,
            step=0.05,
        )
        if budget_enabled
        else None
    )

    st.divider()
    st.header("Исходный шок")
    shocks: dict[str, float] = {}
    system_nodes = [
        node
        for node in st.session_state.graph["nodes"]
        if node["kind"] == "system"
    ]
    selected_shock = st.selectbox(
        "Узел шока",
        [node["id"] for node in system_nodes],
        format_func=lambda node_id: next(
            node["label"]
            for node in system_nodes
            if node["id"] == node_id
        ),
    )
    shock_value = st.slider(
        "Величина шока",
        -0.50,
        0.50,
        0.20,
        0.01,
    )
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
        deterministic = st.session_state.last_result.get(
            "deterministic",
            {},
        )
        impact_map = dict(
            zip(
                deterministic.get("nodes", []),
                deterministic.get("total_impact", []),
            )
        )
    render_graph(st.session_state.graph, impact_map)

with tab_lab:
    st.subheader("Ручной сценарий")
    decision_nodes = [
        node
        for node in st.session_state.graph["nodes"]
        if node["kind"] == "decision"
    ]
    decisions: dict[str, float] = {}
    decision_columns = st.columns(
        min(3, max(1, len(decision_nodes)))
    )
    for index, node in enumerate(decision_nodes):
        with decision_columns[index % len(decision_columns)]:
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
            st.session_state.last_result = api_post(
                "/simulate",
                payload,
            )
        except Exception as exc:
            st.error(f"Ошибка API: {exc}")

    if st.session_state.last_result:
        result = st.session_state.last_result
        deterministic = result["deterministic"]
        monte_carlo = result["monte_carlo"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Risk score",
            f"{deterministic['risk_score']:.4f}",
        )
        c2.metric(
            "Stability",
            f"{deterministic['stability_score']:.1f}/100",
        )
        c3.metric(
            "Stress risk P95",
            f"{monte_carlo['risk_p95']:.4f}",
        )
        c4.metric(
            "Effective radius",
            f"{deterministic['effective_radius']:.3f}",
        )

        impacts = pd.DataFrame(
            {
                "node": deterministic["nodes"],
                "impact": deterministic["total_impact"],
                "p05": monte_carlo["impact_p05"],
                "p95": monte_carlo["impact_p95"],
            }
        ).sort_values(
            "impact",
            key=lambda series: series.abs(),
            ascending=False,
        )
        st.dataframe(impacts, use_container_width=True)

        centrality_df = (
            pd.DataFrame.from_dict(
                result["centrality"],
                orient="index",
            )
            .reset_index(names="node")
            .sort_values("pagerank", ascending=False)
        )
        st.subheader("Network centrality")
        st.dataframe(
            centrality_df,
            use_container_width=True,
        )

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
            with st.spinner(
                "Поиск и стресс-тестирование вариантов..."
            ):
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
        st.session_state.history.append(
            {"role": "user", "content": prompt}
        )
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
                    recommendations = response["data"]["recommendations"]
                    if recommendations:
                        best = recommendations[0]
                        actions = best["explanation"]["actions"]
                        if actions:
                            st.markdown("**Предлагаемые действия:**")
                            st.dataframe(
                                pd.DataFrame(actions),
                                use_container_width=True,
                            )

                        paths = best["explanation"]["causal_paths"]
                        if paths:
                            st.markdown(
                                "**Ключевые цепочки влияния:**"
                            )
                            st.dataframe(
                                pd.DataFrame(paths),
                                use_container_width=True,
                            )

                        comparison = []
                        for recommendation in recommendations:
                            evaluation = recommendation["evaluation"]
                            comparison.append(
                                {
                                    "rank": recommendation["rank"],
                                    "risk_p05": evaluation.get(
                                        "risk_p05"
                                    ),
                                    "risk_mean": evaluation.get(
                                        "risk_mean",
                                        evaluation["risk"],
                                    ),
                                    "stress_risk_p95": evaluation[
                                        "robust_risk_p95"
                                    ],
                                    "stability": evaluation["stability"],
                                    "cost": evaluation["cost"],
                                }
                            )
                        st.markdown(
                            "**Сравнение лучших вариантов:**"
                        )
                        st.dataframe(
                            pd.DataFrame(comparison),
                            use_container_width=True,
                        )
                        st.info(
                            "Полный Pareto-коридор доступен "
                            "во вкладке «Коридор решений»."
                        )
                elif response["intent"] == "centrality":
                    st.dataframe(
                        pd.DataFrame(response["data"]),
                        use_container_width=True,
                    )
                elif response["intent"] == "stress_test":
                    st.json(response["data"])

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": response["message"],
                    }
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
        value=json.dumps(
            st.session_state.graph,
            ensure_ascii=False,
            indent=2,
        ),
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
        uploaded = st.file_uploader(
            "Загрузить graph.json",
            type=["json"],
        )
        if uploaded:
            st.session_state.graph = json.load(uploaded)
            st.session_state.last_recommendation = None
            st.success("Файл загружен")
