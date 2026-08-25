from langgraph.graph import StateGraph, END

from .state import AgentState

from .tools import (
    get_engine_status,
    get_sensor_trends,
    get_feature_importance,
)


# ============================================================
# 1. VALIDATION
# ============================================================

def validate_input(state: AgentState):

    required_fields = [
        "engine_id",
        "predicted_rul",
        "recent_sensor_data",
        "feature_importance",
    ]

    missing = [
        field
        for field in required_fields
        if field not in state
    ]

    if missing:

        state["validation_result"] = (
            "Validation failed. Missing fields: "
            + ", ".join(missing)
        )

    else:

        state["validation_result"] = (
            "Input validation successful."
        )

    return state


# ============================================================
# 2. ANALYZE ENGINE / RUL
# ============================================================

def analyze_engine(state: AgentState):

    if "predicted_rul" not in state:

        state["health_status"] = "Unknown"

        state["risk_level"] = "Unknown"

        state["final_assessment"] = (
            "The GRU prediction is unavailable, "
            "so the engine cannot be assessed."
        )

        return state

    rul = float(state["predicted_rul"])

    status = get_engine_status(rul)

    state["health_status"] = status

    # Risk interpretation
    if rul >= 70:

        risk = "Low"

        degradation_summary = (
            "The predicted RUL is relatively high. "
            "The model does not currently indicate "
            "an immediate degradation concern."
        )

    elif rul >= 40:

        risk = "Moderate"

        degradation_summary = (
            "The engine is showing signs of degradation. "
            "The predicted RUL suggests that closer "
            "monitoring may be appropriate."
        )

    elif rul >= 20:

        risk = "High"

        degradation_summary = (
            "The predicted RUL is relatively low. "
            "The engine appears to be approaching "
            "a more advanced degradation region."
        )

    else:

        risk = "Critical"

        degradation_summary = (
            "The predicted RUL is very low. "
            "The model indicates a potentially "
            "advanced degradation condition."
        )

    state["risk_level"] = risk

    state["degradation_summary"] = degradation_summary

    state["final_assessment"] = (
        f"Engine {state['engine_id']} has an estimated "
        f"Remaining Useful Life of {rul:.2f} cycles. "
        f"The model-based health status is {status}. "
        f"The corresponding model interpretation is "
        f"{risk} risk."
    )

    return state


# ============================================================
# 3. SENSOR ANALYSIS
# ============================================================

def analyze_sensors(state: AgentState):

    sensor_data = state.get(
        "recent_sensor_data",
        {}
    )

    analysis = get_sensor_trends(sensor_data)

    state["sensor_analysis"] = analysis

    # Extract individual lines as risk factors
    factors = []

    if isinstance(analysis, str):

        for line in analysis.splitlines():

            line = line.strip()

            if line:
                factors.append(line)

    state["key_risk_factors"] = factors[:5]

    return state


# ============================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ============================================================

def analyze_features(state: AgentState):

    feature_importance = state.get(
        "feature_importance",
        []
    )

    state["feature_analysis"] = (
        get_feature_importance(
            feature_importance
        )
    )

    return state


# ============================================================
# 5. RECOMMENDATION
# ============================================================

def generate_recommendation(state: AgentState):

    status = state.get(
        "health_status",
        "Unknown"
    )

    risk = state.get(
        "risk_level",
        "Unknown"
    )

    if status == "Healthy":

        recommendation = (
            "Continue routine monitoring. "
            "The model does not currently indicate "
            "an immediate high-risk condition. "
            "This prediction should still be interpreted "
            "alongside engineering information."
        )

    elif status == "Degrading":

        recommendation = (
            "Increase monitoring frequency and consider "
            "maintenance planning using additional "
            "engineering information. Review the recent "
            "sensor trends and dominant model features "
            "for further investigation."
        )

    elif status == "Critical":

        recommendation = (
            "Prioritize engineering review because the "
            "model indicates a relatively low remaining "
            "useful life. This is a predictive result and "
            "must not be treated as a certified maintenance "
            "decision."
        )

    else:

        recommendation = (
            "Review the available prediction and sensor "
            "information before making any operational "
            "decision."
        )

    state["recommendation"] = recommendation

    return state


# ============================================================
# 6. FINAL AGENT REVIEW
# ============================================================

def generate_agent_review(state: AgentState):

    engine_id = state.get(
        "engine_id",
        "Unknown"
    )

    rul = state.get(
        "predicted_rul",
        None
    )

    status = state.get(
        "health_status",
        "Unknown"
    )

    risk = state.get(
        "risk_level",
        "Unknown"
    )

    assessment = state.get(
        "final_assessment",
        ""
    )

    sensor_analysis = state.get(
        "sensor_analysis",
        "No sensor analysis available."
    )

    feature_analysis = state.get(
        "feature_analysis",
        "No feature analysis available."
    )

    recommendation = state.get(
        "recommendation",
        "No recommendation available."
    )

    degradation_summary = state.get(
        "degradation_summary",
        ""
    )

    if rul is not None:

        rul_text = f"{float(rul):.2f} cycles"

    else:

        rul_text = "Unavailable"

    review = f"""
ENGINE {engine_id} — AGENT REVIEW
=================================

MODEL PREDICTION
----------------
Estimated RUL: {rul_text}
Health Status: {status}
Risk Level: {risk}

ASSESSMENT
----------
{assessment}

DEGRADATION INTERPRETATION
--------------------------
{degradation_summary}

RECENT SENSOR ANALYSIS
----------------------
{sensor_analysis}

MODEL FEATURE ANALYSIS
----------------------
{feature_analysis}

RECOMMENDATION
--------------
{recommendation}

IMPORTANT LIMITATION
--------------------
This system is a predictive analytics and decision-support
prototype. The GRU prediction and agent interpretation should
not be treated as a certified aviation maintenance decision.
Actual maintenance decisions require appropriate engineering,
inspection, operational, and safety procedures.
"""

    state["agent_review"] = review.strip()

    return state


# ============================================================
# BUILD GRAPH
# ============================================================

def build_agent_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "validate",
        validate_input
    )

    workflow.add_node(
        "analyze_engine",
        analyze_engine
    )

    workflow.add_node(
        "analyze_sensors",
        analyze_sensors
    )

    workflow.add_node(
        "analyze_features",
        analyze_features
    )

    workflow.add_node(
        "recommend",
        generate_recommendation
    )

    workflow.add_node(
        "generate_review",
        generate_agent_review
    )

    workflow.set_entry_point(
        "validate"
    )

    workflow.add_edge(
        "validate",
        "analyze_engine"
    )

    workflow.add_edge(
        "analyze_engine",
        "analyze_sensors"
    )

    workflow.add_edge(
        "analyze_sensors",
        "analyze_features"
    )

    workflow.add_edge(
        "analyze_features",
        "recommend"
    )

    workflow.add_edge(
        "recommend",
        "generate_review"
    )

    workflow.add_edge(
        "generate_review",
        END
    )

    return workflow.compile()