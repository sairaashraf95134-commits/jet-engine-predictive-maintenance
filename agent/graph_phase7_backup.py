from langgraph.graph import StateGraph, END

from .state import AgentState

from .tools import (
    get_engine_status,
    get_sensor_trends,
    get_feature_importance,
)


def validate_input(state: AgentState):

    if "predicted_rul" not in state:

        state["validation_result"] = (
            "Prediction data is missing."
        )

    else:

        state["validation_result"] = (
            "Input validation successful."
        )

    return state


def analyze_engine(state: AgentState):

    rul = float(
        state["predicted_rul"]
    )

    status = get_engine_status(rul)

    state["health_status"] = status

    state["final_assessment"] = (
        f"Engine {state['engine_id']} has an estimated "
        f"Remaining Useful Life of {rul:.2f} cycles. "
        f"The model-based health status is {status}."
    )

    return state


def analyze_sensors(state: AgentState):

    sensor_data = state.get(
        "recent_sensor_data",
        {}
    )

    state["sensor_analysis"] = (
        get_sensor_trends(sensor_data)
    )

    return state


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


def generate_recommendation(state: AgentState):

    status = state["health_status"]

    if status == "Healthy":

        recommendation = (
            "Continue routine monitoring. "
            "The model does not currently indicate "
            "an immediate high-risk condition."
        )

    elif status == "Degrading":

        recommendation = (
            "Increase monitoring frequency and consider "
            "maintenance planning using additional "
            "engineering information."
        )

    else:

        recommendation = (
            "Prioritize engineering review and maintenance "
            "planning because the model indicates a "
            "relatively low remaining useful life."
        )

    state["recommendation"] = recommendation

    return state


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
        END
    )

    return workflow.compile()