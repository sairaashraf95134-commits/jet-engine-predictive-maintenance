from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):

    # Engine information
    engine_id: int
    latest_cycle: int

    # ML prediction
    predicted_rul: float
    health_status: str

    # Raw analysis data
    recent_sensor_data: Dict[str, List[float]]
    feature_importance: List[Dict[str, Any]]

    # User question
    user_question: str

    # Agent outputs
    validation_result: str
    sensor_analysis: str
    feature_analysis: str
    final_assessment: str

    # Detailed interpretation
    risk_level: str
    degradation_summary: str
    key_risk_factors: List[str]

    # Recommendation
    recommendation: str

    # Final response
    agent_review: str