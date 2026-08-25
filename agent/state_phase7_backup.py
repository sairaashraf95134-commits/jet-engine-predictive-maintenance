from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    engine_id: int
    latest_cycle: int
    predicted_rul: float
    health_status: str

    recent_sensor_data: Dict[str, List[float]]

    feature_importance: List[Dict[str, Any]]

    user_question: str

    validation_result: str
    sensor_analysis: str
    feature_analysis: str
    final_assessment: str
    recommendation: str