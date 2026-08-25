import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_agent_graph


print("=" * 60)
print("PHASE 8 END-TO-END AGENT TEST")
print("=" * 60)


# ---------------------------------------------------------
# 1. Build LangGraph agent
# ---------------------------------------------------------

agent = build_agent_graph()

print("\n[PASS] LangGraph agent created.")


# ---------------------------------------------------------
# 2. Prepare ML output + analysis data
# ---------------------------------------------------------
#
# The GRU model is responsible for producing predicted_rul.
# LangGraph receives that result and interprets it.
#
# These values represent the output already demonstrated
# by the working application for Engine 46.
# ---------------------------------------------------------

initial_state = {
    "engine_id": 46,

    "latest_cycle": 146,

    "predicted_rul": 43.84,

    "recent_sensor_data": {
        "sensor_2": [100, 105, 110, 120, 125, 130],
        "sensor_7": [200, 205, 210, 215, 220, 240],
        "sensor_11": [300, 305, 310, 315, 320, 340],
    },

    "feature_importance": [
        {
            "feature": "sensor_11",
            "importance": 0.15,
        },
        {
            "feature": "sensor_12",
            "importance": 0.12,
        },
        {
            "feature": "sensor_13",
            "importance": 0.10,
        },
    ],
}


print("\nEngine:", initial_state["engine_id"])
print("Latest cycle:", initial_state["latest_cycle"])
print("Predicted RUL:", initial_state["predicted_rul"])


# ---------------------------------------------------------
# 3. Run LangGraph
# ---------------------------------------------------------

print("\nRunning LangGraph workflow...")

try:

    result = agent.invoke(initial_state)

    print("\n[PASS] LangGraph workflow completed.")


    # -----------------------------------------------------
    # 4. Validate returned state
    # -----------------------------------------------------

    assert "validation_result" in result
    assert "health_status" in result
    assert "final_assessment" in result
    assert "sensor_analysis" in result
    assert "feature_analysis" in result
    assert "recommendation" in result


    # -----------------------------------------------------
    # 5. Display agent output
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("AGENT OUTPUT")
    print("-" * 60)

    print("\nValidation:")
    print(result["validation_result"])

    print("\nAssessment:")
    print(result["final_assessment"])

    print("\nHealth Status:")
    print(result["health_status"])

    print("\nSensor Analysis:")
    print(result["sensor_analysis"])

    print("\nFeature Analysis:")
    print(result["feature_analysis"])

    print("\nRecommendation:")
    print(result["recommendation"])


    # -----------------------------------------------------
    # 6. Final test result
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 8 END-TO-END AGENT TEST PASSED")
    print("=" * 60)


except Exception as e:

    print("\n" + "=" * 60)
    print("PHASE 8 END-TO-END AGENT TEST FAILED")
    print("=" * 60)

    print("\nError:")
    print(type(e).__name__, str(e))

    raise