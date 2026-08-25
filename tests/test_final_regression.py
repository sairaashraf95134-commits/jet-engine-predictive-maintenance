import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_agent_graph


print("=" * 60)
print("FINAL REGRESSION TEST")
print("=" * 60)


# Representative output from the working GRU application
state = {
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


print("\n[1/6] Creating LangGraph agent...")

agent = build_agent_graph()

assert agent is not None

print("[PASS]")


print("\n[2/6] Running workflow...")

result = agent.invoke(state)

print("[PASS]")


print("\n[3/6] Checking prediction...")

assert "predicted_rul" in result
assert isinstance(result["predicted_rul"], (int, float))
assert result["predicted_rul"] >= 0

print(f"[PASS] RUL = {result['predicted_rul']:.2f} cycles")


print("\n[4/6] Checking health status...")

assert result["health_status"] in [
    "Healthy",
    "Degrading",
    "Critical",
]

print(f"[PASS] Status = {result['health_status']}")


print("\n[5/6] Checking agent interpretation...")

assert result.get("validation_result")
assert result.get("final_assessment")
assert result.get("sensor_analysis")
assert result.get("feature_analysis")
assert result.get("recommendation")

print("[PASS]")


print("\n[6/6] Checking recommendation...")

assert len(result["recommendation"].strip()) > 0

print("[PASS]")


print("\n" + "=" * 60)
print("FINAL REGRESSION TEST PASSED")
print("=" * 60)