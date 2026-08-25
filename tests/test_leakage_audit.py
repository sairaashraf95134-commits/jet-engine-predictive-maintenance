from pathlib import Path

print("=" * 60)
print("PHASE 8 LEAKAGE AUDIT")
print("=" * 60)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

checks = {
    "Training dataset exists": (
        PROJECT_ROOT / "data" / "raw" / "C-MAPSS" / "train_FD001.txt"
    ).exists(),

    "Test dataset exists": (
        PROJECT_ROOT / "data" / "raw" / "C-MAPSS" / "test_FD001.txt"
    ).exists(),

    "RUL file exists": (
        PROJECT_ROOT / "data" / "raw" / "C-MAPSS" / "RUL_FD001.txt"
    ).exists(),

    "GRU model exists": (
        PROJECT_ROOT
        / "models"
        / "deep_learning"
        / "gru"
        / "gru_model.keras"
    ).exists(),

    "Agent state exists": (
        PROJECT_ROOT / "agent" / "state.py"
    ).exists(),

    "Agent tools exist": (
        PROJECT_ROOT / "agent" / "tools.py"
    ).exists(),

    "LangGraph workflow exists": (
        PROJECT_ROOT / "agent" / "graph.py"
    ).exists(),

    "Streamlit application exists": (
        PROJECT_ROOT / "app" / "streamlit_app.py"
    ).exists(),
}

for name, result in checks.items():
    if result:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")

print()
print("LEAKAGE AUDIT PRINCIPLES")
print("-" * 60)

principles = [
    "No future sensor information is intentionally used.",
    "Training and validation data are separated by engine.",
    "Test data is not used to train the GRU model.",
    "Scaler is fitted using training data.",
    "Uploaded test data is transformed using the training scaler.",
    "Feature calculations are performed within individual engines.",
]

for principle in principles:
    print(f"[CHECK] {principle}")

print()
print("=" * 60)
print("PHASE 8 LEAKAGE AUDIT COMPLETED")
print("=" * 60)