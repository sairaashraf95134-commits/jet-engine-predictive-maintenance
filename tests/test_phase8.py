
import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_agent_graph

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_file(path, description):
    full_path = PROJECT_ROOT / path

    if full_path.exists():
        print(f"[PASS] {description}")
        return True

    print(f"[FAIL] {description}")
    print(f"       Missing: {full_path}")
    return False


def main():
    print("=" * 60)
    print("PHASE 8 BASELINE TEST")
    print("=" * 60)

    results = []

    results.append(
        check_file(
            "app/streamlit_app.py",
            "Streamlit application"
        )
    )

    results.append(
        check_file(
            "agent/state.py",
            "LangGraph state"
        )
    )

    results.append(
        check_file(
            "agent/tools.py",
            "Agent tools"
        )
    )

    results.append(
        check_file(
            "agent/graph.py",
            "LangGraph workflow"
        )
    )

    results.append(
        check_file(
            "models/deep_learning/gru_model.keras",
            "GRU model"
        )
    )

    results.append(
        check_file(
            "data/raw/C-MAPSS/train_FD001.txt",
            "Training dataset"
        )
    )

    print()
    print("=" * 60)

    if all(results):
        print("PHASE 8 BASELINE TEST PASSED")
    else:
        print("PHASE 8 BASELINE TEST FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()