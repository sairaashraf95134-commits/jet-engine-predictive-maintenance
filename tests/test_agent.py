import sys
from pathlib import Path

# Add project root before importing the agent package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_agent_graph

print("=" * 30)
print("LANGGRAPH AGENT TEST")
print("=" * 30)

agent = build_agent_graph()

print()
print("Agent graph created successfully.")
print("LangGraph agent integration test passed.")
