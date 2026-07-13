"""
workflow.py - LangGraph Multi-Agent Workflow

Defines the directed graph that orchestrates:
  Planner → DataAcquisition → Analyst → Reporter

With conditional edges for error handling and retries.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.analyst import AnalystAgent
from agents.data_acquisition import DataAcquisitionAgent
from agents.planner import PlannerAgent
from agents.reporter import ReporterAgent
from state import GeoAgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Error handling wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _safe_node(agent_name: str, fn):
    """Wrap an agent node with error catching → marks state as errored."""
    def wrapper(state: GeoAgentState) -> dict:
        try:
            return fn(state)
        except Exception as exc:
            logger.exception("[%s] Unhandled error: %s", agent_name, exc)
            return {
                "status": "error",
                "error_message": f"{agent_name}: {exc}",
                "current_agent": agent_name,
                "messages": [
                    {
                        "agent": agent_name,
                        "type": "error",
                        "content": str(exc),
                    }
                ],
            }
    wrapper.__name__ = agent_name
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
#  Routing conditions
# ─────────────────────────────────────────────────────────────────────────────
def _route_after_planner(state: GeoAgentState) -> Literal["data_acquisition", "error"]:
    if state.get("status") == "error" or not state.get("plan"):
        return "error"
    return "data_acquisition"


def _route_after_acquisition(state: GeoAgentState) -> Literal["analyst", "error"]:
    if state.get("status") == "error":
        return "error"
    # Continue even if no files were downloaded (analyst will use demo data)
    return "analyst"


def _route_after_analyst(state: GeoAgentState) -> Literal["reporter", "error"]:
    if state.get("status") == "error":
        return "error"
    return "reporter"


def _error_node(state: GeoAgentState) -> dict:
    """Terminal error node — logs and returns."""
    logger.error("[Workflow] Pipeline ended with error: %s", state.get("error_message"))
    return {"status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
#  Graph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_workflow() -> StateGraph:
    """
    Construct and return the compiled LangGraph workflow.

    Graph structure:
      START
        ↓
      planner ──(error)──→ error_node → END
        ↓
      data_acquisition ──(error)──→ error_node
        ↓
      analyst ──(error)──→ error_node
        ↓
      reporter
        ↓
       END
    """
    planner   = PlannerAgent()
    acquirer  = DataAcquisitionAgent()
    analyst   = AnalystAgent()
    reporter  = ReporterAgent()

    graph = StateGraph(GeoAgentState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("planner",          _safe_node("planner",          planner))
    graph.add_node("data_acquisition", _safe_node("data_acquisition", acquirer))
    graph.add_node("analyst",          _safe_node("analyst",          analyst))
    graph.add_node("reporter",         _safe_node("reporter",         reporter))
    graph.add_node("error_node",       _error_node)

    # ── Add edges ────────────────────────────────────────────────────────
    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"data_acquisition": "data_acquisition", "error": "error_node"},
    )
    graph.add_conditional_edges(
        "data_acquisition",
        _route_after_acquisition,
        {"analyst": "analyst", "error": "error_node"},
    )
    graph.add_conditional_edges(
        "analyst",
        _route_after_analyst,
        {"reporter": "reporter", "error": "error_node"},
    )

    graph.add_edge("reporter",   END)
    graph.add_edge("error_node", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
#  Public runner
# ─────────────────────────────────────────────────────────────────────────────
def run_query(user_query: str, session_id: str | None = None) -> GeoAgentState:
    """
    Run the full multi-agent pipeline for a user query.

    Args:
        user_query:  Natural-language Earth Observation question.
        session_id:  Optional session identifier; auto-generated if not provided.

    Returns:
        Final GeoAgentState with all results populated.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    initial_state: GeoAgentState = {
        "user_query":        user_query,
        "session_id":        session_id,
        "plan":              None,
        "location":          None,
        "date_range":        None,
        "analysis_type":     None,
        "required_indices":  None,
        "satellites":        None,
        "available_scenes":  None,
        "selected_scene":    None,
        "downloaded_files":  None,
        "pre_scene_files":   None,
        "sar_available":     None,
        "acquisition_error": None,
        "generated_code":    None,
        "execution_result":  None,
        "computed_indices":  None,
        "output_files":      None,
        "map_overlays":      None,
        "code_iterations":   0,
        "analysis_error":    None,
        "report_markdown":   None,
        "map_html":          None,
        "chart_paths":       None,
        "messages":          [],
        "current_agent":     "planner",
        "status":            "running",
        "error_message":     None,
    }

    workflow = build_workflow()

    logger.info("[Workflow] Starting pipeline. Session=%s Query=%s", session_id, user_query)
    final_state = workflow.invoke(initial_state)
    logger.info("[Workflow] Pipeline complete. Status=%s", final_state.get("status"))

    return final_state
