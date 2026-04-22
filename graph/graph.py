# ============================================================
# graph/graph.py
#
# Compiles the LangGraph state machine and builds the initial state.
#
# Public API:
#   build_graph(session: dict) -> tuple[CompiledGraph, GraphState]
#
# Graph topology (fixed linear pipeline — no conditional edges):
#
#   [UNDERSTAND] ──► [ACT] ──► [RESPOND]
#        ▲                          │
#        └──────────────────────────┘  (next user message)
#
# The graph does NOT loop internally — main.py / app.py drive
# the turn loop. Each call to agent.invoke(state) is one full turn:
# user message in → assistant response out → updated state returned.
#
# build_graph() returns BOTH the compiled graph AND the initial state
# so the caller only ever calls this once at startup.
# ============================================================

from langgraph.graph import StateGraph

from graph.state import GraphState
from graph.nodes import understand_node, act_node, respond_node


def build_graph(session: dict) -> tuple:
    """
    Compile the LangGraph state machine and return it with a
    fresh initial state pre-loaded with the pipeline session.

    Parameters
    ----------
    session : dict
        The full session dict from pipeline/session.py.
        Stored in state["session"] and never mutated.

    Returns
    -------
    tuple[CompiledGraph, GraphState]
        agent        — call agent.invoke(state) each turn
        initial_state — pass as the first state to agent.invoke()
    """

    # ── Build graph ───────────────────────────────────────────
    g = StateGraph(GraphState)

    # Register nodes
    g.add_node("UNDERSTAND", understand_node)
    g.add_node("ACT",        act_node)
    g.add_node("RESPOND",    respond_node)

    # Wire edges (fixed linear flow — no branching)
    g.add_edge("UNDERSTAND", "ACT")
    g.add_edge("ACT",        "RESPOND")

    # Entry and finish points
    g.set_entry_point("RESPOND" if False else "UNDERSTAND")   # always UNDERSTAND
    g.set_finish_point("RESPOND")

    agent = g.compile()

    # ── Build initial state ───────────────────────────────────
    # ── Build initial state ───────────────────────────────────
    initial_state: GraphState = {
        # Pipeline data
        "session"      : session,

        # Conversation
        "messages"     : [],

        # Decision ledger
        "decisions"    : {},
        "undo_stack"   : [],
        "decision_log" : [],

        # Focus & Context
        "active_focus" : None,
        "focus_age"    : 0,        # <-- ADDED (I-3)

        # Intent routing
        "intent"       : None,
        "intent_params": {},
        "action_result": None,

        # Output & UI Modes
        "last_response": None,
        "overview_mode": None,
        "draft_mode"   : False,    # <-- ADDED (I-2)
        "guardrail_pending": False,
    }

    return agent, initial_state
