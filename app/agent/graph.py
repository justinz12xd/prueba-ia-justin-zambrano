"""Construcción del grafo LangGraph del agente conversacional.

Flujo (ver diagrama del enunciado):

    INICIO -> classify_intent -> [greeting/farewell] -----------------\
                               -> get_customer_info                    v
                                     -> handle_account_query      -> check_escalation -> generate_response -> FIN
                                     -> handle_technical_support -/
                                     -> handle_general_info     -/
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (check_escalation, classify_intent, generate_response,
                              get_customer_info, handle_account_query, handle_general_info,
                              handle_technical_support)
from app.agent.state import AgentState


def _route_after_intent(state: AgentState) -> str:
    if state.get("intent") in ("greeting", "farewell"):
        return "generate_response"
    return "get_customer_info"


def _route_by_intent(state: AgentState) -> str:
    return {
        "account_query": "handle_account_query",
        "technical_support": "handle_technical_support",
        "general_info": "handle_general_info",
    }.get(state.get("intent"), "handle_general_info")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("get_customer_info", get_customer_info)
    graph.add_node("handle_account_query", handle_account_query)
    graph.add_node("handle_technical_support", handle_technical_support)
    graph.add_node("handle_general_info", handle_general_info)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent", _route_after_intent,
        {"generate_response": "generate_response", "get_customer_info": "get_customer_info"},
    )
    graph.add_conditional_edges(
        "get_customer_info", _route_by_intent,
        {
            "handle_account_query": "handle_account_query",
            "handle_technical_support": "handle_technical_support",
            "handle_general_info": "handle_general_info",
        },
    )
    graph.add_edge("handle_account_query", "check_escalation")
    graph.add_edge("handle_technical_support", "check_escalation")
    graph.add_edge("handle_general_info", "check_escalation")
    graph.add_edge("check_escalation", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


@lru_cache
def get_compiled_graph():
    return build_graph()
