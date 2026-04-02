from langgraph.graph import StateGraph, START, END
from state import TravelState
from agents import (
    parser_agent, airport_resolver_agent, validator_agent,
    parallel_flight_search_node, aggregator_agent, formatter_agent,
)


def _route(state: TravelState) -> str:
    return "format" if state.get("needs_clarification") else "search_flights"


def build_app():
    g = StateGraph(TravelState)

    g.add_node("parse",           parser_agent)
    g.add_node("resolve_airports", airport_resolver_agent)
    g.add_node("validate",        validator_agent)
    g.add_node("search_flights",  parallel_flight_search_node)
    g.add_node("aggregate",       aggregator_agent)
    g.add_node("format",          formatter_agent)

    g.add_edge(START,               "parse")
    g.add_edge("parse",             "resolve_airports")
    g.add_edge("resolve_airports",  "validate")

    g.add_conditional_edges("validate", _route, {
        "search_flights": "search_flights",
        "format":         "format",
    })

    g.add_edge("search_flights", "aggregate")
    g.add_edge("aggregate",      "format")
    g.add_edge("format",         END)

    return g.compile()