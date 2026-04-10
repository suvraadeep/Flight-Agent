from langgraph.graph import StateGraph, START, END
from state import TravelState
from agents import (
    parser_agent, airport_resolver_agent, validator_agent,
    parallel_flight_search_node, connecting_flight_search_node,
    alt_transport_search_node, aggregator_agent, formatter_agent,
)


def _route_after_validate(state: TravelState) -> str:
    """Skip to formatter if clarification needed, else search flights."""
    return "format" if state.get("needs_clarification") else "search_flights"


def _route_after_flights(state: TravelState) -> str:
    """If direct flights found → aggregate; otherwise → try connecting flights."""
    results = state.get("flight_results", [])
    ok = [r for r in results if r.get("success")]
    has_flights = any(
        f for r in ok for f in r.get("flights", [])
    )
    return "aggregate" if has_flights else "search_connecting"


def _route_after_connecting(state: TravelState) -> str:
    """If connecting flights found → aggregate; otherwise → try alt transport."""
    connecting = state.get("connecting_flight_results") or []
    return "aggregate" if connecting else "search_alt_transport"


def build_app():
    g = StateGraph(TravelState)

    g.add_node("parse",              parser_agent)
    g.add_node("resolve_airports",   airport_resolver_agent)
    g.add_node("validate",          validator_agent)
    g.add_node("search_flights",    parallel_flight_search_node)
    g.add_node("search_connecting", connecting_flight_search_node)
    g.add_node("search_alt_transport", alt_transport_search_node)
    g.add_node("aggregate",         aggregator_agent)
    g.add_node("format",            formatter_agent)

    # Linear: parse → resolve → validate
    g.add_edge(START,               "parse")
    g.add_edge("parse",             "resolve_airports")
    g.add_edge("resolve_airports",  "validate")

    # validate → search_flights OR format (clarification)
    g.add_conditional_edges("validate", _route_after_validate, {
        "search_flights": "search_flights",
        "format":         "format",
    })

    # search_flights → aggregate (direct found) OR search_connecting (fallback)
    g.add_conditional_edges("search_flights", _route_after_flights, {
        "aggregate":        "aggregate",
        "search_connecting": "search_connecting",
    })

    # search_connecting → aggregate (connecting found) OR search_alt_transport
    g.add_conditional_edges("search_connecting", _route_after_connecting, {
        "aggregate":            "aggregate",
        "search_alt_transport": "search_alt_transport",
    })

    # alt transport always goes to aggregate
    g.add_edge("search_alt_transport", "aggregate")

    # aggregate → format → END
    g.add_edge("aggregate", "format")
    g.add_edge("format",    END)

    return g.compile()