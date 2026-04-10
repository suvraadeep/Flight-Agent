import json, time
from typing import Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from config import GROQ_API_KEY, MODEL_NAME
from state import TravelState
from utils.logger import trace_entry

console = Console()

_SYSTEM_DIRECT = """You are a friendly, efficient travel assistant.
Format the search results into a clear, engaging response under 150 words.
Structure:
1. Quick summary line (X flights found, date, route)
2. Best overall deal highlighted (airline, price, duration, stops)
3. Cheapest option if different from best
4. Fastest option if different from best
5. One-line alternatives (max 2 more)
6. If any searches failed, briefly note it
7. Price context from insights if available (e.g. "prices are currently low")

Use natural conversational tone. No markdown headers. Use ✈️ 💰 ⚡ emoji sparingly."""

_SYSTEM_CONNECTING = """You are a friendly, efficient travel assistant.
The user asked for flights but no direct flights were found. However, connecting flights
through hub airports ARE available. Format these clearly under 200 words.
Structure:
1. Note that no direct flights exist for this route
2. Present connecting options clearly showing both legs:
   - Leg 1: airline, origin → hub, price, duration
   - Leg 2: airline, hub → destination, price, duration
   - Total price and estimated total travel time
3. Show top 3 options max, from cheapest to most expensive
4. Brief note about connection city

Use natural conversational tone. Use 🔀 ✈️ 💰 emoji sparingly."""

_SYSTEM_HYBRID = """You are a friendly, efficient travel assistant.
The user asked for flights to a remote destination. No direct or connecting flights exist,
but a HYBRID option is available: fly to the nearest airport, then take a train or bus.
Format these clearly under 250 words.
Structure:
1. Explain that no flights go directly to the destination
2. Present the hybrid travel plan clearly:
   - ✈️ Flight leg: airline, route, price, duration
   - 🚆/🚌 Ground leg: type (train/bus), operator, route, estimated price, duration
3. Show the total estimated cost (flight + ground transport)
4. Show top 2-3 options max
5. Any helpful tips about the connection

Use natural conversational tone. Use ✈️ 🚆 🚌 💰 emoji sparingly."""

_SYSTEM_ALT_TRANSPORT = """You are a friendly, efficient travel assistant.
The user asked for flights but no flights (direct or connecting) were found for this route.
However, alternative transport (trains/buses) IS available. Format clearly under 200 words.
Structure:
1. Note that no flights are available for this route
2. Suggest the available train/bus options:
   - Type (train/bus), operator, estimated duration
   - Price estimate if available
   - Frequency if available
3. Any helpful travel tips

Use natural conversational tone. Use 🚆 🚌 emoji sparingly."""


def formatter_agent(state: TravelState) -> Dict:
    t0 = time.time()
    console.print("\n[bold blue]📝 FormatterAgent[/bold blue]")

    # --- Clarification ---
    if state.get("needs_clarification"):
        msg = state.get("clarification_message","Could you provide more trip details?")
        console.print("  💬 Clarification needed")
        return {"final_response": msg,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "clarification", msg[:60])]}

    search_mode = state.get("search_mode", "direct")
    p = state.get("parsed_data") or {}
    route_info = {
        "origin":      state.get("origin_display")      or p.get("origin",""),
        "destination": state.get("destination_display") or p.get("destination",""),
        "origin_code": state.get("origin_iata",""),
        "dest_code":   state.get("destination_iata",""),
    }

    # --- Direct flights ---
    if search_mode == "direct" and state.get("best_flight"):
        return _format_direct(state, t0, p, route_info)

    # --- Connecting flights ---
    connecting = state.get("connecting_flight_results") or []
    if search_mode == "connecting" and connecting:
        return _format_connecting(state, t0, p, route_info, connecting)

    # --- Hybrid (flight + train/bus) ---
    hybrid = state.get("hybrid_itinerary") or []
    if search_mode == "hybrid" and hybrid:
        return _format_hybrid(state, t0, p, route_info, hybrid)

    # --- Alt transport ---
    alt = state.get("alt_transport_results") or []
    if search_mode in ("alt_transport", "none_found") and alt:
        return _format_alt_transport(state, t0, p, route_info, alt)

    # --- Nothing found at all ---
    msg = (
        f"Sorry, I couldn't find any travel options from "
        f"{route_info['origin']} to {route_info['destination']}. "
        f"I searched for direct flights, connecting flights, "
        f"hybrid flight+train/bus, and direct trains/buses — "
        f"but nothing is available right now. "
        f"Please double-check the destination name, or try a different date."
    )
    console.print("  ❌ No results at all — returning error message")
    return {"final_response": msg,
            "execution_trace": [trace_entry("FormatterAgent", t0,
                "no results", msg[:60], "error")]}


def _format_direct(state, t0, p, route_info):
    insights = state.get("price_insights") or {}
    ctx = {
        "route":             route_info,
        "travel_date":       p.get("travel_date",""),
        "passengers":        p.get("num_passengers",1),
        "cabin_class":       p.get("cabin_class","economy"),
        "total_flights":     len(state.get("all_flights_ranked") or []),
        "best_overall":      state["best_flight"],
        "cheapest":          state.get("cheapest_flight"),
        "fastest":           state.get("fastest_flight"),
        "top_options":       (state.get("all_flights_ranked") or [])[:5],
        "searches_ok":       state.get("total_searches_succeeded",0),
        "searches_failed":   (state.get("total_searches_attempted",0) -
                              state.get("total_searches_succeeded",0)),
        "price_level":       insights.get("price_level",""),
        "typical_range":     insights.get("typical_range",""),
        "lowest_price_seen": insights.get("lowest_price",""),
    }

    try:
        llm   = ChatGroq(model=MODEL_NAME, temperature=0.3,
                         api_key=GROQ_API_KEY, max_tokens=1024)
        chain = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_DIRECT),
            ("human", "{ctx}")
        ]) | llm

        resp  = chain.invoke({"ctx": json.dumps(ctx, indent=2)})
        final = resp.content.strip()
        console.print(f"  ✅ Direct flight response ({len(final)} chars)")
        return {"final_response": final,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    f"{ctx['total_flights']} flights", f"{len(final)} chars")]}
    except Exception as e:
        b = state["best_flight"]
        c = state.get("cheapest_flight") or b
        f = state.get("fastest_flight")  or b
        fb = (f"✈️  Found {ctx['total_flights']} flights "
              f"from {ctx['route']['origin']} to {ctx['route']['destination']} "
              f"on {ctx['travel_date']}.\n\n"
              f"🏆 Best overall:  {b['airline']} "
              f"${b['price_usd']} | {b['duration_h']}h | {b['stops_label']} | "
              f"Departs {b['departure_time']}\n"
              f"💰 Cheapest:      {c['airline']} ${c['price_usd']}\n"
              f"⚡ Fastest:       {f['airline']} {f['duration_h']}h\n")
        console.print(f"  ⚡ Fallback template (LLM error: {e})")
        return {"final_response": fb,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "LLM fallback", "template", "fallback")]}


def _format_connecting(state, t0, p, route_info, connecting):
    ctx = {
        "route":         route_info,
        "travel_date":   p.get("travel_date",""),
        "passengers":    p.get("num_passengers",1),
        "cabin_class":   p.get("cabin_class","economy"),
        "connecting_options": [
            {
                "connection_hub": c["connection_hub"],
                "leg1": {
                    "airline": c["leg1"]["airline"],
                    "route":   f"{c['leg1']['departure_code']}→{c['leg1']['arrival_code']}",
                    "price":   c["leg1"]["price_usd"],
                    "duration": c["leg1"]["duration_h"],
                    "departure": c["leg1"]["departure_time"],
                },
                "leg2": {
                    "airline": c["leg2"]["airline"],
                    "route":   f"{c['leg2']['departure_code']}→{c['leg2']['arrival_code']}",
                    "price":   c["leg2"]["price_usd"],
                    "duration": c["leg2"]["duration_h"],
                    "departure": c["leg2"]["departure_time"],
                },
                "total_price": c["total_price"],
                "total_duration_h": c["total_duration_h"],
            }
            for c in connecting[:5]
        ],
    }

    try:
        llm   = ChatGroq(model=MODEL_NAME, temperature=0.3,
                         api_key=GROQ_API_KEY, max_tokens=1024)
        chain = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_CONNECTING),
            ("human", "{ctx}")
        ]) | llm

        resp  = chain.invoke({"ctx": json.dumps(ctx, indent=2)})
        final = resp.content.strip()
        console.print(f"  ✅ Connecting flight response ({len(final)} chars)")
        return {"final_response": final,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    f"{len(connecting)} connecting", f"{len(final)} chars")]}
    except Exception as e:
        lines = [f"🔀 No direct flights from {route_info['origin']} to "
                 f"{route_info['destination']}. Here are connecting options:\n"]
        for i, c in enumerate(connecting[:3], 1):
            lines.append(
                f"\nOption {i} — via {c['connection_hub']}:\n"
                f"  ✈️  Leg 1: {c['leg1']['airline']} "
                f"{c['leg1']['departure_code']}→{c['leg1']['arrival_code']} "
                f"${c['leg1']['price_usd']} ({c['leg1']['duration_h']}h)\n"
                f"  ✈️  Leg 2: {c['leg2']['airline']} "
                f"{c['leg2']['departure_code']}→{c['leg2']['arrival_code']} "
                f"${c['leg2']['price_usd']} ({c['leg2']['duration_h']}h)\n"
                f"  💰 Total: ${c['total_price']} | ⏱ {c['total_duration_h']}h\n"
            )
        fb = "".join(lines)
        console.print(f"  ⚡ Fallback template (LLM error: {e})")
        return {"final_response": fb,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "connecting fallback", "template", "fallback")]}


def _format_hybrid(state, t0, p, route_info, hybrid):
    ctx = {
        "route":       route_info,
        "travel_date": p.get("travel_date",""),
        "passengers":  p.get("num_passengers",1),
        "cabin_class": p.get("cabin_class","economy"),
        "hybrid_options": [
            {
                "flight": {
                    "airline": h["flight"]["airline"],
                    "route": f"{h['flight']['departure_code']}→{h['flight']['arrival_code']}",
                    "price_usd": h["flight"]["price_usd"],
                    "duration_h": h["flight"]["duration_h"],
                    "departure": h["flight"]["departure_time"],
                },
                "ground": {
                    "type": h["ground_type"],
                    "operator": h["ground_operator"],
                    "price_estimate": h["ground_price_estimate"],
                    "duration": h["ground_duration"],
                },
                "airport_city": h["airport_city"],
                "final_destination": h["final_destination"],
                "flight_price_usd": h["flight_price_usd"],
            }
            for h in hybrid[:4]
        ],
    }

    try:
        llm   = ChatGroq(model=MODEL_NAME, temperature=0.3,
                         api_key=GROQ_API_KEY, max_tokens=1024)
        chain = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_HYBRID),
            ("human", "{ctx}")
        ]) | llm

        resp  = chain.invoke({"ctx": json.dumps(ctx, indent=2)})
        final = resp.content.strip()
        console.print(f"  ✅ Hybrid response ({len(final)} chars)")
        return {"final_response": final,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    f"{len(hybrid)} hybrid", f"{len(final)} chars")]}
    except Exception as e:
        lines = [f"🔀 No direct flights to {route_info['destination']}. "
                 f"Here's a hybrid travel plan:\n"]
        for i, h in enumerate(hybrid[:3], 1):
            lines.append(
                f"\nOption {i} — via {h['airport_city']}:\n"
                f"  ✈️  Flight: {h['flight']['airline']} "
                f"{h['flight']['departure_code']}→{h['flight']['arrival_code']} "
                f"${h['flight']['price_usd']} ({h['flight']['duration_h']}h)\n"
                f"  🚆 Ground: {h['ground_type']} by {h['ground_operator']} "
                f"{h['airport_city']}→{h['final_destination']} "
                f"{h['ground_price_estimate']} ({h['ground_duration']})\n"
            )
        fb = "".join(lines)
        console.print(f"  ⚡ Fallback template (LLM error: {e})")
        return {"final_response": fb,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "hybrid fallback", "template", "fallback")]}


def _format_alt_transport(state, t0, p, route_info, alt):
    ctx = {
        "route":       route_info,
        "travel_date": p.get("travel_date",""),
        "passengers":  p.get("num_passengers",1),
        "options":     alt[:6],
    }

    try:
        llm   = ChatGroq(model=MODEL_NAME, temperature=0.3,
                         api_key=GROQ_API_KEY, max_tokens=1024)
        chain = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_ALT_TRANSPORT),
            ("human", "{ctx}")
        ]) | llm

        resp  = chain.invoke({"ctx": json.dumps(ctx, indent=2)})
        final = resp.content.strip()
        console.print(f"  ✅ Alt transport response ({len(final)} chars)")
        return {"final_response": final,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    f"{len(alt)} alt options", f"{len(final)} chars")]}
    except Exception as e:
        lines = [f"🚫 No flights available from {route_info['origin']} to "
                 f"{route_info['destination']}. Here are alternative options:\n"]
        for opt in alt[:4]:
            emoji = "🚆" if opt.get("type") == "train" else "🚌"
            lines.append(
                f"\n{emoji} {opt.get('operator', 'Unknown')} "
                f"({opt.get('type','?')}) — "
                f"{opt.get('duration_estimate','?')} — "
                f"{opt.get('price_estimate','?')}"
            )
        fb = "".join(lines)
        console.print(f"  ⚡ Fallback template (LLM error: {e})")
        return {"final_response": fb,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "alt transport fallback", "template", "fallback")]}