import json, time
from typing import Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from config import GROQ_API_KEY, MODEL_NAME
from state import TravelState
from utils.logger import trace_entry

console = Console()

_SYSTEM = """You are a friendly, efficient travel assistant.
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


def formatter_agent(state: TravelState) -> Dict:
    t0 = time.time()
    console.print("\n[bold blue]📝 FormatterAgent[/bold blue]")

    if state.get("needs_clarification"):
        msg = state.get("clarification_message","Could you provide more trip details?")
        console.print("  💬 Clarification needed")
        return {"final_response": msg,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "clarification", msg[:60])]}

    if not state.get("best_flight"):
        msg = ("Sorry, I couldn't find any flights right now. This could be due to "
               "an unavailable route or a temporary issue with the flight search API. "
               "Please try again, or check if the destination and date are correct.")
        console.print("  ❌ No flights found — returning error message")
        return {"final_response": msg,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "no results", msg[:60], "error")]}

    p        = state.get("parsed_data") or {}
    insights = state.get("price_insights") or {}
    ctx = {
        "route": {
            "origin":      state.get("origin_display")      or p.get("origin",""),
            "destination": state.get("destination_display") or p.get("destination",""),
            "origin_code": state.get("origin_iata",""),
            "dest_code":   state.get("destination_iata",""),
        },
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
            ("system", _SYSTEM),
            ("human", "{ctx}")
        ]) | llm

        resp  = chain.invoke({"ctx": json.dumps(ctx, indent=2)})
        final = resp.content.strip()
        console.print(f"  ✅ Response generated ({len(final)} chars)")
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
        if ctx["searches_failed"]:
            fb += f"\n⚠️  {ctx['searches_failed']} search perspective(s) unavailable."
        console.print(f"  ⚡ Fallback template (LLM error: {e})")
        return {"final_response": fb,
                "execution_trace": [trace_entry("FormatterAgent", t0,
                    "LLM fallback", "template", "fallback")]}