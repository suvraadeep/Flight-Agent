import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

import config
from state import TravelState
from utils.logger import trace_entry

console = Console()

_NEAREST_AIRPORT_SYSTEM = """You are an airport and geography expert.
Given a destination city/region that has NO airport or only a very small airport,
tell me the nearest major airport that has good flight connectivity.

Return ONLY a JSON object with keys:
  - "airport_iata": 3-letter IATA code
  - "airport_city": city name of the airport
  - "ground_distance_km": approximate distance from airport to destination
  - "ground_transport_note": brief note on how to get from airport to destination

No markdown, no explanation — ONLY the JSON object."""

_EXTRACT_SYSTEM = """You are a transport information extractor.
Given search snippets about trains and buses between two cities, extract available options.
Return ONLY a JSON array of objects. Each object should have:
  - "type": "train" or "bus"
  - "operator": operator/service name (e.g. "Indian Railways", "RedBus", "KSRTC")
  - "duration_estimate": estimated travel time as string (e.g. "24h", "18h 30m")
  - "price_estimate": approximate price as string (e.g. "₹800-1500", "₹2000")
  - "frequency": how often it runs (e.g. "daily", "3 times/week") or null
  - "notes": any other useful info (e.g. "Rajdhani Express", "AC Sleeper") or null

Return an empty array [] if no relevant options are found.
No markdown, no explanation — ONLY the JSON array."""


def _find_nearest_airport(destination: str, dest_display: str) -> Optional[Dict]:
    """Use LLM to find the nearest major airport to the destination."""
    try:
        llm = ChatGroq(
            model=config.MODEL_NAME,
            temperature=0.0,
            api_key=config.GROQ_API_KEY,
            max_tokens=256,
        )
        chain = ChatPromptTemplate.from_messages([
            ("system", _NEAREST_AIRPORT_SYSTEM),
            ("human",
             "Destination: {destination} ({dest_display})\n"
             "What is the nearest major airport with good flight connectivity?")
        ]) | llm

        resp = chain.invoke({
            "destination": destination,
            "dest_display": dest_display,
        })

        text = resp.content.strip()
        for pat in [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```', r'(\{[\s\S]*\})']:
            m = re.search(pat, text)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except Exception:
                    continue
        try:
            return json.loads(text)
        except Exception:
            return None
    except Exception as e:
        console.print(f"  [dim yellow]LLM nearest airport lookup failed: {e}[/dim yellow]")
        return None


def _search_flights(dep_iata: str, arr_iata: str, date: str,
                    pax: int, cabin: str) -> List[Dict]:
    """Search flights via SerpAPI."""
    cabin_map = {"economy": "1", "premium_economy": "2",
                 "business": "3", "first": "4"}
    params = {
        "engine":        "google_flights",
        "departure_id":  dep_iata,
        "arrival_id":    arr_iata,
        "outbound_date": date,
        "currency":      config.CURRENCY,
        "hl":            config.LANGUAGE,
        "adults":        str(pax),
        "type":          "2",
        "sort_by":       "2",
        "travel_class":  cabin_map.get(cabin.lower(), "1"),
        "api_key":       config.SERPAPI_KEY,
    }

    data = GoogleSearch(params).get_dict()
    if "error" in data or data.get("search_metadata", {}).get("status") != "Success":
        return []

    flights = []
    for cat in ["best_flights", "other_flights"]:
        for item in data.get(cat, [])[:3]:
            segs = item.get("flights", [])
            if not segs or not item.get("price"):
                continue
            first, last = segs[0], segs[-1]
            dep = first.get("departure_airport", {})
            arr = last.get("arrival_airport", {})
            flights.append({
                "airline":        first.get("airline", "Unknown"),
                "flight_number":  first.get("flight_number", ""),
                "price_usd":      item.get("price", 0),
                "duration_min":   item.get("total_duration", 0),
                "duration_h":     round(item.get("total_duration", 0) / 60, 1),
                "departure_time": dep.get("time", ""),
                "departure_code": dep.get("id", ""),
                "arrival_time":   arr.get("time", ""),
                "arrival_code":   arr.get("id", ""),
                "stops":          len(item.get("layovers", [])),
                "cabin_class":    first.get("travel_class", "Economy"),
            })
    return flights


def _search_google(query: str) -> List[Dict]:
    """Use SerpAPI Google Search for train/bus information."""
    try:
        params = {
            "engine": "google",
            "q":      query,
            "hl":     config.LANGUAGE,
            "num":    "8",
            "api_key": config.SERPAPI_KEY,
        }
        data = GoogleSearch(params).get_dict()
        results = data.get("organic_results", [])
        return [
            {"title": r.get("title", ""), "snippet": r.get("snippet", "")}
            for r in results[:6]
        ]
    except Exception:
        return []


def _extract_transport(snippets: List[Dict],
                       origin: str, destination: str) -> List[Dict]:
    """Use LLM to extract structured transport options from search snippets."""
    if not snippets:
        return []

    snippets_text = "\n\n".join(
        f"Title: {s['title']}\nSnippet: {s['snippet']}"
        for s in snippets
    )

    try:
        llm = ChatGroq(
            model=config.MODEL_NAME,
            temperature=0.0,
            api_key=config.GROQ_API_KEY,
            max_tokens=1024,
        )
        chain = ChatPromptTemplate.from_messages([
            ("system", _EXTRACT_SYSTEM),
            ("human",
             "Route: {origin} to {destination}\n\n"
             "Search results:\n{snippets}")
        ]) | llm

        resp = chain.invoke({
            "origin": origin,
            "destination": destination,
            "snippets": snippets_text,
        })

        text = resp.content.strip()
        for pat in [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```', r'(\[[\s\S]*\])']:
            m = re.search(pat, text)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except Exception:
                    continue
        try:
            return json.loads(text)
        except Exception:
            return []
    except Exception:
        return []


def alt_transport_search_node(state: TravelState) -> Dict:
    """
    Hybrid search: find flight to nearest airport + train/bus from there
    to the final destination. Also searches direct train/bus as fallback.
    """
    t0 = time.time()
    p = state.get("parsed_data") or {}
    origin_iata = state.get("origin_iata") or p.get("origin", "")
    dest_iata = state.get("destination_iata") or p.get("destination", "")
    origin_display = state.get("origin_display") or p.get("origin", "")
    dest_display = state.get("destination_display") or p.get("destination", "")
    date = p.get("travel_date", datetime.now().strftime("%Y-%m-%d"))
    pax = p.get("num_passengers", 1)
    cabin = p.get("cabin_class", "economy")

    # Clean display names
    origin_clean = re.sub(r'\s*\([A-Z]{3}\)', '', origin_display).strip()
    dest_clean = re.sub(r'\s*\([A-Z]{3}\)', '', dest_display).strip()

    console.print(
        f"\n[bold blue]🚆 AltTransportSearchNode[/bold blue] "
        f"— hybrid search: {origin_clean} → {dest_clean}"
    )

    hybrid_results = []
    alt_results = []

    # --- Step 1: Find nearest major airport to destination ---
    console.print("  🧠 Finding nearest airport via LLM…")
    nearest = _find_nearest_airport(dest_iata, dest_clean)

    if nearest and nearest.get("airport_iata"):
        airport_iata = nearest["airport_iata"].upper()
        airport_city = nearest.get("airport_city", airport_iata)
        distance_km = nearest.get("ground_distance_km", "?")
        transport_note = nearest.get("ground_transport_note", "")

        console.print(
            f"  ✅ Nearest airport: [yellow]{airport_city} ({airport_iata})[/yellow] "
            f"— ~{distance_km}km from {dest_clean}"
        )

        # Skip if nearest airport is same as destination (already tried)
        if airport_iata != dest_iata:
            # --- Step 2: Search flights to the nearest airport ---
            console.print(f"  ✈️  Searching flights {origin_iata} → {airport_iata}…")
            flights = _search_flights(origin_iata, airport_iata, date, pax, cabin)

            if flights:
                console.print(f"  ✅ {len(flights)} flight(s) to {airport_city}")

                # --- Step 3: Search ground transport from airport city to destination ---
                console.print(
                    f"  🔍 Searching trains/buses {airport_city} → {dest_clean}…"
                )
                train_snippets = _search_google(
                    f"train from {airport_city} to {dest_clean} schedule price"
                )
                bus_snippets = _search_google(
                    f"bus from {airport_city} to {dest_clean} schedule price"
                )
                ground_options = _extract_transport(
                    train_snippets + bus_snippets, airport_city, dest_clean
                )

                if ground_options:
                    console.print(
                        f"  ✅ {len(ground_options)} ground transport option(s)"
                    )
                else:
                    console.print("  ⚠️  No ground transport found, using LLM note")
                    ground_options = [{
                        "type": "ground",
                        "operator": "Local transport",
                        "duration_estimate": "varies",
                        "price_estimate": "varies",
                        "frequency": None,
                        "notes": transport_note or f"~{distance_km}km by road",
                    }]

                # Build hybrid itineraries
                for flight in flights[:3]:
                    for ground in ground_options[:2]:
                        hybrid_results.append({
                            "type": "hybrid",
                            "flight": flight,
                            "ground": ground,
                            "airport_city": airport_city,
                            "airport_iata": airport_iata,
                            "final_destination": dest_clean,
                            "flight_price_usd": flight["price_usd"],
                            "ground_price_estimate": ground.get("price_estimate", "varies"),
                            "flight_duration_h": flight["duration_h"],
                            "ground_duration": ground.get("duration_estimate", "varies"),
                            "ground_type": ground.get("type", "ground"),
                            "ground_operator": ground.get("operator", ""),
                            "summary": (
                                f"✈️ {flight['airline']} {origin_iata}→{airport_iata} "
                                f"(${flight['price_usd']}) + "
                                f"🚆 {ground.get('type','')} {airport_city}→{dest_clean} "
                                f"({ground.get('price_estimate','?')})"
                            ),
                        })
            else:
                console.print(f"  ❌ No flights to {airport_city} either")
    else:
        console.print("  ⚠️  Could not determine nearest airport")

    # --- Step 4: Also search direct trains/buses as pure fallback ---
    console.print(f"  🔍 Also checking direct trains/buses {origin_clean} → {dest_clean}…")
    direct_train = _search_google(
        f"train from {origin_clean} to {dest_clean} schedule price booking"
    )
    direct_bus = _search_google(
        f"bus from {origin_clean} to {dest_clean} schedule price booking"
    )
    alt_results = _extract_transport(
        direct_train + direct_bus, origin_clean, dest_clean
    )

    ms = round((time.time() - t0) * 1000)

    if hybrid_results:
        console.print(
            f"  ✅ [bold]{len(hybrid_results)}[/bold] hybrid (flight+ground) options | {ms}ms"
        )
        for h in hybrid_results[:2]:
            console.print(f"    {h['summary']}")

    if alt_results:
        console.print(
            f"  ✅ [bold]{len(alt_results)}[/bold] direct train/bus options | {ms}ms"
        )

    if not hybrid_results and not alt_results:
        console.print(f"  ❌ No transport options found at all | {ms}ms")

    return {
        "hybrid_itinerary": hybrid_results[:6],
        "alt_transport_results": alt_results,
        "search_mode": (
            "hybrid" if hybrid_results
            else "alt_transport" if alt_results
            else "none_found"
        ),
        "execution_trace": [trace_entry(
            "AltTransportSearchNode", t0,
            f"hybrid+alt {origin_clean}→{dest_clean}",
            f"{len(hybrid_results)} hybrid + {len(alt_results)} alt in {ms}ms"
        )]
    }
