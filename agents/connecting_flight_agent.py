import json
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

import config
from state import TravelState
from utils.logger import trace_entry

console = Console()

_HUB_SYSTEM = """You are an airport routing expert.
Given an origin and destination, suggest the best connecting hub airports.
Consider geographic proximity, airline connectivity, and common travel routes.

Return ONLY a JSON array of objects with keys "iata" and "name".
Example: [{"iata": "GAU", "name": "Guwahati"}, {"iata": "CCU", "name": "Kolkata"}]

Rules:
- Suggest 3 hub airports maximum
- Do NOT include the origin or destination airports
- Prioritize airports that have good connections to BOTH origin and destination
- For remote/northeast India destinations, consider regional hubs like Guwahati, Bagdogra, Imphal
- For international routes, consider major transit hubs
- No markdown, no explanation — ONLY the JSON array."""


def _discover_hubs(origin: str, destination: str,
                   origin_display: str, dest_display: str) -> List[Dict]:
    """Use LLM to discover the best connecting hub airports for a route."""
    try:
        llm = ChatGroq(
            model=config.MODEL_NAME,
            temperature=0.0,
            api_key=config.GROQ_API_KEY,
            max_tokens=256,
        )
        chain = ChatPromptTemplate.from_messages([
            ("system", _HUB_SYSTEM),
            ("human",
             "Origin: {origin_display} (IATA: {origin})\n"
             "Destination: {dest_display} (IATA: {destination})\n\n"
             "Suggest the best connecting hub airports.")
        ]) | llm

        resp = chain.invoke({
            "origin": origin,
            "destination": destination,
            "origin_display": origin_display,
            "dest_display": dest_display,
        })

        text = resp.content.strip()
        # Parse JSON array
        for pat in [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```', r'(\[[\s\S]*\])']:
            m = re.search(pat, text)
            if m:
                try:
                    hubs = json.loads(m.group(1).strip())
                    if isinstance(hubs, list):
                        # Validate and filter
                        valid = []
                        for h in hubs:
                            if (isinstance(h, dict) and
                                    h.get("iata") and h.get("name") and
                                    len(h["iata"]) == 3 and
                                    h["iata"] not in (origin, destination)):
                                valid.append({"iata": h["iata"].upper(), "name": h["name"]})
                        return valid[:3]
                except Exception:
                    continue
        try:
            hubs = json.loads(text)
            if isinstance(hubs, list):
                valid = []
                for h in hubs:
                    if (isinstance(h, dict) and
                            h.get("iata") and h.get("name") and
                            len(h["iata"]) == 3 and
                            h["iata"] not in (origin, destination)):
                        valid.append({"iata": h["iata"].upper(), "name": h["name"]})
                return valid[:3]
        except Exception:
            pass

        return []
    except Exception as e:
        console.print(f"  [dim yellow]LLM hub discovery failed: {e}[/dim yellow]")
        return []


def _search_leg(dep_iata: str, arr_iata: str, date: str,
                pax: int, cabin: str) -> List[Dict]:
    """Search a single flight leg via SerpAPI. Returns a list of flight options."""
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
        "sort_by":       "2",            # cheapest first
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
                "stops_label":    ("Direct" if not item.get("layovers")
                                   else f"{len(item.get('layovers', []))} stop(s)"),
                "cabin_class":    first.get("travel_class", "Economy"),
            })
    return flights


def _search_via_hub(hub: Dict, origin: str, destination: str,
                    date: str, pax: int, cabin: str) -> List[Dict]:
    """Search both legs through a single hub and combine them."""
    hub_iata = hub["iata"]
    hub_name = hub["name"]

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_search_leg, origin, hub_iata, date, pax, cabin): "leg1",
            pool.submit(_search_leg, hub_iata, destination, date, pax, cabin): "leg2",
        }
        for fut in as_completed(futures, timeout=config.PROVIDER_TIMEOUT + 5):
            key = futures[fut]
            try:
                results[key] = fut.result(timeout=config.PROVIDER_TIMEOUT)
            except Exception:
                results[key] = []

    leg1_flights = results.get("leg1", [])
    leg2_flights = results.get("leg2", [])

    if not leg1_flights or not leg2_flights:
        return []

    combos = []
    for l1 in leg1_flights[:2]:
        for l2 in leg2_flights[:2]:
            combos.append({
                "type":           "connecting",
                "connection_hub": hub_name,
                "hub_iata":       hub_iata,
                "leg1":           l1,
                "leg2":           l2,
                "total_price":    l1["price_usd"] + l2["price_usd"],
                "total_duration_min": l1["duration_min"] + l2["duration_min"],
                "total_duration_h":   round((l1["duration_min"] + l2["duration_min"]) / 60, 1),
                "summary": (
                    f"{l1['airline']} {origin}→{hub_iata} "
                    f"+ {l2['airline']} {hub_iata}→{destination}"
                ),
            })
    return combos


def connecting_flight_search_node(state: TravelState) -> Dict:
    """Search for connecting flights through LLM-discovered hub airports."""
    t0 = time.time()
    p = state.get("parsed_data") or {}
    origin = state.get("origin_iata") or p.get("origin", "")
    destination = state.get("destination_iata") or p.get("destination", "")
    origin_display = state.get("origin_display") or p.get("origin", "")
    dest_display = state.get("destination_display") or p.get("destination", "")
    date = p.get("travel_date", datetime.now().strftime("%Y-%m-%d"))
    pax = p.get("num_passengers", 1)
    cabin = p.get("cabin_class", "economy")

    # Discover best hubs using LLM
    console.print(
        f"\n[bold blue]🔀 ConnectingFlightSearchNode[/bold blue] "
        f"— discovering smart hubs via LLM…"
    )
    hubs = _discover_hubs(origin, destination, origin_display, dest_display)

    if not hubs:
        console.print("  ❌ Could not discover any hub airports")
        return {
            "connecting_flight_results": [],
            "search_mode": "no_connecting",
            "execution_trace": [trace_entry(
                "ConnectingFlightSearchNode", t0,
                "hub discovery failed", "0 hubs found"
            )]
        }

    hub_names = ', '.join(h['name'] + ' (' + h['iata'] + ')' for h in hubs)
    console.print(
        f"  🧠 LLM suggested hubs: {hub_names}"
    )

    all_combos = []

    for hub in hubs:
        try:
            combos = _search_via_hub(hub, origin, destination, date, pax, cabin)
            if combos:
                console.print(
                    f"  ✅ via [yellow]{hub['name']} ({hub['iata']})[/yellow] "
                    f"→ {len(combos)} connecting option(s)"
                )
                all_combos.extend(combos)
            else:
                console.print(
                    f"  ❌ via [red]{hub['name']} ({hub['iata']})[/red] → no valid combinations"
                )
        except Exception as e:
            console.print(
                f"  ❌ via [red]{hub['name']}[/red] → error: {e}"
            )

    all_combos.sort(key=lambda x: x["total_price"])

    ms = round((time.time() - t0) * 1000)
    console.print(
        f"  📊 [bold]{len(all_combos)}[/bold] connecting options found | {ms}ms"
    )

    return {
        "connecting_flight_results": all_combos[:8],
        "search_mode": "connecting" if all_combos else "no_connecting",
        "execution_trace": [trace_entry(
            "ConnectingFlightSearchNode", t0,
            f"{len(hubs)} LLM-discovered hubs",
            f"{len(all_combos)} connecting options in {ms}ms"
        )]
    }
