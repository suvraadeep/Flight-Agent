import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FTE
from typing import Dict, List

from serpapi import GoogleSearch
from rich.console import Console

import config
from state import TravelState
from utils.logger import trace_entry

console = Console()


def _parse_flights(data: Dict, sort_label: str) -> List[Dict]:
    flights = []
    for cat in ["best_flights", "other_flights"]:
        for item in data.get(cat, [])[:5]:
            segs = item.get("flights", [])
            if not segs or not item.get("price"):
                continue
            first, last = segs[0], segs[-1]
            dep = first.get("departure_airport", {})
            arr = last.get("arrival_airport",   {})
            lays = item.get("layovers", [])
            flights.append({
                "airline":          first.get("airline","Unknown"),
                "airline_logo":     first.get("airline_logo",""),
                "flight_number":    first.get("flight_number",""),
                "price_usd":        item.get("price", 0),
                "duration_min":     item.get("total_duration", 0),
                "duration_h":       round(item.get("total_duration",0)/60, 1),
                "departure_time":   dep.get("time",""),
                "departure_code":   dep.get("id",""),
                "arrival_time":     arr.get("time",""),
                "arrival_code":     arr.get("id",""),
                "stops":            len(lays),
                "stops_label":      "Direct" if not lays else f"{len(lays)} stop(s)",
                "layover_info":     [l.get("name","") for l in lays],
                "cabin_class":      first.get("travel_class","Economy"),
                "airplane":         first.get("airplane",""),
                "legroom":          first.get("legroom",""),
                "sort_perspective": sort_label,
            })
    return flights


def _fetch_one(sort_label: str, sort_by: str, dep_iata: str, arr_iata: str,
               date: str, pax: int, cabin: str) -> Dict:
    cabin_map = {"economy":"1","premium_economy":"2","business":"3","first":"4"}

    if sort_label in config.FORCE_FAIL_SEARCHES:
        raise RuntimeError(f"[test] forced failure for '{sort_label}' search")

    params = {
        "engine":        "google_flights",
        "departure_id":  dep_iata,
        "arrival_id":    arr_iata,
        "outbound_date": date,
        "currency":      config.CURRENCY,
        "hl":            config.LANGUAGE,
        "adults":        str(pax),
        "type":          "2",
        "sort_by":       sort_by,
        "travel_class":  cabin_map.get(cabin.lower(), "1"),
        "api_key":       config.SERPAPI_KEY,
    }

    data = GoogleSearch(params).get_dict()
    if "error" in data:
        raise RuntimeError(f"SerpAPI error: {data['error']}")
    if data.get("search_metadata",{}).get("status","") != "Success":
        raise RuntimeError(f"SerpAPI status: {data.get('search_metadata',{}).get('status','')}")

    return {
        "label":          sort_label,
        "flights":        _parse_flights(data, sort_label),
        "price_insights": data.get("price_insights", {}),
    }


def _run_one(cfg: Dict, state: TravelState) -> Dict:
    p    = state["parsed_data"] or {}
    t0   = time.time()
    label = cfg["label"]
    try:
        result = _fetch_one(
            sort_label = label,
            sort_by    = cfg["sort_by"],
            dep_iata   = state.get("origin_iata")      or p.get("origin","JFK"),
            arr_iata   = state.get("destination_iata") or p.get("destination",""),
            date       = p.get("travel_date", datetime.now().strftime("%Y-%m-%d")),
            pax        = p.get("num_passengers", 1),
            cabin      = p.get("cabin_class", "economy"),
        )
        total = len(result["flights"])
        ms = round((time.time()-t0)*1000)
        console.print(
            f"  ✅ {cfg['emoji']} [green]{label:10}[/green] "
            f"→ {total} flights found  ({ms}ms)"
        )
        return {**result, "success": True, "error": None, "total_found": total}
    except Exception as e:
        ms = round((time.time()-t0)*1000)
        console.print(
            f"  ❌ {cfg['emoji']} [red]{label:10}[/red] "
            f"→ FAILED: {e}  ({ms}ms)"
        )
        return {
            "label": label, "flights": [], "price_insights": {},
            "total_found": 0, "success": False, "error": str(e),
        }


def parallel_flight_search_node(state: TravelState) -> Dict:
    t0      = time.time()
    configs = config.SEARCH_CONFIGS[:1] if config.LITE_MODE else config.SEARCH_CONFIGS

    console.print(
        f"\n[bold blue]🚀 ParallelFlightSearchNode[/bold blue] "
        f"— {len(configs)} search(es) "
        f"{'[dim](lite mode)[/dim]' if config.LITE_MODE else 'simultaneously'}"
    )

    results = []

    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = {pool.submit(_run_one, cfg, state): cfg for cfg in configs}
        for fut in as_completed(futures, timeout=config.PROVIDER_TIMEOUT + 5):
            cfg = futures[fut]
            try:
                results.append(fut.result(timeout=config.PROVIDER_TIMEOUT))
            except FTE:
                console.print(f"  ⏰ [yellow]{cfg['label']:10}[/yellow] → TIMEOUT")
                results.append({
                    "label": cfg["label"], "flights": [], "price_insights": {},
                    "total_found": 0, "success": False, "error": "timeout"
                })
            except Exception as exc:
                results.append({
                    "label": cfg["label"], "flights": [], "price_insights": {},
                    "total_found": 0, "success": False, "error": str(exc)
                })

    ok = sum(1 for r in results if r.get("success"))
    ms = round((time.time()-t0)*1000)

    console.print(
        f"  📊 [bold]{ok}/{len(configs)}[/bold] succeeded | "
        f"parallel={ms}ms"
    )

    insights = next(
        (r["price_insights"] for r in results if r.get("success") and r.get("price_insights")),
        {}
    )

    return {
        "flight_results":           results,
        "total_searches_attempted": len(configs),
        "total_searches_succeeded": ok,
        "price_insights":           insights,
        "execution_trace": [trace_entry(
            "ParallelFlightSearchNode", t0,
            f"{len(configs)} searches (sort=best,cheapest,fastest)",
            f"{ok}/{len(configs)} succeeded in {ms}ms"
        )]
    }