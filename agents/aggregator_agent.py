import time
from typing import Dict

from rich.console import Console

from state import TravelState
from utils.logger import trace_entry

console = Console()


def _key(f: Dict) -> str:
    fn = (f.get("flight_number") or "").replace(" ","")
    if fn:
        return f"{fn}_{(f.get('departure_time',''))[:10]}"
    return f"{f.get('airline','')}_{f.get('departure_time','')}_{f.get('price_usd','')}"


def aggregator_agent(state: TravelState) -> Dict:
    t0      = time.time()
    console.print("\n[bold blue]📊 AggregatorAgent[/bold blue]")

    results = state.get("flight_results", [])
    ok      = [r for r in results if r.get("success")]
    fail_res = [r for r in results if not r.get("success")]

    # ---------- Direct flights ----------
    if fail_res:
        names = [r["label"] for r in fail_res]
        console.print(f"  ⚠️  Partial failure: {names} — continuing with {len(ok)} result set(s)")

    seen, all_f = {}, []
    for res in ok:
        for f in res.get("flights",[]):
            k = _key(f)
            if k not in seen:
                seen[k] = True
                all_f.append(f)

    if all_f:
        cheapest = min(all_f, key=lambda x: x["price_usd"])
        fastest  = min(all_f, key=lambda x: x["duration_min"])

        min_p = min(f["price_usd"]    for f in all_f)
        max_p = max(f["price_usd"]    for f in all_f)
        min_d = min(f["duration_min"] for f in all_f)
        max_d = max(f["duration_min"] for f in all_f)
        pr    = max(max_p - min_p, 1)
        dr    = max(max_d - min_d, 1)

        all_f.sort(key=lambda f:
            0.6*((f["price_usd"]-min_p)/pr) + 0.4*((f["duration_min"]-min_d)/dr)
        )
        best = all_f[0]

        ranked = [
            {"rank": i+1, "airline": f["airline"], "flight_number": f["flight_number"],
             "price_usd": f["price_usd"], "duration_h": f["duration_h"],
             "stops": f["stops_label"], "departure": f["departure_time"],
             "arrival": f["arrival_time"], "cabin": f["cabin_class"],
             "airplane": f["airplane"], "legroom": f["legroom"]}
            for i, f in enumerate(all_f[:8])
        ]

        console.print(
            f"  ✅ {len(all_f)} unique direct flights | "
            f"cheapest=[green]${cheapest['price_usd']}[/green] {cheapest['airline']} | "
            f"fastest=[green]{fastest['duration_h']}h[/green] {fastest['airline']} | "
            f"best_overall=[green]{best['airline']} ${best['price_usd']}[/green]"
        )

        return {
            "best_flight":       best,
            "cheapest_flight":   cheapest,
            "fastest_flight":    fastest,
            "all_flights_ranked": ranked,
            "search_mode":       "direct",
            "execution_trace": [trace_entry("AggregatorAgent", t0,
                f"{len(all_f)} unique direct flights from {len(ok)} search(es)",
                f"best={best['airline']} ${best['price_usd']}")]
        }

    # ---------- Connecting flights ----------
    connecting = state.get("connecting_flight_results") or []
    if connecting:
        connecting.sort(key=lambda x: x["total_price"])
        best_conn = connecting[0]

        console.print(
            f"  ✅ {len(connecting)} connecting options | "
            f"cheapest=[green]${best_conn['total_price']}[/green] "
            f"via {best_conn['connection_hub']}"
        )

        return {
            "best_flight": None,
            "cheapest_flight": None,
            "fastest_flight": None,
            "all_flights_ranked": [],
            "search_mode": "connecting",
            "execution_trace": [trace_entry("AggregatorAgent", t0,
                f"{len(connecting)} connecting options",
                f"best via {best_conn['connection_hub']} ${best_conn['total_price']}")]
        }

    # ---------- Hybrid itinerary (flight + train/bus) ----------
    hybrid = state.get("hybrid_itinerary") or []
    if hybrid:
        best_h = hybrid[0]
        console.print(
            f"  ✅ {len(hybrid)} hybrid (flight+ground) options | "
            f"cheapest flight=[green]${best_h['flight_price_usd']}[/green] "
            f"+ ground={best_h.get('ground_price_estimate','?')}"
        )
        return {
            "best_flight": None,
            "cheapest_flight": None,
            "fastest_flight": None,
            "all_flights_ranked": [],
            "search_mode": "hybrid",
            "execution_trace": [trace_entry("AggregatorAgent", t0,
                f"{len(hybrid)} hybrid options",
                f"flight+ground via {best_h.get('airport_city','')}")]
        }

    # ---------- Alt transport ----------
    alt = state.get("alt_transport_results") or []
    if alt:
        console.print(f"  ✅ {len(alt)} alternative transport option(s)")
        return {
            "best_flight": None,
            "cheapest_flight": None,
            "fastest_flight": None,
            "all_flights_ranked": [],
            "search_mode": state.get("search_mode", "alt_transport"),
            "execution_trace": [trace_entry("AggregatorAgent", t0,
                f"{len(alt)} alt transport options",
                f"trains/buses found")]
        }

    # ---------- Nothing found ----------
    console.print(f"  ❌ No flights, connecting flights, or alt transport found")
    return {
        "best_flight": None, "cheapest_flight": None,
        "fastest_flight": None, "all_flights_ranked": [],
        "search_mode": "none_found",
        "execution_trace": [trace_entry("AggregatorAgent", t0,
            f"0/{len(results)} OK", "nothing found", "error")]
    }