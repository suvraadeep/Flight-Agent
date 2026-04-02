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

    if not ok:
        console.print(f"  ❌ All {len(results)} searches failed — no flights available")
        return {
            "best_flight": None, "cheapest_flight": None,
            "fastest_flight": None, "all_flights_ranked": [],
            "execution_trace": [trace_entry("AggregatorAgent", t0,
                f"0/{len(results)} OK", "all failed", "error")]
        }

    if fail_res:
        names = [r["label"] for r in fail_res]
        console.print(f"  ⚠️  Partial failure: {names} — continuing with {len(ok)} result set(s)")

    # Flatten + deduplicate
    seen, all_f = {}, []
    for res in ok:
        for f in res.get("flights",[]):
            k = _key(f)
            if k not in seen:
                seen[k] = True
                all_f.append(f)

    if not all_f:
        console.print("  Searches succeeded but returned 0 flights")
        return {
            "best_flight": None, "cheapest_flight": None,
            "fastest_flight": None, "all_flights_ranked": [],
            "execution_trace": [trace_entry("AggregatorAgent", t0,
                "searches OK", "0 flights", "error")]
        }

    cheapest = min(all_f, key=lambda x: x["price_usd"])
    fastest  = min(all_f, key=lambda x: x["duration_min"])

    # Composite scoring: 60% price + 40% duration (normalised)
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
        f"  ✅ {len(all_f)} unique flights | "
        f"cheapest=[green]${cheapest['price_usd']}[/green] {cheapest['airline']} | "
        f"fastest=[green]{fastest['duration_h']}h[/green] {fastest['airline']} | "
        f"best_overall=[green]{best['airline']} ${best['price_usd']}[/green]"
    )

    return {
        "best_flight":       best,
        "cheapest_flight":   cheapest,
        "fastest_flight":    fastest,
        "all_flights_ranked": ranked,
        "execution_trace": [trace_entry("AggregatorAgent", t0,
            f"{len(all_f)} unique flights from {len(ok)} search(es)",
            f"best={best['airline']} ${best['price_usd']}")]
    }