import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from rich.console import Console

from state import TravelState
from utils.logger import trace_entry
from utils.iata   import lookup_iata

console = Console()


def airport_resolver_agent(state: TravelState) -> Dict:
    t0 = time.time()
    console.print("\n[bold blue]🗺️  AirportResolverAgent[/bold blue]")

    p  = state.get("parsed_data") or {}
    origin_city = p.get("origin", "New York")
    dest_city   = p.get("destination", "")

    if not dest_city:
        return {
            "origin_iata": origin_city, "destination_iata": "",
            "origin_display": origin_city, "destination_display": "",
            "airport_resolution_notes": "no destination to resolve",
            "execution_trace": [trace_entry("AirportResolverAgent", t0,
                "no destination", "skipped", "skipped")]
        }

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(lookup_iata, origin_city): "origin",
            pool.submit(lookup_iata, dest_city):   "destination",
        }
        for fut in as_completed(futures, timeout=15):
            key = futures[fut]
            try:
                results[key] = fut.result(timeout=12)
            except Exception as exc:
                city = origin_city if key == "origin" else dest_city
                results[key] = {"iata": city, "display": city}
                console.print(f"  ⚠️  {key} lookup failed: {exc}")

    origin_res = results.get("origin",      {"iata": origin_city, "display": origin_city})
    dest_res   = results.get("destination", {"iata": dest_city,   "display": dest_city})

    # Safety net: if iata is still a multi-word city name, warn
    for label, res in [("origin", origin_res), ("destination", dest_res)]:
        if " " in res["iata"]:
            console.print(
                f"  [red]⚠️  {label} IATA resolved to '{res['iata']}' "
                f"(multi-word — SerpAPI may reject).[/red]"
            )

    console.print(
        f"  ✅ origin=[yellow]{origin_res['iata']}[/yellow]"
        f" ({origin_res['display']}) | "
        f"dest=[yellow]{dest_res['iata']}[/yellow]"
        f" ({dest_res['display']})"
    )

    ms = round((time.time()-t0)*1000)
    return {
        "origin_iata":              origin_res["iata"],
        "destination_iata":         dest_res["iata"],
        "origin_display":           origin_res["display"],
        "destination_display":      dest_res["display"],
        "airport_resolution_notes": f"resolved in {ms}ms",
        "execution_trace": [trace_entry(
            "AirportResolverAgent", t0,
            f"{origin_city} + {dest_city}",
            f"{origin_res['iata']} + {dest_res['iata']}"
        )]
    }