import time
from datetime import datetime
from typing import Dict

from rich.console import Console

from state import TravelState
from utils.logger import trace_entry

console = Console()

_REQUIRED = {
    "flight":  ["destination", "travel_date"],
    "hotel":   ["destination", "travel_date"],
    "train":   ["destination", "travel_date"],
    "cruise":  ["destination", "travel_date"],
    "default": ["destination"],
}
_CLARIFY = {
    "destination": "Where would you like to travel? (city name)",
    "travel_date": "When do you want to travel? (e.g. 'next Friday' or 'Dec 15')",
    "travel_type": "What type of travel? (flight / hotel / train / cruise)",
}


def validator_agent(state: TravelState) -> Dict:
    t0 = time.time()
    console.print("\n[bold blue]✔️  ValidatorAgent[/bold blue]")

    if state.get("parse_error") or not state.get("parsed_data"):
        msg = ("I couldn't parse that request. "
               "Please try: 'Book a flight to Paris on December 15th'")
        return {
            "validation_result":   {"is_valid": False, "errors": ["parse_failed"]},
            "needs_clarification": True,
            "clarification_message": msg,
            "execution_trace": [trace_entry("ValidatorAgent", t0,
                "no data", msg[:60], "clarification_needed")]
        }

    p      = state["parsed_data"]
    ttype  = (p.get("travel_type") or "default").lower()
    req    = _REQUIRED.get(ttype, _REQUIRED["default"])
    missing, errors = [], []

    for f in req:
        if not p.get(f):
            missing.append(f)

    if p.get("travel_date"):
        try:
            td = datetime.strptime(p["travel_date"], "%Y-%m-%d")
            if td.date() < datetime.now().date():
                errors.append(f"Date {p['travel_date']} is in the past")
        except ValueError:
            errors.append(f"Unrecognised date format: {p['travel_date']}")
            if "travel_date" not in missing:
                missing.append("travel_date")

    is_valid    = not missing and not errors
    clarify_msg = None
    if not is_valid:
        parts = errors + [_CLARIFY.get(f, f"Please provide {f}") for f in missing]
        clarify_msg = "I need a bit more info: " + "  ⬤  ".join(parts)

    status = "success" if is_valid else "clarification_needed"
    console.print(f"  {'✅ Valid' if is_valid else f'⚠️  missing={missing} errors={errors}'}")

    return {
        "validation_result": {
            "is_valid": is_valid, "missing_fields": missing,
            "errors": errors, "travel_type": ttype,
        },
        "needs_clarification":   not is_valid,
        "clarification_message": clarify_msg,
        "execution_trace": [trace_entry("ValidatorAgent", t0,
            f"required={req}",
            f"valid={is_valid}, missing={missing}",
            status)]
    }