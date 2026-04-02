import re, time
from typing import Dict, List, Optional

from ddgs import DDGS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

import config

console = Console()

_iata_cache: Dict[str, Dict[str, str]] = {}

_IATA_BLOCKLIST = {
    "THE","AND","FOR","BUT","NOT","ARE","WAS","HAS","HAD","ITS",
    "ALL","NEW","ONE","TWO","ANY","CAN","MAY","GET","SET","USE",
    "URL","API","CSS","USA","USD","EUR","GBP","GMT","UTC","EST",
    "PDT","PST","NYC",
}


def _ddg_search(query: str, max_results: int = 5) -> List[Dict]:
    """DuckDuckGo text search with retry."""
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return []


def _extract_iata_from_text(text: str) -> Optional[str]:
    """Regex extraction of IATA code from a block of text."""
    patterns = [
        r'IATA\s*(?:code)?\s*[:\s]+([A-Z]{3})\b',
        r'\(([A-Z]{3})\)',
        r'\b([A-Z]{3})\s+(?:Airport|International|Intl)\b',
        r'code\s+([A-Z]{3})\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            code = m.group(1).upper()
            if code not in _IATA_BLOCKLIST and len(code) == 3:
                return code
    return None


def _llm_lookup_iata(city: str) -> Optional[str]:
    """
    Ask the LLM for the IATA code.
    Groq's LLaMA model has every major and minor airport in its training data.
    Returns a validated 3-letter code or None.
    """
    try:
        llm = ChatGroq(
            model=config.MODEL_NAME,
            temperature=0.0,
            api_key=config.GROQ_API_KEY,
            max_tokens=10,
        )
        chain = ChatPromptTemplate.from_messages([
            ("system",
             "You are an airport code expert. "
             "Given any city, region, or airport name, return ONLY the primary "
             "international airport IATA code — exactly 3 uppercase letters, nothing else. "
             "No punctuation, no explanation, no markdown. "
             "If genuinely unknown, return the 3 letters UNK."),
            ("human", "{city}")
        ]) | llm

        resp = chain.invoke({"city": city})
        code = resp.content.strip().upper()

        if re.match(r'^[A-Z]{3}$', code) and code != "UNK" and code not in _IATA_BLOCKLIST:
            return code
        return None

    except Exception as e:
        console.print(f"  [dim yellow]LLM IATA lookup failed for '{city}': {e}[/dim yellow]")
        return None


def _ddg_lookup_iata(city: str) -> Optional[str]:
    """DuckDuckGo fallback for cities the LLM couldn't confidently resolve."""
    results = _ddg_search(f"{city} international airport IATA code", max_results=5)
    if not results:
        return None
    combined = " ".join(
        r.get("title", "") + " " + r.get("body", "")
        for r in results[:4]
    )
    return _extract_iata_from_text(combined)


def lookup_iata(city: str) -> Dict[str, str]:
    """
    Resolve any city/airport name → IATA code.

    Returns {"iata": "GAU", "display": "Guwahati (GAU)"}
    Never raises — falls back gracefully at every step.
    """
    city = city.strip()
    if not city:
        return {"iata": "", "display": ""}

    # 1. Already a valid IATA code
    if re.match(r'^[A-Z]{3}$', city):
        return {"iata": city, "display": city}

    cache_key = city.lower()

    # 2. Session cache
    if cache_key in _iata_cache:
        cached = _iata_cache[cache_key]
        console.print(
            f"  [dim]💾 cache hit: [yellow]{city}[/yellow] → "
            f"[green]{cached['iata']}[/green][/dim]"
        )
        return cached

    # 3. LLM lookup (primary)
    console.print(f"  🔍 resolving [yellow]{city}[/yellow] via LLM…")
    iata = _llm_lookup_iata(city)
    source = "LLM"

    # 4. DuckDuckGo fallback
    if not iata:
        console.print(f"  [dim]LLM miss → trying DuckDuckGo…[/dim]")
        iata = _ddg_lookup_iata(city)
        source = "DuckDuckGo"

    # 5. Pass city name through (graceful degradation)
    if not iata:
        console.print(
            f"  [yellow]⚠️  Could not resolve IATA for '{city}'. "
            f"Passing city name to SerpAPI — it may still work.[/yellow]"
        )
        result = {"iata": city, "display": city}
        _iata_cache[cache_key] = result
        return result

    display = f"{city.strip().title()} ({iata})"
    console.print(
        f"  ✅ [yellow]{city}[/yellow] → [green]{iata}[/green] "
        f"[dim]({source})[/dim]"
    )

    result = {"iata": iata, "display": display}
    _iata_cache[cache_key] = result
    return result