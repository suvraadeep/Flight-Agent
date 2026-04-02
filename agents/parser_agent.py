import re, json, time
from datetime import datetime, timedelta
from typing import Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from config import GROQ_API_KEY, MODEL_NAME
from state import TravelState
from utils.logger import trace_entry

console = Console()

_SYSTEM = """You are a travel request parser.
Return ONLY a JSON object — no markdown, no extra text.
Keys:
  destination    : string or null  (city name e.g. "Paris")
  origin         : string (default "New York" if not stated)
  raw_date_text  : exact phrase user said for date, e.g. "next Friday", or null
  travel_date    : YYYY-MM-DD if date is explicit and absolute, else null
  travel_type    : flight|hotel|train|cruise or null
  num_passengers : integer (default 1)
  cabin_class    : economy|business|first|premium_economy (default "economy")
  budget_hint    : string or null (e.g. "cheap","under $500","business class")
Today is {today}."""


def _safe_json(text: str) -> Optional[Dict]:
    for pat in [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```', r'(\{[\s\S]*\})']:
        m = re.search(pat, text)
        if m:
            try: return json.loads(m.group(1).strip())
            except: pass
    try: return json.loads(text.strip())
    except: return None


def _resolve_date(raw: str) -> Optional[str]:
    raw   = raw.lower().strip()
    today = datetime.now()
    simple = {
        "today":0,"tomorrow":1,"day after tomorrow":2,
        "next week":7,"next month":30,"this weekend":5,"next weekend":6,
    }
    for phrase, delta in simple.items():
        if phrase in raw:
            return (today + timedelta(days=delta)).strftime("%Y-%m-%d")
    days = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
            "friday":4,"saturday":5,"sunday":6}
    for name, idx in days.items():
        if name in raw:
            ahead = idx - today.weekday()
            if ahead <= 0: ahead += 7
            return (today + timedelta(days=ahead)).strftime("%Y-%m-%d")
    m = re.search(r'(\d{4}-\d{2}-\d{2})', raw)
    if m: return m.group(1)
    return None


def parser_agent(state: TravelState) -> Dict:
    t0 = time.time()
    console.print("\n[bold blue]🔤 ParserAgent[/bold blue]")

    hist = "".join(
        f"\n{m['role']}: {m['content']}"
        for m in state.get("conversation_history",[])[-4:]
    )

    try:
        llm   = ChatGroq(model=MODEL_NAME, temperature=0.0,
                         api_key=GROQ_API_KEY, max_tokens=1024)
        chain = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM.format(today=datetime.now().strftime("%Y-%m-%d"))),
            ("human", "Request: {inp}\nConversation history:{hist}")
        ]) | llm

        resp   = chain.invoke({"inp": state["user_input"], "hist": hist})
        parsed = _safe_json(resp.content)
        if not parsed:
            raise ValueError(f"No JSON in LLM output: {resp.content[:120]}")

        if not parsed.get("travel_date") and parsed.get("raw_date_text"):
            parsed["travel_date"] = _resolve_date(parsed["raw_date_text"])

        console.print(
            f"  ✅ dest=[yellow]{parsed.get('destination')}[/yellow] "
            f"origin=[yellow]{parsed.get('origin')}[/yellow] "
            f"date=[yellow]{parsed.get('travel_date')}[/yellow] "
            f"pax=[yellow]{parsed.get('num_passengers')}[/yellow] "
            f"cabin=[yellow]{parsed.get('cabin_class')}[/yellow]"
        )

        return {
            "parsed_data": parsed, "parse_error": None,
            "execution_trace": [trace_entry(
                "ParserAgent", t0, state["user_input"][:60],
                f"dest={parsed.get('destination')}, date={parsed.get('travel_date')}"
            )]
        }
    except Exception as e:
        console.print(f"  ❌ {e}")
        return {
            "parsed_data": None, "parse_error": str(e),
            "execution_trace": [trace_entry("ParserAgent", t0,
                state["user_input"][:60], str(e), "error")]
        }