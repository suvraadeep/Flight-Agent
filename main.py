import argparse, sys, time
from typing import List, Dict

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.prompt  import Prompt
from rich         import box

import config
from state import make_state
from graph.orchestrator import build_app

console = Console()
app     = build_app()

BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║   ✈️   REAL-TIME TRAVEL AGENT  —  CLI Chatbot                 ║
║   6 Agents · LangGraph · Groq LLaMA · SerpAPI Flights        ║
╠═══════════════════════════════════════════════════════════════╣
║  /quit  /clear  /history  /trace  /help  /lite  /status      ║
╚═══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
[bold]Commands[/bold]
  /quit      Exit chatbot
  /clear     Clear conversation history
  /history   Show conversation history
  /trace     Show last execution trace
  /lite      Toggle LITE_MODE (saves SerpAPI quota)
  /status    Show current config
  /help      Show this help

[bold]Example queries[/bold]
  "Book a flight to Paris next Friday"
  "2 business class tickets to Tokyo December 10th"
  "Cheapest flights to Dubai next month"
  "One way to London on 2026-05-15"
"""

STATUS_COLOR = {
    "success":              "green",
    "error":                "red",
    "fallback":             "yellow",
    "clarification_needed": "yellow",
    "skipped":              "dim",
}


def _print_trace(trace: List[Dict], total_ms: int,
                 searches_ok: int, searches_total: int,
                 price_insights: Dict):
    tbl = Table(title="Execution Trace", box=box.ROUNDED,
                show_header=True, header_style="bold magenta", width=95)
    tbl.add_column("Agent",  style="cyan", width=26)
    tbl.add_column("Status", width=22)
    tbl.add_column("Time",   width=8)
    tbl.add_column("Output", width=39)

    for e in trace:
        c = STATUS_COLOR.get(e.get("status",""), "white")
        tbl.add_row(
            e["agent"],
            f"[{c}]{e.get('status','')}[/{c}]",
            f"{e.get('duration_ms',0):.0f}ms",
            (e.get("output_summary","") or "")[:39],
        )
    console.print(tbl)

    note = ""
    if price_insights.get("price_level"):
        note += f" | price level: {price_insights['price_level']}"
    if price_insights.get("typical_range"):
        r = price_insights["typical_range"]
        note += f" | typical ${r[0]}-${r[1]}"

    console.print(
        f"[dim]Total: {total_ms}ms | "
        f"Searches: {searches_ok}/{searches_total} OK{note}[/dim]\n"
    )


def run_query(query: str, history: List[Dict], verbose: bool = False) -> Dict:
    state   = make_state(query, history)
    t_start = time.time()
    final   = app.invoke(state)
    total   = round((time.time() - t_start) * 1000)

    if verbose:
        _print_trace(
            final.get("execution_trace", []),
            total,
            final.get("total_searches_succeeded", 0),
            final.get("total_searches_attempted", 0),
            final.get("price_insights") or {},
        )
    else:
        agents = [e["agent"] for e in final.get("execution_trace", [])]
        ok     = final.get("total_searches_succeeded", 0)
        tot    = final.get("total_searches_attempted", 0)
        console.print(
            f"[dim]⚡ {' → '.join(agents)} | {total}ms"
            + (f" | {ok}/{tot} searches OK" if tot else "") + "[/dim]"
        )
    return final


def main():
    parser = argparse.ArgumentParser(description="Real-time Travel Planning Agent CLI")
    parser.add_argument("--fail", action="append", default=[],
                        metavar="LABEL",
                        help="Force-fail a search by label: best|cheapest|fastest")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full execution trace after each query")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Run a single query non-interactively and exit")
    parser.add_argument("--lite", action="store_true",
                        help="Enable LITE_MODE (1 SerpAPI call instead of 3)")
    args = parser.parse_args()

    if args.lite:
        config.LITE_MODE = True
        console.print("[yellow]⚡ LITE_MODE enabled — single SerpAPI call (saves quota)[/yellow]")

    config.FORCE_FAIL_SEARCHES = set(args.fail)
    if args.fail:
        console.print(f"[yellow]⚠️  Force-failing: {args.fail}[/yellow]")

    if args.query:
        final = run_query(args.query, [], verbose=True)
        console.print(Panel(
            final.get("final_response","—"),
            title="🤖  Assistant", border_style="green"
        ))
        sys.exit(0)

    console.print(BANNER, style="bold cyan")

    history:    List[Dict] = []
    last_trace: List[Dict] = []
    last_insights: Dict    = {}

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Safe travels ✈️ [/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Safe travels ✈️ [/dim]")
            break

        if cmd == "/clear":
            history.clear()
            console.print("[dim]History cleared.[/dim]")
            continue

        if cmd == "/history":
            if not history:
                console.print("[dim]No history yet.[/dim]")
            else:
                for m in history:
                    console.print(f"[bold]{m['role'].upper()}:[/bold] {m['content'][:120]}")
            continue

        if cmd == "/trace":
            if last_trace:
                _print_trace(last_trace, 0, 0, 0, last_insights)
            else:
                console.print("[dim]No trace yet.[/dim]")
            continue

        if cmd == "/lite":
            config.LITE_MODE = not config.LITE_MODE
            console.print(
                f"[yellow]LITE_MODE {'ON (1 search)' if config.LITE_MODE else 'OFF (3 searches)'}[/yellow]"
            )
            continue

        if cmd == "/status":
            console.print(
                f"  model:    {config.MODEL_NAME}\n"
                f"  lite:     {config.LITE_MODE}\n"
                f"  currency: {config.CURRENCY}\n"
                f"  timeout:  {config.PROVIDER_TIMEOUT}s\n"
                f"  history:  {len(history)//2} turns\n"
                f"  fail_set: {config.FORCE_FAIL_SEARCHES or 'none'}"
            )
            continue

        if cmd == "/help":
            console.print(HELP_TEXT)
            continue

        console.print("[dim]Searching…[/dim]")
        final         = run_query(user_input, history, verbose=args.verbose)
        last_trace    = final.get("execution_trace", [])
        last_insights = final.get("price_insights") or {}
        response      = final.get("final_response", "—")

        console.print(Panel(response, title="🤖  Assistant", border_style="green"))

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})
        if len(history) > config.MAX_HISTORY * 2:
            history = history[-(config.MAX_HISTORY * 2):]


if __name__ == "__main__":
    main()