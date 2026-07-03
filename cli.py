#!/usr/bin/env python3
"""
cli.py - Command-line interface for testing the multi-agent pipeline.

Usage:
    python cli.py "Analiza la vegetación en Madrid en 2024"
    python cli.py --demo   # Run with a built-in demo query
"""

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()


DEMO_QUERY = (
    "Analiza el estado de la vegetación en la Comunidad de Madrid "
    "entre junio y agosto de 2024 usando imágenes Sentinel-2. "
    "Calcula el NDVI y el EVI y muestra las estadísticas principales."
)


def main():
    parser = argparse.ArgumentParser(description="GeoMultiAgent CLI")
    parser.add_argument("query", nargs="?", help="Natural language query")
    parser.add_argument("--demo", action="store_true", help="Run demo query")
    parser.add_argument("--json", action="store_true", help="Output raw JSON state")
    args = parser.parse_args()

    query = DEMO_QUERY if args.demo else args.query
    if not query:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  GeoMultiAgent Pipeline")
    print(f"{'='*60}")
    print(f"  Query: {query}")
    print(f"{'='*60}\n")

    from workflow import run_query
    state = run_query(query)

    if args.json:
        # Remove non-serialisable keys
        safe = {k: v for k, v in state.items() if k not in ("map_html",)}
        print(json.dumps(safe, indent=2, ensure_ascii=False, default=str))
        return

    # Human-readable summary
    print("\n📋 PLAN")
    if state.get("plan"):
        p = state["plan"]
        print(f"  Location   : {p.get('location',{}).get('name','?')}")
        print(f"  Dates      : {p.get('date_range',{}).get('start','?')} → {p.get('date_range',{}).get('end','?')}")
        print(f"  Analysis   : {p.get('analysis_type','?')}")
        print(f"  Indices    : {', '.join(p.get('required_indices',[]))}")
        print(f"  Satellites : {', '.join(p.get('satellites',[]))}")

    print("\n🛰  DATA ACQUISITION")
    scenes = state.get("available_scenes") or []
    files  = state.get("downloaded_files") or []
    print(f"  Scenes found : {len(scenes)}")
    print(f"  Files downloaded : {len(files)}")

    print("\n📊 ANALYSIS")
    indices = state.get("computed_indices") or {}
    for name, stats in indices.items():
        print(f"  {name}: {stats}")
    print(f"  Code iterations : {state.get('code_iterations', 0)}")

    print("\n📝 REPORT")
    report = state.get("report_markdown", "")
    if report:
        # Print first 800 chars
        print(report[:800])
        if len(report) > 800:
            print("  … [truncated]")

    print("\n✅ STATUS:", state.get("status"))
    if state.get("error_message"):
        print("❌ ERROR:", state["error_message"])

    output_files = state.get("output_files") or []
    if output_files:
        print("\n📁 OUTPUT FILES:")
        for f in output_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
