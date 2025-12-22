#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    p = argparse.ArgumentParser(
        description="NPT Agents: minimal project CLI (safe, incremental, demo-friendly)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Safe extract: only add NEW pdfs; never drop table
    sub.add_parser("extract", help="Extract paragraphs from NEW PDFs into positions.db (safe)")

    # Classify only NULL issues rows (incremental)
    sub.add_parser("classify", help="Classify paragraphs where issues IS NULL (incremental)")

    # Export buckets from DB
    sub.add_parser("export", help="Export agent buckets from DB")

    # Build embeddings
    sub.add_parser("embed", help="Build bucket embeddings")

    # Build lite buckets
    sub.add_parser("lite", help="Build lite buckets")

    # Run simulation
    sim = sub.add_parser("simulate", help="Run agent simulation (writes outputs/simulation_logs)")
    sim.add_argument("--config", default=None, help="Optional config yaml if your script supports it")

    # Run streamlit
    sub.add_parser("ui", help="Launch Streamlit viewer")

    args = p.parse_args()

    if args.cmd == "extract":
        run([sys.executable, "extract_paragraphs_to_db.py"])

    elif args.cmd == "classify":
        run([sys.executable, "classify_paragraphs_with_gpt.py"])

    elif args.cmd == "export":
        run([sys.executable, "export_buckets_python_only.py"])

    elif args.cmd == "embed":
        run([sys.executable, "build_bucket_embeddings.py"])

    elif args.cmd == "lite":
        run([sys.executable, "build_lite_buckets.py"])

    elif args.cmd == "simulate":
        cmd = [sys.executable, "agent_simulation_new.py"]
        if args.config:
            cmd += ["--config", args.config]
        run(cmd)

    elif args.cmd == "ui":
        run(["streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    main()