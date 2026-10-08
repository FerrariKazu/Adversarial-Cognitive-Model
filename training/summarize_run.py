\"\"\"Offline summarizer for a phase epoch log (Step 5).

Usage:
  python -m training.summarize_run --phase backbone_only
  python -m training.summarize_run --phase backbone_only \\
      --jsonl /path/to/backbone_only_epoch_log.jsonl

Reads the JSONL and prints a table of all epochs (clean, robust, losses,
grad norms, flags) plus a plain-text trend verdict. Works from the
HF-synced file alone, so a dead session can be inspected from a phone
without a GPU.
\"\"\"\nfrom __future__ import annotations\n\nimport argparse\nimport os\nimport sys\nfrom typing import Optional, Sequence\n\ntry:\n    from training.run_report_jsonl import summarize_phase_from_jsonl\nexcept Exception as e:\n    print(f\"ERROR: cannot import summarizer: {e}\", file=sys.stderr)\n    sys.exit(2)\n\n\ndef main(argv: Optional[Sequence[str]] = None) -> int:\n    ap = argparse.ArgumentParser(\n        description=\"Summarize a phase epoch log from its JSONL.\")\n    ap.add_argument(\"--phase\", default=\"backbone_only\")\n    ap.add_argument(\"--jsonl\", default=None)\n    ap.add_argument(\"--report-dir\", default=None)\n    args = ap.parse_args(argv)\n    report_dir = args.report_dir or os.environ.get(\n        \"REPORT_DIR\",\n        os.path.join(os.path.dirname(__file__), \"..\", \"report\"))\n    jsonl = args.jsonl or os.path.join(\n        report_dir, f\"{args.phase}_epoch_log.jsonl\")\n    print(summarize_phase_from_jsonl(jsonl))\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n