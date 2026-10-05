"""Read-only inspection of the Stage 1 verdict as recorded on HF (2026-08-09).

Fetches rhan_next_roadmap.json from FerrariKazu/rhan-checkpoints-rolling and
prints the stage-1 fields that matter for the verdict review, plus lists the
rhan_next_* artifacts present on the two HF repos. Does not write anything.
"""
import json
import urllib.request

REPO = "FerrariKazu/rhan-checkpoints-rolling"
BEST = "FerrariKazu/rhan-checkpoints"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "stage1-review"})
    return urllib.request.urlopen(req, timeout=90)


def main():
    # ── 1. Roadmap (the recorded verdict) ──────────────────────────────────
    try:
        data = json.load(get(
            f"https://huggingface.co/datasets/{REPO}/resolve/main/"
            "rhan_next_roadmap.json"))
        print(f"roadmap fetched OK | roadmap_rev = {data.get('roadmap_rev')}")
    except Exception as e:
        print("roadmap fetch FAILED:", e)
        return

    s1 = data.get("stages", {}).get("1", {})
    print("\n=== stages['1'] recorded fields ===")
    for k in ("run_label", "run_identity", "validated", "validated_date",
              "gate_clear_path", "gate_clear_reason", "step_b_resume"):
        v = s1.get(k)
        print(f"  {k}: {json.dumps(v)[:300] if v else v}")

    note = s1.get("validated_note") or ""
    print(f"  validated_note: {note[:220]}")

    sv = s1.get("stage1_verdict", {})
    print(f"\n  stage1_verdict keys: {list(sv.keys())}")
    print("  results rows:")
    for r in (sv.get("results") or []):
        print(f"    {r.get('ckpt_label'):<28} eps={r.get('eps_pixel')} "
              f"acc={r.get('acc_mean')}±{r.get('acc_std')}  d'={r.get('dprime_mean')}")
    print("  crossover_verdicts:")
    for cv in (sv.get("crossover_verdicts") or []):
        print(f"    {cv}")
    print("  checkpoints recorded in provenance:")
    for c in (sv.get("checkpoints") or []):
        print(f"    {c.get('label'):<28} -> {c.get('path')}")
    mc = sv.get("masking_check") or {}
    print(f"\n  masking_check keys: {list(mc.keys())}")
    for label in ("rhan_next_ais_v1_halting_only", "trades_large_baseline"):
        print(f"    {label}: {json.dumps(mc.get(label))[:240]}")

    iso = s1.get("isolation_verdict") or {}
    print(f"\n  isolation_verdict.status = {iso.get('status')} | "
          f"selected_step_b_config = {iso.get('selected_step_b_config')}")
    print(f"  isolation_verdict.decision: {(iso.get('decision') or '')[:160]}")
    for arm, rec in (iso.get("arms") or {}).items():
        print(f"    arm {arm:<28} top2={rec.get('top2')} restored={rec.get('car_truck_restored')}")

    # ── 2. Artifacts on HF ────────────────────────────────────────────────
    print("\n=== rhan_next_* artifacts on HF ===")
    for repo, tag in ((REPO, "rolling"), (BEST, "best")):
        files = [f["path"] for f in json.load(get(
            f"https://huggingface.co/api/datasets/{repo}/tree/main"))
            if "rhan_next" in f["path"]]
        print(f"  [{tag} repo] {repo}:")
        for f in sorted(files):
            print(f"    {f}")


if __name__ == "__main__":
    main()
