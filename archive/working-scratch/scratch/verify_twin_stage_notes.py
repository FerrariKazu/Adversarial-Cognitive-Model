#!/usr/bin/env python3
"""Post-edit twin verification for the NOESIS notebooks.

The Colab/Kaggle twins must remain byte-identical in every *code* region;
only the docstring (platform framing) is allowed to differ. This script
checks:
  1. The Stage 1 toggle block (Step 4) is identical.
  2. The Stage 2 block (from its markdown header to EOF) is identical.
  3. The Done cell region is identical.
  4. The new Stage 1 defaults are all False.
"""
from pathlib import Path

A = Path("cloud_setup/colab_notebook_noesis.py").read_text()
B = Path("cloud_setup/Kaggle_NOESIS.py").read_text()


def region(src, start, end=None):
    i = src.index(start)
    j = src.index(end, i + 1) if end else len(src)
    return src[i:j]


checks = []

# 1. Stage 1 toggles block
tog_start = "DO_STEP_A   = False"
tog_end = "DO_RESUME_SELFTEST = False"
checks.append(("Stage1 toggles", region(A, tog_start, tog_end) == region(B, tog_start, tog_end)))

# 2. Stage 2 block -> EOF
s2 = "# ## Stage 2 — HPC (Pillar 1, matrix C)"
checks.append(("Stage2 block->EOF", region(A, s2) == region(B, s2)))

# 3. Done cell region
done_start = "# ## Done — next gate"
done_end = 'print("="*70)'  # last banner line of the done cell
checks.append(("Done cell", region(A, done_start, done_end) == region(B, done_start, done_end)))

# 4. Stage 1 defaults all False
import re
flags = ["DO_STEP_A", "DO_STEP_B", "DO_STEP_C", "DO_ISOLATION",
         "SEED_STEP_B_FROM_ISOB", "DO_RESUME_SELFTEST", "DO_STEP_C2"]
for f in flags:
    m = re.search(rf"^{f}\s*=\s*(True|False)", A, re.M)
    checks.append((f"{f} = {m.group(1)}", m and m.group(1) == "False"))
    m2 = re.search(rf"^{f}\s*=\s*(True|False)", B, re.M)
    assert m2.group(1) == m.group(1), f"{f} differs between twins"

ok = True
for name, passed in checks:
    print(f"  {'OK ' if passed else 'FAIL'} {name}")
    ok = ok and passed

# Stage 2 toggles must remain ACTIVE
for f in ["DO_STEP2_A", "DO_STEP2_B", "DO_STEP2_C"]:
    m = re.search(rf"^{f}\s*=\s*True", A, re.M)
    print(f"  {'OK ' if m else 'FAIL'} {f} still defaults True")
    ok = ok and bool(m)

print("\nALL TWIN CHECKS PASS" if ok else "\nTWIN CHECKS FAILED")
raise SystemExit(0 if ok else 1)
