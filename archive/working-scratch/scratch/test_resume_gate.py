#!/usr/bin/env python3
"""Standalone unit test for the resume-gate helpers in colab_v12_step0.py.

Extracts the Step 4b helper cells from the notebook file and executes the
actual function definitions with a mocked HF layer, covering:
  1. first run (no HF rolling)            -> returns None, no download
  2. mandatory resume (HF has rolling)    -> restores locally, returns epoch
  3. HF has rolling but download fails    -> RuntimeError (fail-fast)
  4. local already present                -> local epoch, no HF call
  5. verify_no_restart passes when epoch advanced / held
  6. verify_no_restart aborts on backward epoch
  7. restore_best_or_abort succeeds
"""
import os
import sys
import json
import types

# ── extract the Step 4b helper cell ────────────────────────────────────────────
src = open("cloud_setup/colab_v12_step0.py").read()
cells = src.split("# %%")
helpers_src = None
for c in cells:
    if "def _ckpt_epoch" in c and "def restore_rolling_or_abort" in c:
        helpers_src = c
        break
assert helpers_src is not None, "Step 4b helper cell not found"

# keep only Python code lines (drop markdown `# ...` prose and `# %%` marks)
code_lines = []
for line in helpers_src.splitlines():
    if line.startswith("#"):
        continue
    code_lines.append(line)
helpers_code = "\n".join(code_lines)


# ── mocks ──────────────────────────────────────────────────────────────────────
os.environ["HF_TOKEN"] = "fake"

class _TorchStub:
    @staticmethod
    def load(path, map_location=None, weights_only=False):
        return json.load(open(path))

# ── patch sys.modules['huggingface_hub'] so the helpers' internal
#    `from huggingface_hub import hf_hub_download` gets our mock (the import is
#    inside the functions, so a global ns entry would be shadowed) ────────────
def _fake_download(**kw):
    fn = kw["filename"]
    with open(f"checkpoints/{fn}", "w") as f:
        f.write(json.dumps({"epoch": 7, "best_acc": 42.1}))
    return f"checkpoints/{fn}"

_hf_hub_mod = types.ModuleType("huggingface_hub")
_hf_hub_mod.hf_hub_download = _fake_download
_hf_hub_mod.HfApi = object
sys.modules["huggingface_hub"] = _hf_hub_mod

ns = {"os": os, "torch": _TorchStub}
exec(helpers_code, ns)

restore_rolling_or_abort = ns["restore_rolling_or_abort"]
verify_no_restart = ns["verify_no_restart"]
restore_best_or_abort = ns["restore_best_or_abort"]

os.makedirs("checkpoints", exist_ok=True)


def _rm(name):
    if os.path.exists(f"checkpoints/{name}"):
        os.remove(f"checkpoints/{name}")


results = []
def check(name, ok):
    results.append((name, ok))
    print(("PASS" if ok else "FAIL"), "-", name)


# Case 1: first run (rolling file not on HF) -> returns None
ns["_hf_rolling_files"] = lambda: ["rhan_v12_mixA_rolling.pth", "rhan_v12_mixB_rolling.pth"]
ns["_time"] = types.SimpleNamespace(sleep=lambda s: None)
_rm("rhan_v12_ghost_rolling.pth")
check("case1 first-run returns None (no download)", restore_rolling_or_abort("rhan_v12_ghost") is None)

# Case 2: HF HAS rolling, download OK -> restores and returns epoch
ns["ROLLING_REPO"] = "FerrariKazu/rhan-checkpoints-rolling"
ns["BEST_REPO"] = "FerrariKazu/rhan-checkpoints"
ns["_hf_rolling_files"] = lambda: ["rhan_v12_mixA_rolling.pth"]
_rm("rhan_v12_mixA_rolling.pth")
check("case2 mandatory resume restores epoch 7", restore_rolling_or_abort("rhan_v12_mixA") == 7)

# Case 3: HF HAS rolling but download fails 3x -> RuntimeError (fail-fast)
ns["_hf_rolling_files"] = lambda: ["rhan_v12_mixB_rolling.pth"]
_rm("rhan_v12_mixB_rolling.pth")
def failing_download(**kw):
    raise ConnectionError("boom")
_hf_hub_mod.hf_hub_download = failing_download
try:
    restore_rolling_or_abort("rhan_v12_mixB")
    check("case3 fail-fast aborts", False)
except RuntimeError as e:
    check("case3 fail-fast aborts", "FATAL" in str(e))

# Case 4: local already present -> returns local epoch, no HF listing call
with open("checkpoints/rhan_v12_mixA_rolling.pth", "w") as f:
    f.write(json.dumps({"epoch": 9}))
called = {"n": 0}
def _counting_list():
    called["n"] += 1
    return []
ns["_hf_rolling_files"] = _counting_list
check("case4 local-first returns 9, no HF call",
      restore_rolling_or_abort("rhan_v12_mixA") == 9 and called["n"] == 0)

# Case 5: verify passes when epoch held/advanced (9 >= 7)
try:
    verify_no_restart("rhan_v12_mixA", 7)
    check("case5 verify passes (9 >= 7)", True)
except RuntimeError:
    check("case5 verify passes (9 >= 7)", False)

# Case 6: verify aborts when epoch went backward (3 < 7)
with open("checkpoints/rhan_v12_mixB_rolling.pth", "w") as f:
    f.write(json.dumps({"epoch": 3}))
try:
    verify_no_restart("rhan_v12_mixB", 7)
    check("case6 verify aborts on backward epoch", False)
except RuntimeError as e:
    check("case6 verify aborts on backward epoch", "BACKWARD" in str(e))

# Case 7: restore_best_or_abort succeeds
_hf_hub_mod.hf_hub_download = _fake_download
try:
    restore_best_or_abort("rhan_v12_mixA_best.pth")
    check("case7 best restore OK", os.path.exists("checkpoints/rhan_v12_mixA_best.pth"))
except Exception:
    check("case7 best restore OK", False)

ok_all = all(ok for _, ok in results)
print("\nALL HELPER TESTS PASSED" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
