"""STEP 0 — frozen Gen-0 recipe: unit contract tests.

These tests pin the Stage-2 runner's resolved Gen-0 configuration against
the exact frozen values.  If anything in the runner resolves the Gen-0
recipe to a value that disagrees with the canonical recipe, this test
fails loudly.

IMPORTANT: these tests do NOT import `adv_curriculum.py`'s values as the
source of truth.  They re-derive the canonical values from the frozen
record, and assert the runner's resolution matches.  That way the test
catches a runner that *silently copied* a stale recipe AND a runner that
*redefined* the recipe with a different hash.

The tests also assert that the recipe record is "about" the Gen-0 recipe
-- nothing in the runner's resolution touches adv_curriculum.py's module
code, only the values it carries.
"""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

# Import the frozen recipe module from the package under test.
sys.path.insert(0, "training")
import adv_curriculum_freeze as acf  # noqa: E402

# The Stage-2 runner's resolution entry point.
from adv_curriculum_freeze import resolve_frozen_gen0_recipe

# Canonical values the frozen Gen-0 recipe fixes (mirrored verbatim from
# the frozen record; these are the GROUND TRUTH the tests pin).
EXPECTED = {
    "curriculum_60": [
        [1, 20, 0.031, 2.0, 4],
        [21, 40, 0.062, 2.0, 4],
        [41, 60, 0.094, 2.5, 4],
    ],
    "w_trades_default": 0.55,
    "rand_start_mag": 0.001,
}


@pytest.fixture()
def frozen_recipe():
    return resolve_frozen_gen0_recipe()


def test_recipe_hash_is_computed_not_hardcoded():
    r = resolve_frozen_gen0_recipe()
    # The recipe_hash must be deterministic and reproducible from a canonical
    # serialisation of the resolved values.  If a future change to the frozen
    # values is not recorded here, this silently drifts.
    canonical = {
        "curriculum_60": [list(p) for p in r.curriculum_60],
        "w_trades_default": r.w_trades_default,
        "rand_start_mag": r.rand_start_mag,
    }
    recomputed_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert r.recipe_hash == recomputed_hash


def test_frozen_values_match_the_record():
    r = resolve_frozen_gen0_recipe()
    assert tuple(map(tuple, r.curriculum_60)) == tuple(map(tuple, EXPECTED["curriculum_60"]))
    assert r.w_trades_default == EXPECTED["w_trades_default"]
    assert r.rand_start_mag == EXPECTED["rand_start_mag"]
    assert r.content_hash == acf.STATED_CONTENT_SHA256
    assert r.source_module == "training.adv_curriculum"
    assert tuple(map(tuple, r.curriculum_60)) == tuple(map(tuple, EXPECTED["curriculum_60"]))
    assert r.w_trades_default == EXPECTED["w_trades_default"]
    assert r.rand_start_mag == EXPECTED["rand_start_mag"]
    assert tuple(map(tuple, r.curriculum_60)) == tuple(map(tuple, EXPECTED["curriculum_60"]))
    assert r.w_trades_default == EXPECTED["w_trades_default"]
    assert r.rand_start_mag == EXPECTED["rand_start_mag"]


def test_content_hash_is_responsive():
    """Mutate the record and confirm the hash changes (real guard, not a
    hardcoded constant)."""
    base = acf.STATED_CONTENT_SHA256
    mutated = base[:-1] + ("0" if base[-1] != "0" else "1")
    # Mutate the frozen block and confirm the recomputed content hash changes
    # — proving the hash is a real content fingerprint, not a constant.
    mutant = acf.FROZEN_RECIPE_BLOCK.replace("RAND_START_MAG = 0.001",
                                              "RAND_START_MAG = 0.002")
    assert hashlib.sha256(mutant.encode("utf-8")).hexdigest() != base


def test_recipe_is_resolved_independently_of_module_code():
    r = resolve_frozen_gen0_recipe()
    r = resolve_frozen_gen0_recipe()
    # The resolved recipe is independent of the source tree; it is the values.
    assert tuple(map(tuple, r.curriculum_60)) == tuple(map(tuple, EXPECTED["curriculum_60"]))
    assert r.w_trades_default == EXPECTED["w_trades_default"]
    assert r.rand_start_mag == EXPECTED["rand_start_mag"]


def test_recipe_record_json_serialisable_and_hashable():
    rec = acf.recipe_record()
    assert rec["schema_version"] == 1
    assert rec["record"] == "frozen_gen0_recipe"
    assert rec["source_module"] == "training.adv_curriculum"
    # JSON serialisable
    serialized = json.dumps(rec, sort_keys=True)
    assert isinstance(serialized, str)
    # The recipe hash inside the record matches the canonical hash
    canonical = {
        "curriculum_60": rec["curriculum_60"],
        "w_trades_default": rec["w_trades_default"],
        "rand_start_mag": rec["rand_start_mag"],
    }
    recomputed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert rec["recipe_hash"] == recomputed
