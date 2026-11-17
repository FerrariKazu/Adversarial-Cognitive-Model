# Gate-2 Rules v1.0 Erratum 01

Status: **RECORDED, NOT REWRITTEN.**

Applies to: `prereg/gate2_rules.json` (SHA-256 `75286ee00a803f41b1d80b3dad11ef6d7eeb5bb999876fc0f774b7e26c73b63a`), frozen in commit `fe039292fe1a7eac5c8956bf8046e78f01e1549a`.

## 1. Timestamp

v1.0 records `created_at_utc: 2026-11-17T01:57:00Z`. Commit `fe03929` is dated `2026-11-17T02:00:31+02:00`, which is `2026-11-17T00:00:31Z`. The recorded value is therefore in the future relative to the freeze commit.

The commit time is the real freeze time. v1.0 is preserved in git history and is not edited. The corrected value is recorded in `prereg/gate2_rules_v1_1.json` under `erratum_01`.

## 2. Scene counts

v1.0 described 1,700 fit scenes, 1,700 calibration scenes, and 1,600 test scenes. Under the frozen "one image pair is the strict unit" contract, those counts are not achievable without reusing images inside a pool. Each 1,700-image target pool contains 850 target and 850 background images, so at most 850 non-reused pairs are available; the 1,600-image test pool has at most 800.

Reusing images inside a pool would make scenes dependent and break LTT exchangeability. v1.1 therefore caps scene generation at 850 fit null scenes, 850 calibration scenes, and 800 test scenes. The image pools themselves are unchanged. No image is reused within a pool; cross-pool disjointness is unchanged.

## 3. Gate-2 markdown hash

The Gate-2 markdown draft hash is pinned in the code-lock block of `prereg/gate2_rules_v1_1.json`:

`docs/research/RHAN_IM_GATE2_PREREG_DRAFT.md` -> `26f74e7926fb387fb69da85a3ad6a3bf61594de5565661f30e3afe3efbb10713`
