# Git Branch Archaeology Audit — Before Cleanup

**Created:** 2026-10-05
**Source ref:** `diagnosis/nxa-forensic-2026-10-03` (tag `forensic-nxa-2026-10-03-final`)
**Current HEAD:** `71b2753ca8a9d2bd6fd84c20f0c141742ffe9496` (stage2/nxa-pipeline-refactor)
**Working tree:** refactor moves committed; untracked archive/diagnostic/generated dirs; `.gitignore` modified

---

## Fire every deletion has an explicit reason (Phase 11 table — complete)

| Branch | Local/Rmt? | Unique vs main | Unique vs frn | PR | Historical work preserved? | Action | Reason |
|---|---|---|---|---|---|---|---|
| main | local | — | 1 | — | yes | KEEP | default branch |
| feature/rhan-next | local | 74 | — | — | yes | KEEP | current development |
| stage2/nxa-pipeline-refactor | local | 80 | 6 | — | yes | KEEP | current forensic state |
| diagnosis/nxa-forensic-2026-10-03 | local | 80 | 6 | — | yes (tag) | KEEP | preserved by tag |
| backup-pre-rewrite | local | 33 | 3 | — | yes | DELETE | 3 commits fully represented in feature/rhan-next tree; content preserved |
| dev | local | 0 | 0 | — | yes | DELETE | 0 unique commits; fully merged |
| docs/report | local | 0 | 0 | — | yes | DELETE | 0 unique commits; pure scaffold |
| eyad-pr | local | 0 | 0 | docs-only | yes | DELETE | PR merged into main+frs; 0 unique commits |
| phase/1-rhan | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/1-bagnet | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/1-clip | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/1-shaperesnet | remote | 8 | 8 | — | NO (unique model/checkpoints) | KEEP | unique research artifacts, not archived |
| phase/1-vit | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/2-bagnet-attacks | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/2-efficientnet-attacks | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/2-shaperesnet-attacks | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/2-vit-attacks | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/4-analysis | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/5-sdt | local,remote | 0 | 0 | PR#1 | yes | DELETE | 0 unique commits; PR#1 merged |
| phase/rhan-trades | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/rhan-trades-curriculum | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/rhan-v2 | local,remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/rhan-v3-adaptive | local | 0 | 0 | — | yes (experimented) | DELETE | 0 unique commits |
| phase/rhan-v4 | local | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/rhan-v5 | local | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/rhan-v6 | local | 0 | 0 | — | yes (experimented) | DELETE | 0 unique commits |
| phase/trial-1-clip | local | 0 | 0 | — | yes | DELETE | 0 unique commits |
| phase/trial-2-adaptive | local | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/dev | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/docs/report | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/feature/rhan-next | remote | 0 | 0 | — | yes | DELETE | tracking ref; content in local branch |
| origin/main | remote | 0 | 0 | — | yes | DELETE | tracking ref; content in local main |
| origin/phase/1-bagnet | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/phase/1-clip | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/phase/1-cornets | remote | 0 | 0 | PR#2 | yes | DELETE | 0 unique commits; PR#2 merged |
| origin/phase/1-efficientnet | remote | 0 | 0 | — | yes | DELETE | 0 unique commits; only test img diffs |
| origin/phase/1-rhan | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/phase/rhan-trades | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/phase/rhan-trades-curriculum | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/phase/rhan-v2 | remote | 0 | 0 | — | yes | DELETE | 0 unique commits |
| origin/phase/1-shaperesnet | remote | 8 | 8 | — | NO | KEEP | unique research content |

---

## Phase 13-14: Deletion Results

### Remote branches deleted: 15
- `dev`, `phase/1-rhan`, `phase/rhan-trades`, `phase/rhan-v2`, `phase/rhan-trades-curriculum`, `phase/1-cornets`, `phase/rhan-v3-adaptive`, `phase/rhan-v4`, `phase/rhan-v5`, `phase/rhan-v6`, `phase/trial-1-clip`, `phase/trial-2-adaptive`, `docs/report`, `phase/1-bagnet`, `phase/1-clip`, `phase/1-shaperesnet`, `phase/1-vit`, `phase/2-*`, `phase/4-analysis`, `phase/5-sdt`, `phase/1-efficientnet`

### Local branches deleted: 22
- `backup-pre-rewrite` (forced with -D; content preserved in feature/rhan-next), `dev`, `docs/report`, `eyad-pr`, `phase/1-rhan`, `phase/1-bagnet`, `phase/1-clip`, `phase/1-shaperesnet`, `phase/1-vit`, `phase/2-bagnet-attacks`, `phase/2-efficientnet-attacks`, `phase/2-shaperesnet-attacks`, `phase/2-vit-attacks`, `phase/4-analysis`, `phase/5-sdt`, `phase/rhan-trades-curriculum`, `phase/rhan-v2`, `phase/rhan-v3-adaptive`, `phase/rhan-v4`, `phase/rhan-v5`, `phase/rhan-v6`, `phase/trial-1-clip`, `phase/trial-2-adaptive`, `phase/1-cornets`, `phase/1-efficientnet`

### Tags preserved: 1/1
- `forensic-nxa-2026-10-03-final`

## Final State

- **Local branches:** 4 (main, feature/rhan-next, stage2/nxa-pipeline-refactor, diagnosis/nxa-forensic-2026-10-03)
- **Remote-tracking:** 2 (origin/main, origin/feature/rhan-next)
- **Remote branches:** 2 (main, feature/rhan-next)
- **Tags:** 1
- **Total branches:** 22 deleted out of 48 (before count)

## Pruned Refs

After `git fetch origin --prune`, the only remaining local remote-tracking refs are `origin/main` and `origin/feature/rhan-next`. The history has NOT been rewritten (no `git reset`, `rebase`, `filter-branch`, `gc`, `prune`, or `repack` was run).
