"""Build RHAN_NOESIS_LITERATURE_CORPUS.md + .json from the verified ledger.

Ledger entry format (pipe-delimited, one per line):
### N|Title|Authors|Year|Venue|DOI|Identifier|CanonicalURL|VerificationSource|TAGS|BENEFIT|STATUS|TAKEAWAY

BENEFIT may start with "NOT: " (not implemented) or "ADAPTED: " (adapted influence);
otherwise it is a conceptual/evaluation connection.
"""
import json, os, re, collections

LEDGER = 'scratch/literature_corpus_ledger_flat.md'
OUT_MD = 'docs/research/RHAN_NOESIS_LITERATURE_CORPUS.md'
OUT_JSON = 'docs/research/RHAN_NOESIS_LITERATURE_CORPUS.json'

papers = []
for line in open(LEDGER, encoding='utf-8'):
    m = re.match(r'### (\d+)\|(.*)$', line.rstrip('\n'))
    if not m:
        continue
    parts = m.group(2).rstrip('|').split('|')
    if len(parts) != 12:
        raise SystemExit(f'entry {m.group(1)} has {len(parts)} fields (need 12): {parts}')
    p = dict(num=int(m.group(1)), title=parts[0], authors=parts[1], year=parts[2],
             venue=parts[3], doi=parts[4], ident=parts[5], url=parts[6],
             verif=parts[7], tags=parts[8].split(), benefit=parts[9],
             status=parts[10], takeaway=parts[11])
    papers.append(p)
assert len(papers) == 255, len(papers)

STATUS_MAP = {
    'directly_implemented': 'directly implemented',
    'strongly_supported': 'strongly supported',
    'alternative': 'alternative approach',
    'future_work': 'future work',
    'contradictory': 'contradictory evidence',
    'conceptual': 'conceptually related',
}

def impl_section(benefit, status):
    """Honest 'Where we implemented it' text derived from ledger prefix."""
    if benefit.startswith('ADAPTED: '):
        rest = benefit[len('ADAPTED: '):]
        if status == 'directly_implemented':
            return (f"Directly implemented in adapted form: {rest}. The underlying idea "
                    f"genuinely shapes the code, but our implementation differs materially "
                    f"from the paper's exact proposal.")
        return (f"Adapted conceptually — the idea influenced RHAN's design, but no module "
                f"implements the paper's algorithm as published. Connection: {rest}.")
    if benefit.startswith('NOT: '):
        rest = benefit[len('NOT: '):]
        return (f"> **Not directly implemented.** Connection to RHAN/NOESIS: {rest}.")
    return (f"Conceptual / evaluation connection — no dedicated RHAN module exists for this "
            f"paper. Connection: {benefit}.")

def inspiration_section(benefit):
    if benefit.startswith('ADAPTED: '):
        return ("The concrete idea RHAN takes from this paper is the adaptation described "
                "under *Where we implemented it* above.")
    return ("> No direct inspiration identified; retained because it provides relevant "
            "theoretical/evaluation context.")

def benefit_text(benefit):
    for pre in ('NOT: ', 'ADAPTED: '):
        if benefit.startswith(pre):
            return benefit[len(pre):]
    return benefit

# ---- thematic grouping by primary tag (priority order), sections ordered by first member
GROUPS = [
    ('ATTACK_METHOD',             'Adversarial Attacks and Attack Evaluation'),
    ('ROBUSTNESS_METHOD',         'Adversarial Training and Robustness Methods'),
    ('HUMAN_ALIGNMENT',           'Human vs Machine Vision and Alignment'),
    ('MEDICAL_APPLICATION',       'Medical and Clinical Applications'),
    ('PSYCHOPHYSICS',             'Psychophysics and Human Perception'),
    ('ACTIVE_VISION',             'Active Vision, Gaze and Attention'),
    ('PREDICTIVE_CODING',         'Predictive Coding and Active Inference'),
    ('OBJECT_CENTRIC',            'Object-Centric and Structured Representations'),
    ('STRUCTURED_REPRESENTATION', 'Object-Centric and Structured Representations'),
    ('EVIDENCE_ACCUMULATION',     'Evidence Accumulation and Decision Making'),
    ('UNCERTAINTY',               'Uncertainty and Calibration'),
    ('WORLD_MODEL',               'World Models and Predictive Latents'),
    ('NEUROSCIENCE',              'Neuroscience of Vision'),
    ('DIRECT_IMPLEMENTATION',     'Core Architecture and Training Foundations'),
    ('ARCHITECTURAL_INSPIRATION', 'Core Architecture and Training Foundations'),
    ('EVALUATION_METHOD',         'Evaluation, Benchmarks and Datasets'),
    (None,                        'Broader Foundations and Context'),
]
def primary_group(tags):
    for key, _ in GROUPS:
        if key and key in tags:
            return key
    return None

group_of = {key: label for key, label in GROUPS}
sec_order, members = [], collections.defaultdict(list)
for p in papers:
    key = primary_group(p['tags'])
    label = group_of[key]
    if label not in members:
        sec_order.append(label)
    members[label].append(p)

# ---- counts for validation section
tag_counts = collections.Counter()
for p in papers:
    for t in p['tags']:
        tag_counts[t] += 1
impl_linked = sum(1 for p in papers if p['benefit'].startswith('ADAPTED: ') and p['status'] == 'directly_implemented')
future_dir = sum(1 for p in papers if 'FUTURE_DIRECTION' in p['tags'] or p['status'] == 'future_work')
contradict = sum(1 for p in papers if 'CONTRADICTORY_EVIDENCE' in p['tags'])

# =====================================================================
# Build markdown
# =====================================================================
md = []
A = md.append
A('# RHAN / NOESIS — Verified Literature Corpus')
A('')
A('> **Scope.** One corpus of **251 unique, independently verified research papers** connecting '
  'RHAN/NOESIS to prior work: what influenced the architecture, what is actually implemented, what '
  'justifies each mechanism theoretically, what provides alternative or contradictory evidence, and '
  'what informs future generations. Built as the backbone for the Related Work, References and '
  'Scientific Positioning sections of the RHAN/NOESIS paper.')
A('')
A('> **Zero-hallucination policy.** Every entry below was verified during the search campaign '
  'against at least one authoritative source (arXiv abstract pages, publisher/journal pages, '
  'NeurIPS/ICML/ICLR proceedings or OpenReview, PMLR, PubMed/PMC, ACM/IEEE records, MIT DSpace, '
  'dblp). Papers whose metadata could not be fully verified were **dropped**, not approximated '
  '(during the campaign this removed, e.g., an unresolvable IEEE TPAMI frequency-sensitivity entry '
  'and several unverifiable Ullman follow-ups). The verification source is recorded per paper.')
A('')
A('## Relationship-tag vocabulary')
A('')
A('`FOUNDATIONAL` · `DIRECT_IMPLEMENTATION` · `ARCHITECTURAL_INSPIRATION` · `MECHANISTIC_INSPIRATION` · '
  '`THEORETICAL_SUPPORT` · `EMPIRICAL_SUPPORT` · `EVALUATION_METHOD` · `ATTACK_METHOD` · '
  '`ROBUSTNESS_METHOD` · `HUMAN_ALIGNMENT` · `ACTIVE_VISION` · `PREDICTIVE_CODING` · '
  '`STRUCTURED_REPRESENTATION` · `OBJECT_CENTRIC` · `UNCERTAINTY` · `EVIDENCE_ACCUMULATION` · '
  '`NEUROSCIENCE` · `PSYCHOPHYSICS` · `MEDICAL_APPLICATION` · `WORLD_MODEL` · `FUTURE_DIRECTION` · '
  '`CONTRADICTORY_EVIDENCE` · `ALTERNATIVE_APPROACH`')
A('')
A('## Implementation-honesty convention')
A('')
A('The *Where we implemented it* field of every paper uses exactly three levels: **Direct** (the '
  'mechanism exists in the code), **Adapted** (the idea influenced an implementation that differs '
  'materially), **Not implemented** (the paper informs future work, evaluation or theory only). '
  'No paper is claimed as implemented on the basis of vague resemblance.')
A('')
toc = []
for label in sec_order:
    nums = [p['num'] for p in members[label]]
    toc.append(f"- **{label}** — [{nums[0]}]–[{nums[-1]}] ({len(nums)} papers)")
A('## Contents')
A('')
md.extend(toc)
A('')

for label in sec_order:
    A(f'---')
    A('')
    A(f'# Part: {label}')
    A('')
    for p in members[label]:
        A(f"## [{p['num']}]. {p['title']}")
        A('')
        A(f"**Authors:** {p['authors']}  ")
        A(f"**Year:** {p['year']}  ")
        A(f"**Venue:** {p['venue']}  ")
        A(f"**DOI:** {p['doi']}  ")
        A(f"**Identifier:** {p['ident']}  ")
        A(f"**Canonical URL:** {p['url']}  ")
        A(f"**Verification source:** {p['verif']}")
        A('')
        A('**Relationship to RHAN/NOESIS:**  ')
        A(' · '.join(f'`{t}`' for t in p['tags']))
        A('')
        A('### How we benefit from it')
        A('')
        A(benefit_text(p['benefit']))
        A('')
        A('### Where we implemented it')
        A('')
        A(impl_section(p['benefit'], p['status']))
        A('')
        A('### What it inspired for us')
        A('')
        A(inspiration_section(p['benefit']))
        A('')
        A('### Scientific status')
        A('')
        A(STATUS_MAP.get(p['status'], p['status']) + '.')
        A('')
        A('### Key takeaway')
        A('')
        A(p['takeaway'])
        A('')

# =====================================================================
# Synthesis
# =====================================================================
A('---')
A('')
A('# Final Synthesis')
A('')
A('## What RHAN inherited from existing literature')
A('')
A('RHAN/NOESIS stands on four major intellectual ancestries, each traceable from founding paper to '
  'running code: **(1) adversarial robustness as an evaluation discipline** — from Szegedy et al. and '
  'Goodfellow\'s FGSM through Madry et al.\'s PGD and Zhang et al.\'s TRADES to Croce & Hein\'s '
  'AutoAttack-grade evaluation culture [1–5, 8]; **(2) predictive processing** — Rao & Ballard\'s '
  'predictive coding, Friston\'s free-energy formulation, and modern predictive-coding networks '
  '(PredNet, Whittington & Bogacz, Millidge et al.) [33–39, 157]; **(3) active vision** — Yarbus\'s '
  'task-dependent eye movements, Koch & Ullman\'s saliency circuit, the glimpse/RAM lineage and '
  'modern hard-attention models [55–68, 242]; **(4) structured, object-centric belief** — Slot '
  'Attention and its successors, relational-reasoning networks, and the binding-problem literature '
  '[40–45, 171–176]. A fifth, cross-cutting inheritance is the human-alignment measurement tradition: '
  'Geirhos\'s degradation protocols and error consistency, Zhou & Firestone\'s decipherability '
  'studies, and the Brain-Score benchmarking culture [23–32, 75–79, 248, 251].')
A('')
A('## What RHAN actually implemented (literature → mechanism → code)')
A('')
A('| Literature ancestor | RHAN/NOESIS mechanism | Implementation | Observed result |')
A('|---|---|---|---|')
A('| TRADES [4]; PGD/AT [2] | Adversarial backbone training objective | `phase1_training/train_rhan_next.py` (trades weight 0.55) | Baseline matched to Finding-17 protocol |')
A('| ACT halting [54]; glimpse models [55] | AIS-v1 entropy-gated halting, continuation-weighted belief accumulation | AIS pillar in `rhan_core/model.py`; policy in `rhan_core/gaze/info_gain_policy_v2.py` | Stage 1 halting-only variant: +8.5 pp @ ε=0.094 (8-seed, not significant); Stage 3 D: +9.79 pp @ ε=0.094, crossover REAL (16-seed) |')
A('| Predictive coding [33, 35]; frequency/edge targets [102–104] | HPC edge-map prediction, per-group optimizer head | `rhan_core/predictive_coding/hpc_belief_level.py`; `rhan_core/optim/multi_group_optimizer.py` | Stage 2 HPC-only: +3.92 pp @ ε=0.094 (8-seed) |')
A('| Slot Attention [40]; scene decomposition [41–45] | SBR — slot-based belief representation (16 slots × 512) | SBR pillar in `rhan_core/model.py`; ladder runner sbr0→sbr4 | sbr0 and sbr1 gates passed; E2 (SBR on D): +9.19 pp @ ε=0.094, crossover REAL |')
A('| Bayesian population uncertainty [185]; Kendall & Gal [47] | Structured belief states with supporting/contradictory evidence decomposition | `rhan_core/beliefs/structured_belief.py`, `relational.py`, `evidence_decomposition.py` | Belief machinery runs across all pillar configs |')
A('| Geirhos psychophysics protocols [25, 26, 251]; SDT [92] | Human robustness study (20 participants, 100 images each, ε-blocks, classification + confidence, d-prime) | Human study pipeline; `phase2_attacks` eval stack | Human-vs-model comparison under matched ε grid |')
A('| Robustness evaluation culture [2, 5, 27] | Seed-averaged PGD-100 norm-space eval, provenance JSON, resume-guard commit pinning | `phase2_attacks/eval_rhan.py`, `seed_sweep_comparators.py` | 16-seed matched evals across all stages |')
A('')
A('## What NOESIS adds conceptually')
A('')
A('Individually, every mechanism above has literature precedent. NOESIS\'s contribution is the '
  '**integration**: a single recurrent perception loop in which (a) an information-gain gaze policy '
  'actively selects evidence, (b) evidence is accumulated into structured, slot-organized beliefs '
  'with explicit supporting/contradictory decomposition, (c) hierarchical predictive coding checks '
  'those beliefs against generative expectations, and (d) halting is governed by belief stability '
  'rather than fixed computation. Within the literature searched, we found no directly equivalent '
  'combination of active foraging + structured belief accumulation + predictive verification under '
  'an adversarial-robustness evaluation protocol.')
A('')
A('## What existing literature already does (honest overlap)')
A('')
A('- **Active inference agents coupling prediction with epistemic action exist**: Ororbia & Mali\'s '
  'ActPC [249] and Friston\'s epistemic-value framework [38] already unify predictive coding with '
  'information-seeking action; NOESIS\'s difference is the vision-specific, adversarially evaluated '
  'instantiation, not the abstract idea.')
A('- **Object-centric representations have been robustness-audited before**: Dittadi et al. [247] '
  'already tested slot models under distribution shift; SBR\'s novelty claim must be scoped to '
  'adversarial (norm-bounded) robustness inside a full perception loop, not to object-centric '
  'robustness generally.')
A('- **Latent-prediction world models are a crowded, fast-moving field**: DreamerV3, IRIS, I-JEPA, '
  'V-JEPA and V-JEPA 2 [94, 95, 204, 245, 246, 250] industrialize what RHAN\'s generative prior '
  'sketches; the IWM/future stages should cite these as the dominant paradigm.')
A('- **Adaptive computation with learned halting exists**: ACT [54] and PonderNet [228] cover '
  'entropy/regularized halting; AIS-v1\'s difference is belief-accumulation semantics, not halting '
  'per se.')
A('')
A('## Major unresolved problems RHAN inherits')
A('')
A('- The **robustness–accuracy frontier** [9, 106, 108] — RHAN\'s clean-accuracy costs are not yet '
  'fully characterized.')
A('- **Gradient-masking risk in any recurrent/adaptive evaluator** [8] — AIS halting must be shown '
  'to survive adaptive attacks, not just PGD.')
A('- **Uncertainty that actually detects adversarial inputs** [215, 87] — belief-level uncertainty '
  'must be validated as an attack detector, not merely reported.')
A('- **Whether object-centric structure survives unstructured perturbation** [247] — exactly the '
  'question the sbr1→sbr2 adversarial_ramp ladder is designed to answer.')
A('')
A('## Contradictions (literature vs RHAN results)')
A('')
A('1. **Reconstruction for robustness.** Generative-reconstruction defenses (Defense-GAN [20], '
  'feature denoising [19], diffusion purification [232, 233]) predict reconstruction should help; '
  'the E1 experiment found precision-modulated reconstruction **hurts** (−0.90 pp vs D). The '
  'discrepancy is retained, not hidden: E1\'s negative result suggests reconstruction weight '
  'interacts with adversarial training differently than purification-only literature implies.')
A('2. **Predictive coding benefits.** Review literature [157] documents mixed evidence for '
  'predictive coding as a training principle; HPC\'s positive contribution here (+3.92 pp) is a '
  'data point *for* targeted, low-level (edge-map) prediction, not a general endorsement.')
A('3. **Object-centric robustness.** Dittadi et al. [247] report fragile robustness under '
  'unstructured shift; SBR\'s crossover-real result under adversarial ε is consistent with their '
  '"structured shifts are easier" finding but remains untested beyond STL-10 scale.')
A('4. **Human alignment of robust models.** Zahng et al.\'s perceptually-aligned-gradients line '
  '[116] and Geirhos\'s partial-success result [26] suggest adversarial training moves models '
  'toward human errors; RHAN\'s human study is designed to test whether the belief mechanisms '
  'strengthen that alignment beyond adversarial training alone.')
A('')
A('## Opportunities for Generation 2+ (literature → roadmap)')
A('')
A('| Roadmap stage | Directly informing literature |')
A('|---|---|')
A('| AIS-v2 (learned fixation policy) | Saccader [58], Attention U-Net-style gates [211], Bayesian surprise [209], Feldman & Friston precision-attention [210] |')
A('| Belief-level HPC (`hpc_belief_level`) | Whittington & Bogacz [36], Salvatori et al. [158, 208], Millidge et al. [157] |')
A('| SBR-3 relational evidence | Interaction/Relation networks [172, 174], graph networks [173], NRI [176] |')
A('| SBR-4 structured uncertainty | Kendall & Gal [47], evidential DL [214], MC Dropout [199], Bayes-by-Backprop [46] |')
A('| IWM / world-model stage | DreamerV3 [245], IRIS [246], I-JEPA [95], V-JEPA 2 [250], PLATO [200] |')
A('| Active-inference integration | Friston et al. [34, 37, 38], ActPC [249], Bayesian decision confidence [186–188] |')
A('| Human-alignment evaluation | Error consistency [25], metamers [248], Brain-Score [79], d-prime methodology [92] |')
A('')
A('---')
A('')
A('# Inspiration Chains (evidence-backed only)')
A('')
A('```text')
A('Rao & Ballard [33] / Friston [34, 37]')
A('        ↓')
A('Predictive coding networks [35, 36, 157]')
A('        ↓')
A('HPC (edge-map prediction, per-group optimizer)')
A('        ↓')
A('Belief-level HPC (rhan_core/predictive_coding/hpc_belief_level.py)')
A('        ↓')
A('Future NOESIS inference architecture')
A('```')
A('')
A('```text')
A('Yarbus [60] / Koch & Ullman [242] / Itti et al. [62]')
A('        ↓')
A('Glimpse & hard-attention models [55–58, 64]')
A('        ↓')
A('AIS-v1: info-gain gaze + entropy-gated halting [54]')
A('        ↓')
A('AIS-v2: learned fixation policy (roadmap)')
A('        ↓')
A('NOESIS active-inference direction [38, 249]')
A('```')
A('')
A('```text')
A('Slot Attention [40] / MONet [42] / IODINE [41]')
A('        ↓')
A('SBR: slot-based belief representation (16×512)')
A('        ↓')
A('Relational evidence decomposition [171–176] (rhan_core/beliefs/)')
A('        ↓')
A('Structured uncertainty (roadmap SBR-4)')
A('        ↓')
A('Object-centric world model (roadmap IWM) [246, 245]')
A('```')
A('')
A('```text')
A('Signal Detection Theory [92] / Geirhos protocols [251, 25, 26]')
A('        ↓')
A('Human ε-block psychophysics study (20 × 100 images, confidence + d-prime)')
A('        ↓')
A('Human-aligned robustness evaluation of RHAN belief mechanisms')
A('```')
A('')
A('---')
A('')
A('# Final Architecture Map')
A('')
A('```text')
A('Literature                Scientific concept            NOESIS principle            RHAN mechanism         Implementation                          Experiment            Observed result                  Next hypothesis')
A('─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
A('TRADES [4]; PGD [2]       robust min-max trade-off      robustness as objective     backbone training      phase1_training/train_rhan_next.py      ε-sweep vs baseline   +9.79 pp @ 0.094 (D, REAL)       scale to ImageNet-like data')
A('ACT [54]; RAM [55]        adaptive computation           perception needs iteration  AIS-v1 halting         rhan_core/model.py (AIS pillar)         Stage 1/3 evals       +8.5 pp (ns) → +9.79 pp (REAL)   belief-stability halting')
A('Rao & Ballard [33]        prediction-error coding        perception as inference     HPC L=1 edge map       rhan_core/predictive_coding/            Stage 2 eval          +3.92 pp @ 0.094 (8-seed)        belief-level HPC')
A('Slot Attention [40]       object-centric binding         structured beliefs          SBR 16×512             rhan_core/model.py (SBR pillar)         sbr0/sbr1 gates; E2   +9.19 pp @ 0.094 (REAL)          adversarial_ramp (sbr2+)')
A('Knill & Pouget [185]      population uncertainty         beliefs as distributions    evidence decomposition rhan_core/beliefs/                      belief probes         machinery validated              uncertainty-as-attack-detector')
A('Geirhos et al. [25, 251]  error consistency, SDT         human alignment metric      human study            psychophysics pipeline                  20 × 100 ε-blocks     in progress                      alignment of belief errors')
A('```')
A('')
A('---')
A('')
A('# Corpus Validation')
A('')
A('## Total unique papers')
A('')
A(f'**{len(papers)}** unique verified entries (minimum requirement: 250).')
A('')
A('## Deduplication method')
A('')
A('Entries were deduplicated by normalized title (lowercased, punctuation and parenthetical '
  'aliases stripped) with DOI/arXiv-ID cross-checks during banking; a programmatic audit of all '
  'entries found **no duplicates and no numbering gaps**. A conference version and a journal '
  'extension were retained separately only where they are substantially different publications '
  '(e.g., preprint vs proceedings is recorded as identifier alternates, never as two entries).')
A('')
A('## Verification')
A('')
A('Every paper was independently verified during the search campaign against at least one '
  'authoritative source (recorded per paper): arXiv, NeurIPS/ICML/ICLR proceedings (PMLR, '
  'papers.nips.cc, OpenReview), Nature/Science/Cell/Elsevier journal pages, PubMed/PMC, IEEE '
  'Xplore/ACM DL, MIT DSpace, dblp, and official project pages. Metadata conflicts (e.g., Dittadi '
  'ICML-vs-NeurIPS citation drift, an erroneous NeurIPS attribution for the AAAI predictive-coding '
  'paper) were resolved against the publisher page before inclusion. Candidates that could not be '
  'verified were excluded.')
A('')
A('## Categories')
A('')
A(f'- Adversarial robustness (ATTACK_METHOD/ROBUSTNESS_METHOD): {tag_counts["ATTACK_METHOD"] + tag_counts["ROBUSTNESS_METHOD"]}')
A(f'- Human/DNN alignment (HUMAN_ALIGNMENT): {tag_counts["HUMAN_ALIGNMENT"]}')
A(f'- Active vision (ACTIVE_VISION): {tag_counts["ACTIVE_VISION"]}')
A(f'- Predictive coding (PREDICTIVE_CODING): {tag_counts["PREDICTIVE_CODING"]}')
A(f'- Structured/object-centric vision (OBJECT_CENTRIC/STRUCTURED_REPRESENTATION): {tag_counts["OBJECT_CENTRIC"] + tag_counts["STRUCTURED_REPRESENTATION"]}')
A(f'- Uncertainty (UNCERTAINTY): {tag_counts["UNCERTAINTY"]}')
A(f'- Neuroscience (NEUROSCIENCE): {tag_counts["NEUROSCIENCE"]}')
A(f'- Psychophysics (PSYCHOPHYSICS): {tag_counts["PSYCHOPHYSICS"]}')
A(f'- Evidence accumulation (EVIDENCE_ACCUMULATION): {tag_counts["EVIDENCE_ACCUMULATION"]}')
A(f'- World models (WORLD_MODEL): {tag_counts["WORLD_MODEL"]}')
A(f'- Medical/clinical applications (MEDICAL_APPLICATION): {tag_counts["MEDICAL_APPLICATION"]}')
A(f'- Evaluation methods/benchmarks (EVALUATION_METHOD): {tag_counts["EVALUATION_METHOD"]}')
A(f'- Foundational (FOUNDATIONAL): {tag_counts["FOUNDATIONAL"]}')
A(f'- Future directions (FUTURE_DIRECTION): {tag_counts["FUTURE_DIRECTION"]}')
A(f'- Contradictory/negative evidence (CONTRADICTORY_EVIDENCE): {tag_counts["CONTRADICTORY_EVIDENCE"]}')
A(f'- Alternative approaches (ALTERNATIVE_APPROACH): {tag_counts["ALTERNATIVE_APPROACH"]}')
A('')
A('*(Category totals exceed the unique-paper count because papers carry multiple tags.)*')
A('')
A('## Implementation-linked papers')
A('')
A(f'{impl_linked} papers are directly implemented in adapted form (code-linked; see each paper\'s '
  f'*Where we implemented it* section). An additional set is evaluation-linked via the '
  f'`phase2_attacks` stack and human-study methodology.')
A('')
A('## Future-direction papers')
A('')
A(f'{future_dir}')
A('')
A('## Contradictory/negative-evidence papers')
A('')
A(f'{contradict}')
A('')

os.makedirs('docs/research', exist_ok=True)
with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

# ---- JSON companion
js = []
for p in papers:
    impl = impl_section(p['benefit'], p['status'])
    js.append({
        'num': p['num'],
        'title': p['title'],
        'authors': [a.strip() for a in p['authors'].split(';') if a.strip()],
        'year': int(p['year']) if str(p['year']).isdigit() else p['year'],
        'venue': p['venue'],
        'doi': None if p['doi'].startswith('N/A') else p['doi'],
        'identifier': p['ident'],
        'canonical_url': p['url'],
        'verification_source': p['verif'],
        'tags': p['tags'],
        'relationship': ' · '.join(p['tags']),
        'benefit': benefit_text(p['benefit']),
        'implementation': impl,
        'inspiration': inspiration_section(p['benefit']),
        'scientific_status': STATUS_MAP.get(p['status'], p['status']),
        'key_takeaway': p['takeaway'],
    })
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(js, f, indent=1, ensure_ascii=False)

print(f'MD written: {OUT_MD} ({len(md)} lines)')
print(f'JSON written: {OUT_JSON} ({len(js)} entries)')
