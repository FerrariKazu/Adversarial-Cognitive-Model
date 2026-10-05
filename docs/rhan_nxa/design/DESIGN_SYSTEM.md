# RHAN-NXA Technical Handbook: Publication Design System

**Document Version:** 2.0.0  
**Status:** Canonical Publication Standard  
**Authority:** Architectural & Editorial Board  

---

## 1. Design Philosophy & Aesthetic Identity

The RHAN-NXA Technical Handbook is engineered to read as a **scientific research monograph, theoretical field manual, and working laboratory notebook**. It avoids both cold corporate sterility and disposable AI marketing templates.

### The Aesthetic Target
* **Intellectual Rigor + Subtle Handmade Warmth**: Pristine typography, balanced whitespace, and clear mathematical notation paired with organic annotation accents, structured cards, and authentic experimental discipline.
* **Epistemic Honesty**: Visual language mirrors scientific certainty. Verified facts, unconfounded mechanisms, and open hypotheses carry distinct, non-subjective visual treatments.
* **Anti-Patterns Strictly Prohibited**:
  - No generic AI-generated graphics (no floating translucent neural brains, no neon circuit traces, no glowing spheres).
  - No continuous beige or parchment skeuomorphism.
  - No novelty or handwriting body fonts (handwriting accents are restricted exclusively to margin notes and diagram sketches).
  - No flat uniform cards across 100+ pages.

---

## 2. Color Palette & Design Tokens

Colors serve functional semantic roles rather than mere decoration.

| Token | Hex Value | Semantic Role / Application |
|---|---|---|
| `--paper` | `#FAF9F5` | Off-white paper background for cover, callouts, and canvas plates |
| `--surface` | `#FFFFFF` | Primary page background and content container |
| `--surface-subtle` | `#F8FAFC` | Subtle panel backgrounds, table alternating rows, code blocks |
| `--ink` | `#0F172A` | Primary typography, deep slate ink for high-contrast legibility |
| `--muted-ink` | `#475569` | Secondary text, lead paragraph sub-headers, table headings |
| `--faint-ink` | `#94A3B8` | Running headers, page numbers, grid lines, borders |
| `--rule-color` | `#E2E8F0` | Subtle architectural dividing lines |
| `--warm-accent` | `#D97706` | Research notes, theoretical intuition, historical context (amber/ochre) |
| `--structure-accent` | `#0284C7` | Structural belief representations ($z_t$, latent manifolds) (sky blue) |
| `--uncertainty-accent`| `#7C3AED` | Evidential uncertainty ($U_t$, Dirichlet distributions) (violet) |
| `--error-accent` | `#DC2626` | Latent prediction error ($E_t$), confounds, negative results (crimson) |
| `--attention-accent` | `#0D9488` | Active foveation ($a_t$, AIS-v2, gaze trajectories) (teal) |
| `--success-accent` | `#16A34A` | Validated empirical bounds, passing unit invariants (emerald) |

---

## 3. Typographic Hierarchy

The typography pairs an authoritative, highly legible serif or modern humanist sans body with precise technical sans headings and crisp monospaced code.

* **Primary Body Font**: *Bitstream Charter* or *Liberation Serif* (10pt, 1.52 line-height). Clean, robust editorial rhythm.
* **Technical & Heading Font**: *Liberation Sans* / *DejaVu Sans* (Clean geometric sans with strong weights).
* **Code & Tensor Font**: *Liberation Mono* / *DejaVu Sans Mono* (8.5pt with subtle syntax tints).

### Typographic Scale

| Element | Size | Weight | Leading | Style / Tracking |
|---|---|---|---|---|
| **Cover Title** | 34pt | 800 | 1.15 | Liberation Sans, -0.02em |
| **Part Title** | 26pt | 800 | 1.20 | Liberation Sans, uppercase tracking |
| **Chapter Title** | 20pt | 800 | 1.25 | Liberation Sans, bold |
| **Chapter Subtitle** | 11.5pt | 500 (Italic) | 1.40 | Liberation Serif, muted-ink |
| **Section Heading (H2)**| 13.5pt | 700 | 1.30 | Liberation Sans, small caps or bold |
| **Subsection (H3)**| 11.5pt | 700 | 1.30 | Liberation Sans |
| **Lead Paragraph** | 10.5pt | 400 | 1.60 | First paragraph after chapter head |
| **Standard Body** | 9.5pt | 400 | 1.52 | Justified with hyphenation |
| **Captions** | 8.5pt | 500 | 1.35 | Muted-ink, centered or left-aligned |
| **Sidenotes / Margins**| 8.0pt | 400 | 1.35 | Amber/slate accent |
| **Headers & Footers** | 8.0pt | 600 | 1.00 | Uppercase, 1.5px tracking |

---

## 4. Mathematical Notation Standard

The handbook enforces uniform mathematical notation across all 27 chapters and appendices.

### The Canonical 5-Tuple Belief State
The central data structure of RHAN-NXA is the 5-tuple:
$$B_t = (z_t, S_t, U_t, E_t, A_t)$$

1. **$z_t \in \mathbb{R}^{B \times D_z}$**: Holistic visual belief vector ($D_z = 384$). Updated recurrently via UpdateNet:
   $$z_{t+1} = z_t + \tanh(\Delta z_t), \quad \|\Delta z_t\|_\infty \le \delta_{\max}$$
2. **$S_t \in \mathbb{R}^{B \times K \times D_s} \cup \{\text{None}\}$**: Discrete object-slot structural state ($S_t = \text{None}$ strictly enforced in Gen-1 Core via None-propagation).
3. **$U_t \in [0, 1]$**: Epistemic uncertainty scalar computed from Dirichlet evidence parameters:
   $$e_{t,c} \ge 0, \quad \alpha_{t,c} = e_{t,c} + 1, \quad S = \sum_{c=1}^C \alpha_{t,c}, \quad U_t = \frac{C}{S}$$
4. **$E_t \in \mathbb{R}^{B \times N \times D_{\text{feat}}}$**: Latent prediction error between top-down expectation and bottom-up foveal observation ($E_0 := 0$ locked boundary invariant):
   $$E_t = F_{\text{obs}}(g_t) - \hat{F}(z_{t-1}, a_t)$$
5. **$A_t = (a_1, a_2, \dots, a_t)$**: Trajectory of foveal gaze coordinates ($a_t \in [-1, 1]^2$):
   $$A_{t+1} = A_t \cup \{a_{t+1}\}$$

---

## 5. Editorial Chapter Openings

Every chapter begins with an editorial opening layout that sets the conceptual cadence:
1. **Chapter Number Pill**: Clean, oversized sans numeral (`04`, `12`).
2. **Title & Conceptual Subtitle**: Clear semantic labeling.
3. **Conceptual Anchor / Pull Quote**: A 1-2 sentence core thesis in italics.
4. **Thematic Mini-Diagram or Rule**: A concise graphic or rule separating the heading from prose.
5. **Lead Paragraph**: Sets the fundamental problem before diving into mathematical formulations.

---

## 6. Structural Part Openings

The 27 chapters are organized into 6 coherent Parts and Appendices:
* **PART I: Foundations & The Robustness Gap** (Chapters 00–03)
* **PART II: The Perceptual Belief State** (Chapters 04–08)
* **PART III: Active Perception & Foveation** (Chapters 09–12)
* **PART IV: Architectural Substrate & Gradient Flow** (Chapters 13–15)
* **PART V: The Training System & Experimental Record** (Chapters 16–20)
* **PART VI: Synthesis, Epistemic Decisions & Future Frontiers** (Chapters 21–26)
* **APPENDICES: Tensor Reference, Interface ABCs & Port Specifications** (Appendices A–C)

Each Part begins with a dedicated opening page featuring a part synopsis, chapter index, and overarching conceptual theme.

---

## 7. The Callout System

Callouts are stylized semantic sidebars:

* **CORE IDEA** (`--attention-accent` / teal): Foundational architectural principles.
* **WHY IT MATTERS** (`--warm-accent` / amber): Scientific implications and rationale.
* **IMPORTANT DISTINCTION** (Dual-column or split comparison card): Disentangling easily confused concepts ($\theta$ vs $h_t$ vs $B_t$; targeted mechanism vs system-level claim).
* **IMPLEMENTATION NOTE** (`--ink` / slate): Concrete PyTorch code, boundary assertions, and shape invariants.
* **EXPERIMENTAL STATUS** (Categorical badges): Epistemic classification of mechanisms.
* **WARNING / CONFOUND ALERT** (`--error-accent` / crimson): Critical caveats, known bugs, and experimental traps.

---

## 8. Epistemic Decision System (Status Badges)

Status badges use distinct colors, borders, and symbols:

| Status | Symbol | Border / Fill | Scientific Meaning |
|---|---|---|---|
| **LOCKED** | `[● LOCKED]` | Solid Blue (`#0284C7`) | Non-negotiable architectural invariant |
| **REQUIRED** | `[▲ REQUIRED]` | Solid Amber (`#D97706`) | Mandatory component for Gen-1 operation |
| **UNKNOWN** | `[? UNKNOWN]` | Dashed Purple (`#7C3AED`) | Unresolved due to confounds or pending runs |
| **CANDIDATE** | `[◆ CANDIDATE]`| Dotted Teal (`#0D9488`) | Planned experimental ablation branch |
| **DEFERRED** | `[○ DEFERRED]` | Light Grey (`#94A3B8`) | Postponed to Gen-2 or post-core phases |
| **REJECTED** | `[✕ REJECTED]` | Solid Red (`#DC2626`) | Falsified or empirically invalid mechanism |

---

## 9. Visual Rhythm & Figure Composition

Figures are editorial objects with strict spatial relationships:
1. **Full-Width Hero Figures**: Spanning the full text measure (160–170mm). Used for complete perceptual loops, architecture overviews, and system diagrams.
2. **Column Figures & Mini-Plates**: Sized to 80–110mm, anchored with descriptive, multi-sentence captions explaining *how to read the figure*.
3. **Figure Gallery & Previews**: Every SVG must be previewed via PNG rasterization at 300 DPI to verify text bounds, contrast, and alignment before publication.
