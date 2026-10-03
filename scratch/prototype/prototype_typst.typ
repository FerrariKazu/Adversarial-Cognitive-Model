
#set document(title: "RHAN-NXA Technical Handbook Prototype", author: "RHAN-NXA Architectural Team")
#set page(
  paper: "a4",
  margin: (top: 22mm, bottom: 22mm, left: 18mm, right: 18mm),
  header: locate(loc => {
    if loc.page() == 1 or loc.page() == 2 { return none }
    grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 7.5pt, font: "Liberation Sans", weight: "bold", fill: rgb("#94a3b8"), tracking: 1.5pt)[RHAN-NXA TECHNICAL HANDBOOK]],
      align(right)[#text(size: 7.5pt, font: "Liberation Sans", weight: "medium", fill: rgb("#94a3b8"))[PART II: THE PERCEPTUAL STATE]]
    )
  }),
  footer: locate(loc => {
    if loc.page() == 1 or loc.page() == 2 { return none }
    grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 7.5pt, font: "Liberation Sans", fill: rgb("#94a3b8"))[GEN-1 CORE SPECIFICATION]],
      align(right)[#text(size: 9pt, font: "Liberation Sans", weight: "bold", fill: rgb("#0f172a"))[#loc.page()]]
    )
  })
)

#set text(font: "Bitstream Charter", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.7em)

// Cover Page
#place(top + left, dx: -18mm, dy: -22mm,
  rect(
    width: 210mm,
    height: 297mm,
    fill: gradient.linear(rgb("#090d16"), rgb("#0f172a"), rgb("#1e293b"), angle: 145deg),
    inset: (x: 25mm, y: 35mm),
    [
      #v(30mm)
      #text(font: "Liberation Sans", size: 9pt, weight: "bold", tracking: 2pt, fill: rgb("#38bdf8"))[RESEARCH MONOGRAPH & TECHNICAL MANUAL]
      #v(10mm)
      #text(font: "Liberation Sans", size: 34pt, weight: "bold", fill: white, tracking: -0.5pt)[RHAN-NXA\ Technical Handbook]
      #v(8mm)
      #text(font: "Bitstream Charter", size: 14pt, style: "italic", fill: rgb("#94a3b8"))[Active Foveated Inference, Latent Prediction Errors, and Recurrent Belief Dynamics in Adversarial Machine Perception]
      
      #v(70mm)
      #line(length: 100%, stroke: 0.5pt + rgb("#334155"))
      #v(6mm)
      #grid(
        columns: (1fr, 1fr),
        [
          #text(font: "Liberation Sans", size: 9pt, fill: rgb("#94a3b8"))[
            #strong[Architecture:] Generation-1 Visual Core\
            #strong[Epistemic Framework:] Dirichlet Evidential Belief\
            #strong[Model Scale:] ~23.3M Active Parameters
          ]
        ],
        [
          #text(font: "Liberation Sans", size: 9pt, fill: rgb("#94a3b8"))[
            #strong[Status:] Pre-Registration Specification\
            #strong[Document Revision:] 2.0 (Redesigned Edition)\
            #strong[Date:] 2026 Edition
          ]
        ]
      )
    ]
  )
)

#pagebreak()

// Part Break Page
#place(top + left, dx: -18mm, dy: -22mm,
  rect(
    width: 210mm,
    height: 297mm,
    fill: rgb("#0f172a"),
    inset: (x: 30mm, y: 50mm),
    [
      #v(50mm)
      #text(font: "Liberation Sans", size: 14pt, weight: "bold", tracking: 3pt, fill: rgb("#38bdf8"))[PART II]
      #v(8mm)
      #text(font: "Liberation Sans", size: 32pt, weight: "bold", fill: white)[The Perceptual\ Belief State]
      #v(12mm)
      #text(font: "Bitstream Charter", size: 12pt, style: "italic", fill: rgb("#cbd5e1"))[
        Mathematical formalization of the 5-tuple state space, None-propagating structural contracts, closed-form evidential uncertainty, and trajectory drift measurement.
      ]
    ]
  )
)

#pagebreak()

// Chapter 04 Opening
#box(
  width: 100%,
  stroke: (bottom: 2pt + rgb("#0f172a")),
  inset: (bottom: 16pt),
  [
    #rect(fill: rgb("#0f172a"), radius: 4pt, inset: (x: 8pt, y: 4pt))[
      #text(font: "Liberation Sans", size: 10pt, weight: "bold", fill: white)[CHAPTER 04]
    ]
    #v(4pt)
    #text(font: "Liberation Sans", size: 22pt, weight: "bold", fill: rgb("#0f172a"))[The Perceptual Belief State]
    #v(2pt)
    #text(font: "Bitstream Charter", size: 11.5pt, style: "italic", fill: rgb("#475569"))[Formalization of the 5-Tuple Belief Container $B_t = (z_t, S_t, U_t, E_t, A_t)$]
    #v(8pt)
    #rect(fill: rgb("#fffbeb"), stroke: (left: 3pt + rgb("#d97706")), width: 100%, inset: (x: 10pt, y: 8pt))[
      #text(font: "Bitstream Charter", size: 9.5pt, style: "italic", fill: rgb("#92400e"))[
        "An intelligent observer does not simply compute features; it maintains an explicit hypothesis about the external scene that is refined, bounded, and verified over time."
      ]
    ]
  ]
)

#v(8pt)

#text(size: 10.5pt, weight: "medium")[
In standard feedforward computer vision architectures, internal representations are ephemeral activations that vanish once the network produces its final logits. In contrast, RHAN-NXA organizes computation around an explicit, stateful, and interpretable perceptual belief container.
]

#v(6pt)

#align(center)[
  #rect(fill: rgb("#faf9f5"), stroke: 1pt + rgb("#e2e8f0"), radius: 6pt, inset: (x: 20pt, y: 14pt), width: 100%)[
    #text(size: 13pt, weight: "bold")[$ B_t = (z_t, S_t, U_t, E_t, A_t) $]
    #v(4pt)
    #text(font: "Liberation Sans", size: 8.5pt, fill: rgb("#475569"))[
      [Canonical 5-Tuple Belief State #sym.bullet $S_t = "None"$ in Gen-1 Core #sym.bullet $E_0 := 0$ LOCKED]
    ]
  ]
]

#v(6pt)

The five subcomponents are strictly typed, sample-wise differentiable where required, and immutable across recurrent timesteps. The global vector $z_t in RR^(B times 384)$ captures holistic scene semantics; $S_t$ reserves an interface for discrete slot graphs; $U_t in [0, 1]$ quantifies epistemic ignorance; $E_t in RR^(B times 16 times 384)$ measures latent prediction discrepancy; and $A_t$ preserves the spatial gaze fixation history trajectory.

#v(6pt)

#rect(fill: rgb("#fffbeb"), stroke: (left: 4pt + rgb("#d97706"), rest: 1pt + rgb("#fcd34d")), radius: (right: 6pt), inset: 12pt, width: 100%)[
  #text(font: "Liberation Sans", size: 9pt, weight: "bold", fill: rgb("#0f172a"))[IMPORTANT DISTINCTION: $theta$ VS $h_t$ VS $B_t$]
  #v(4pt)
  #text(size: 9pt)[
    #strong[Parametric Weights ($theta$):] Static, learned filter weights stored in checkpoints.\
    #strong[Working Memory ($h_t$):] Transient hidden states in attention layers that discard after processing.\
    #strong[Perceptual Belief ($B_t$):] The explicit, persistent hypothesis about the scene that survives across foveal saccades and governs active information sampling.
  ]
]

#pagebreak()

// Figure Page
#align(center)[
  #image("../../docs/rhan_nxa/figures/belief_state.svg", width: 95%)
]
#v(-6pt)
#text(font: "Liberation Sans", size: 8.5pt, fill: rgb("#475569"))[
  #strong[Figure 2: The Canonical Belief State Container $B_t = (z_t, S_t, U_t, E_t, A_t)$.] Concrete layout of the immutable container. In Generation-1 Core, the structural slot state $S_t$ evaluates strictly to None under the None-propagating rule, while $E_0 := 0$ establishes a locked zero-discrepancy boundary condition for the first glimpse.
]

#v(10pt)

#rect(fill: rgb("#f0fdf4"), stroke: (left: 4pt + rgb("#16a34a"), rest: 1pt + rgb("#bbf7d0")), radius: (right: 6pt), inset: 12pt, width: 100%)[
  #text(font: "Liberation Sans", size: 9pt, weight: "bold", fill: rgb("#166534"))[CORE IDEA: THE NONE-PROPAGATING CONTRACT]
  #v(4pt)
  #text(size: 9pt)[
    To ensure architectural safety during phased research, every member function of `BeliefState` must execute validly when `S_t is None`. No component is permitted to crash, raise unhandled attribute exceptions, or silently coerce `None` to a zero tensor.
  ]
]

#pagebreak()

// Table and Code Page
#text(font: "Liberation Sans", size: 13pt, weight: "bold")[Epistemic Decision Records]
#v(2pt)
#text(size: 9pt, fill: rgb("#475569"))[Every proposed architectural mechanism in RHAN-NXA is governed by an explicit epistemic status badge to eliminate subjectivity.]

#v(8pt)

#table(
  columns: (1.5fr, 1.2fr, 1fr, 2.5fr),
  fill: (x, y) => if y == 0 { rgb("#0f172a") } else if calc.even(y) { rgb("#f8fafc") } else { white },
  stroke: 0.5pt + rgb("#e2e8f0"),
  inset: 7pt,
  align: (col, row) => if row == 0 { left + horizon } else { left + horizon },
  [#text(font: "Liberation Sans", weight: "bold", fill: white, size: 8.5pt)[Mechanism]],
  [#text(font: "Liberation Sans", weight: "bold", fill: white, size: 8.5pt)[Status]],
  [#text(font: "Liberation Sans", weight: "bold", fill: white, size: 8.5pt)[Component]],
  [#text(font: "Liberation Sans", weight: "bold", fill: white, size: 8.5pt)[Scientific Evidence / Justification]],
  
  [Recurrent CompactViT],
  [#rect(fill: rgb("#e0f2fe"), stroke: 1pt + rgb("#bae6fd"), radius: 3pt, inset: (x: 5pt, y: 2pt))[#text(font: "Liberation Sans", size: 7.5pt, weight: "bold", fill: rgb("#0284c7"))[● LOCKED]]],
  [Backbone],
  [23.3M parameter tied within-glimpse visual trunk.],

  [UpdateNet Discrepancy],
  [#rect(fill: rgb("#fef3c7"), stroke: 1pt + rgb("#fde68a"), radius: 3pt, inset: (x: 5pt, y: 2pt))[#text(font: "Liberation Sans", size: 7.5pt, weight: "bold", fill: rgb("#d97706"))[▲ REQUIRED]]],
  [Predictive Coding],
  [Bounded belief revision via tanh clamping ($delta_max = 0.1$).],

  [AIS-v2 Contribution],
  [#rect(fill: rgb("#ede9fe"), stroke: (paint: rgb("#c4b5fd"), dash: "dashed"), radius: 3pt, inset: (x: 5pt, y: 2pt))[#text(font: "Liberation Sans", size: 7.5pt, weight: "bold", fill: rgb("#7c3aed"))[? UNKNOWN]]],
  [Active Sampling],
  [Confounded by legacy `_nx_trainer` SBR flag in 16-seed Gen-0 runs.],

  [Pixel Reconstruction],
  [#rect(fill: rgb("#fee2e2"), stroke: 1pt + rgb("#fca5a5"), radius: 3pt, inset: (x: 5pt, y: 2pt))[#text(font: "Liberation Sans", size: 7.5pt, weight: "bold", fill: rgb("#dc2626"))[✕ REJECTED]]],
  [Loss Objective],
  [Texture bias without robustness gain; eliminated from Gen-1.]
)

#v(14pt)

#text(font: "Liberation Sans", size: 12pt, weight: "bold")[Implementation: VectorBeliefState Invariants]

#v(4pt)

```python
@dataclass(frozen=True)
class VectorBeliefState(BeliefState):
    z: torch.Tensor          # (B, D_z) - Holistic scene latent
    U: DirichletParams       # (B, C)   - Epistemic evidence
    E: torch.Tensor          # (B, N, D_feat) - Prediction discrepancy
    A: GazeState             # Trajectory history list of (B, 2)
    S: Optional[torch.Tensor] = None # None in Gen-1 Core

    def __post_init__(self):
        assert self.z.dim() == 2, f"z must be (B, D_z), got {self.z.shape}"
        assert self.E.dim() == 3, f"E must be (B, N, D_feat), got {self.E.shape}"
        # LOCKED Boundary condition: First glimpse has zero discrepancy
        if self.A.current_glimpse_idx == 0:
            assert torch.all(self.E == 0), "E_0 := 0 invariant violated!"
```
