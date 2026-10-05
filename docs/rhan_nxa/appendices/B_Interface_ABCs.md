# Appendix B — Architectural Interface Contracts and ABCs

> *Authoritative Python Abstract Base Class (ABC) definitions and typed interface signatures defining the RHAN-NXA contract boundaries.*

---

## 1. Overview

To guarantee modularity, prevent monolithic coupling, and enable independent subagent ownership, RHAN-NXA relies on strict Python abstract base classes. Any implementation component must satisfy these contracts.

All interfaces are formally defined in [noesis_vision/beliefs/interfaces.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/interfaces.py) and [noesis_vision/predictive_coding/interfaces.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/interfaces.py).

---

## 2. The `BeliefState` Interface

```python
# noesis_vision/beliefs/interfaces.py

from abc import ABC, abstractmethod
from typing import Mapping, Optional
import torch

class BeliefState(ABC):
    """Abstract interface defining the tripartite perceptual belief state B_t."""

    @property
    @abstractmethod
    def z(self) -> torch.Tensor:
        """Global perceptual state vector, shape (B, D_z)."""
        pass

    @property
    @abstractmethod
    def s(self) -> Optional[torch.Tensor]:
        """Optional structural state tensor. Returns None in Gen-1 core."""
        pass

    @property
    @abstractmethod
    def u(self) -> torch.Tensor:
        """Dirichlet epistemic uncertainty scalar per sample, shape (B,)."""
        pass

    @abstractmethod
    def drift_to(self, other: "BeliefState",
                 weights: Optional[Mapping[str, float]] = None,
                 distance: str = "l2") -> torch.Tensor:
        """Computes sample-wise trajectory drift, returning a (B,) tensor.
        
        Must remain differentiable w.r.t. self.z (and other.z if attached).
        """
        pass

    @abstractmethod
    def detached_copy(self) -> "BeliefState":
        """Returns a detached copy of B_t with autograd severed, for logging."""
        pass

    @abstractmethod
    def updated(self, delta_z: torch.Tensor,
                delta_s: Optional[torch.Tensor] = None,
                new_u: Optional[torch.Tensor] = None) -> "BeliefState":
        """Constructs an updated BeliefState immutably."""
        pass
```

---

## 3. The `GlimpseFeaturePredictor` Interface

```python
# noesis_vision/predictive_coding/interfaces.py

from abc import ABC, abstractmethod
import torch

class GlimpseFeaturePredictor(ABC):
    """Shared predictor consumed jointly by prediction error and gaze planning."""

    @abstractmethod
    def forward(self, z_prev: torch.Tensor,
                a_next: torch.Tensor) -> torch.Tensor:
        """Predicts latent patch features at prospective gaze location.
        
        Args:
            z_prev: Prior global belief state, shape (B, D_z).
            a_next: Prospective fixation coordinate, shape (B, 2) in [-1, 1]^2.
            
        Returns:
            Predicted patch token features, shape (B, 16, D_z).
        """
        pass
```

---

## 4. The `UpdateNet` Interface

```python
# noesis_vision/predictive_coding/interfaces.py

from abc import ABC, abstractmethod
import torch

class UpdateNet(ABC):
    """Fuses prior belief with precision-weighted error into a bounded update."""

    DELTA_BOUND: float = 0.1

    @abstractmethod
    def forward(self, z_prev: torch.Tensor,
                error_term: torch.Tensor,
                precision: torch.Tensor) -> torch.Tensor:
        """Computes bounded state delta.
        
        Args:
            z_prev: Prior belief state, shape (B, D_z).
            error_term: Realized prediction error tensor, shape (B, D_z).
            precision: Dynamic precision weighting scalar, shape (B, 1).
            
        Returns:
            delta_z: Bounded update vector, shape (B, D_z), obeying
                     ||delta_z||_inf <= DELTA_BOUND.
        """
        pass
```

---

## 5. The `AISv2GazePolicy` Interface

```python
# noesis_vision/gaze/ais_v2_policy.py

from abc import ABC, abstractmethod
from typing import Tuple
import torch

class AISv2GazePolicy(ABC):
    """Active Information Sampling gaze policy."""

    @abstractmethod
    def select_action(self, belief: BeliefState,
                      history: torch.Tensor,
                      temperature: float = 1.0,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Proposes and selects the next fixation coordinates.
        
        Args:
            belief: Current BeliefState B_t.
            history: Past fixation history A_t, shape (B, t, 2).
            temperature: Softmax sampling temperature.
            deterministic: If True, uses argmax selection.
            
        Returns:
            Tuple of:
              - a_next: Selected fixation coordinate, shape (B, 2).
              - candidate_scores: Evaluated scores for all K candidates, shape (B, K).
        """
        pass
```
