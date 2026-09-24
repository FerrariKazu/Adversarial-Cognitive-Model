"""
Agent D contract test — gradient flow through the evidential head.

The HARD non-detached assertion (the single most repeated Gen-0 failure
mode, checked here from day one): the evidence path carries autograd back
into the head's parameters — no .detach() anywhere in the forward path.
"""
import torch

from noesis_vision.uncertainty.evidential_head import EvidentialHead

B, D, C = 2, 384, 10


def test_evidence_carries_graph():
    torch.manual_seed(0)
    h = EvidentialHead(input_dim=D, num_classes=C)
    x = torch.randn(B, D, requires_grad=True)
    U = h(x)
    assert U.evidence.grad_fn is not None, (
        "evidence must carry autograd — a detached evidence path is the "
        "Gen-0 failure class, forbidden from day one")


def test_backward_reaches_head_params_and_input():
    torch.manual_seed(0)
    h = EvidentialHead(input_dim=D, num_classes=C)
    x = torch.randn(B, D, requires_grad=True)
    U = h(x)
    U.evidence.sum().backward()
    for name, p in h.named_parameters():
        assert p.grad is not None, f"no gradient reached head param {name}"
        assert torch.any(p.grad != 0), f"zero gradient at {name}"
    assert x.grad is not None and torch.any(x.grad != 0)


def test_uncertainty_and_entropy_paths_differentiable():
    """All three consumers' quantities (Pi_t's scalars, AIS's entropy
    deltas) must be differentiable w.r.t. the head (one representation,
    three consumers — Part 1.F)."""
    torch.manual_seed(0)
    h = EvidentialHead(input_dim=D, num_classes=C)
    x = torch.randn(B, D, requires_grad=True)
    U = h(x)
    (U.uncertainty.sum() + U.entropy().sum()).backward()
    assert x.grad is not None and torch.any(x.grad != 0)


def test_no_detach_in_forward_source():
    """Source-level guard: the head's forward never calls .detach() — the
    hard assertion made structural, not just behavioral."""
    import inspect
    from noesis_vision.uncertainty import evidential_head
    src = inspect.getsource(evidential_head)
    forward_body = src[src.find("def forward"):src.find("class ")]
    # Only the class forward bodies count; the module docstring may mention
    # detach in prose, so scan the actual forward method sources.
    for cls in (EvidentialHead,):
        csrc = inspect.getsource(cls)
        fsrc = csrc[csrc.find("def forward"):]
        assert ".detach()" not in fsrc, (
            f"{cls.__name__}.forward must never detach the evidence path")
