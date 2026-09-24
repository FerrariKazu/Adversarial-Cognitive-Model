"""
Agent E contract test — precision g(U_t) -> Pi_t and its OWN optimizer
group (never lumped with the backbone — the Stage-2 starvation lesson).
"""
import torch

from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
from noesis_vision.predictive_coding.glimpse_predictor import ConcreteGlimpseFeaturePredictor
from noesis_vision.predictive_coding.precision import PrecisionFunction
from noesis_vision.predictive_coding.update_net import ConcreteUpdateNet
from noesis_vision.uncertainty.evidential_head import EvidentialHead

B, D, C = 2, 384, 10
AUX_MULT = 6.67  # the Gen-0 Stage-2 auxiliary-head convention (documented precedent)


def _U():
    torch.manual_seed(0)
    return EvidentialHead(input_dim=D, num_classes=C)(torch.randn(B, D))


def test_positive_learned_differentiable():
    g = PrecisionFunction()
    U = _U()
    Pi = g(U)
    assert Pi.shape == (B,)
    assert bool((Pi > 0).all()), "precision must be strictly positive"
    assert Pi.grad_fn is not None, (
        "the default path is the TRAINING path — non-detached (the "
        "detached=True switch exists for diagnostics only)")
    Pi.sum().backward()
    for name, p in g.named_parameters():
        assert p.grad is not None, f"no gradient reached precision param {name}"


def test_high_uncertainty_path_exists():
    """g is a LEARNED function: it can express low precision at high
    uncertainty (the intended direction is verified empirically by Agent
    H's staged diagnostics — NOT hard-coded here; this test only checks
    the function responds to its input)."""
    from noesis_vision.uncertainty.evidential_head import DirichletParams
    g = PrecisionFunction()
    U_hi, U_lo = _U(), _U()
    with torch.no_grad():
        hi = DirichletParams(evidence=U_hi.evidence.detach())
        lo = DirichletParams(evidence=U_lo.evidence.detach() * 50.0)
    assert not torch.allclose(g(hi), g(lo)), (
        "g must respond to the uncertainty input")


def _registry_with_all_components():
    """The Agent E contract's isolation requirement: g, UpdateNet, and the
    GlimpseFeaturePredictor EACH get their OWN optimizer group."""
    g = PrecisionFunction()
    net = ConcreteUpdateNet(d_z=D)
    pred = ConcreteGlimpseFeaturePredictor(d_z=D)
    backbone_param = torch.nn.Parameter(torch.zeros(4, 4))

    reg = OptimizerGroupRegistry()
    reg.register_backbone([backbone_param])
    reg.register("precision", list(g.parameters()), lr_multiplier=AUX_MULT)
    reg.register("update_net", list(net.parameters()), lr_multiplier=AUX_MULT)
    reg.register("predictor", list(pred.parameters()), lr_multiplier=AUX_MULT)
    return reg, g, net, pred, backbone_param


def test_each_component_own_group():
    reg, g, net, pred, backbone_param = _registry_with_all_components()
    assert reg.group_names == ["backbone", "precision", "update_net", "predictor"]
    for name, module in (("precision", g), ("update_net", net),
                         ("predictor", pred)):
        group_ids = {id(p) for p in reg.group(name)["params"]}
        module_ids = {id(p) for p in module.parameters()}
        assert group_ids == module_ids, f"{name}'s group != its parameters"
        assert id(backbone_param) not in group_ids, (
            f"{name} lumped with the backbone — the Stage-2 starvation "
            f"anti-pattern")
    # lr multiplier semantics flow through the registry exactly.
    opt = reg.build_optimizer(0.003)
    lrs = {pg["name"]: pg["lr"] for pg in opt.param_groups}
    assert lrs["backbone"] == 0.003
    assert lrs["precision"] == 0.003 * AUX_MULT


def test_group_isolation_is_behavioral_not_just_bookkeeping():
    """A loss through precision ONLY gives gradients to the precision
    group — no backbone, no cross-group leakage."""
    reg, g, net, pred, backbone_param = _registry_with_all_components()
    opt = reg.build_optimizer(0.003)
    Pi = g(_U())
    Pi.mean().backward()
    opt.step()
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in g.parameters())
    assert backbone_param.grad is None, (
        "backbone must receive nothing from a precision-only loss")
    assert all(p.grad is None for p in net.parameters())
