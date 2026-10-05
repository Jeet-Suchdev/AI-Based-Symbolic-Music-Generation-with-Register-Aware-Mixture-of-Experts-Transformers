import torch
from musicgen.models.moe import MoEBlock
from musicgen.training.losses import moe_load_balance_loss

def test_moe_shapes_and_grad():
    m = MoEBlock(32, 64, num_experts=3, routing="top2")
    x = torch.randn(2, 8, 32, requires_grad=True)
    y, stats = m(x)
    assert y.shape == x.shape
    loss = y.square().mean() + moe_load_balance_loss(stats["router_probs"], stats["top_indices"])
    loss.backward()
    assert x.grad is not None
