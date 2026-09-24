from __future__ import annotations

import torch

from alignlab.data import Example
from alignlab.model import TinyTransformer, count_parameters


def _batch():
    exs = [Example(23, 19), Example(48, 47)]
    prompts = torch.tensor([e.prompt() for e in exs], dtype=torch.long)
    cots = torch.tensor([e.cot() for e in exs], dtype=torch.long)
    return prompts, cots


def test_seq_logprob_is_nonpositive_and_shaped():
    torch.manual_seed(0)
    model = TinyTransformer(d_model=32)
    prompts, cots = _batch()
    lp = model.seq_logprob(prompts, cots)
    assert lp.shape == (2,)
    assert bool((lp <= 0).all())


def test_sft_loss_decreases_with_steps():
    torch.manual_seed(0)
    model = TinyTransformer(d_model=32)
    prompts, cots = _batch()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    start = float(model.sft_loss(prompts, cots))
    for _ in range(60):
        loss = model.sft_loss(prompts, cots)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(model.sft_loss(prompts, cots)) < start


def test_generate_cot_shape():
    torch.manual_seed(0)
    model = TinyTransformer(d_model=32)
    prompts, _ = _batch()
    out = model.generate_cot(prompts, n_tokens=5, greedy=True)
    assert out.shape == (2, 5)


def test_tiny_model_is_small():
    assert count_parameters(TinyTransformer(d_model=64)) < 200_000
