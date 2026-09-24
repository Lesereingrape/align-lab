from __future__ import annotations

import random

import torch

from alignlab.align import METHODS
from alignlab.data import make_pairs, sample_examples
from alignlab.train import build_base, evaluate_pairs, make_dataset, run_method


def _pair_batch(n=16, seed=0):
    rng = random.Random(seed)
    pairs = make_pairs(sample_examples(n, rng, 10, 99), rng)
    prompts = torch.tensor([p.ex.prompt() for p in pairs], dtype=torch.long)
    chosen = torch.tensor([p.chosen for p in pairs], dtype=torch.long)
    rejected = torch.tensor([p.rejected for p in pairs], dtype=torch.long)
    return prompts, chosen, rejected


def test_all_methods_registered():
    assert set(METHODS) == {"sft", "dpo", "orpo", "simpo"}


def test_each_loss_is_finite():
    prompts, chosen, rejected = _pair_batch()
    model = build_base(steps=5, batch=8, lr=1e-3, seed=0,
                       examples=sample_examples(8, random.Random(0), 10, 99))
    ref = build_base(steps=5, batch=8, lr=1e-3, seed=0,
                     examples=sample_examples(8, random.Random(0), 10, 99))
    for fn in METHODS.values():
        loss = fn(model, ref, (prompts, chosen, rejected))
        assert torch.isfinite(loss)
        assert loss.ndim == 0


def test_dpo_increases_preference_margin():
    te, tp, ee, ep = make_dataset(200, 120, 0, 10, 99)
    base = build_base(steps=60, batch=64, lr=2e-3, seed=0, examples=te)
    base_margin = evaluate_pairs(base, ep)["margin"]
    curve = run_method("dpo", base=base, train_pairs=tp, eval_examples=ee,
                       eval_pairs=ep, steps=120, batch=64, lr=1e-3, seed=1,
                       eval_every=120)
    assert curve[-1]["margin"] > base_margin
    assert 0.0 <= curve[-1]["win_rate"] <= 1.0
