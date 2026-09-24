"""Training + evaluation harness: one weak base, four alignment runs from it.

Every method starts from the *same* SFT'd base and sees the *same* preference
pairs, so differences in the curves are attributable to the objective alone. We
track three held-out numbers over training steps:

  * answer_acc  - greedy-generated chain verified against a+b (the real skill),
  * win_rate    - fraction of held-out pairs the model scores chosen > rejected,
  * margin      - mean sum-logprob(chosen) - sum-logprob(rejected).

answer_acc is measured on the model's own generations, never on teacher forcing.
"""

from __future__ import annotations

import copy
import random

import torch

from .align import METHODS
from .data import Example, PreferencePair, score_solution
from .model import TinyTransformer


def _pair_tensors(pairs: list[PreferencePair]):
    prompts = torch.tensor([p.ex.prompt() for p in pairs], dtype=torch.long)
    chosen = torch.tensor([p.chosen for p in pairs], dtype=torch.long)
    rejected = torch.tensor([p.rejected for p in pairs], dtype=torch.long)
    return prompts, chosen, rejected


def _example_tensors(examples: list[Example]):
    prompts = torch.tensor([e.prompt() for e in examples], dtype=torch.long)
    cots = torch.tensor([e.cot() for e in examples], dtype=torch.long)
    return prompts, cots


def build_base(*, steps: int, batch: int, lr: float, seed: int,
               examples: list[Example], d_model: int = 64) -> TinyTransformer:
    """A *deliberately imperfect* base: SFT on the gold chains for few steps."""
    torch.manual_seed(seed)
    model = TinyTransformer(d_model=d_model)
    prompts, cots = _example_tensors(examples)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = prompts.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        loss = model.sft_loss(prompts[idx], cots[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def evaluate_answer(model: TinyTransformer, examples: list[Example],
                    *, batch: int = 256) -> dict:
    model.eval()
    prompts, _ = _example_tensors(examples)
    n = prompts.shape[0]
    ok = 0
    for start in range(0, n, batch):
        chunk = prompts[start : start + batch]
        cots = model.generate_cot(chunk, n_tokens=5, greedy=True).tolist()
        for ex, cot in zip(examples[start : start + batch], cots, strict=True):
            ok += int(score_solution(ex, cot)["answer_correct"])
    return {"answer_acc": ok / n}


def evaluate_pairs(model: TinyTransformer, pairs: list[PreferencePair],
                   *, batch: int = 512) -> dict:
    model.eval()
    prompts, chosen, rejected = _pair_tensors(pairs)
    n = prompts.shape[0]
    wins = 0
    margin_sum = 0.0
    with torch.no_grad():
        for start in range(0, n, batch):
            sl = slice(start, start + batch)
            lc = model.seq_logprob(prompts[sl], chosen[sl])
            lr = model.seq_logprob(prompts[sl], rejected[sl])
            wins += int((lc > lr).sum().item())
            margin_sum += float((lc - lr).sum().item())
    return {"win_rate": wins / n, "margin": margin_sum / n}


def run_method(name: str, *, base: TinyTransformer, train_pairs: list[PreferencePair],
               eval_examples: list[Example], eval_pairs: list[PreferencePair],
               steps: int, batch: int, lr: float, seed: int, eval_every: int,
               **hp) -> list[dict]:
    """Optimise one objective from ``base``; return the eval curve."""
    loss_fn = METHODS[name]
    torch.manual_seed(seed)
    model = copy.deepcopy(base)
    ref = copy.deepcopy(base).eval()
    for p in ref.parameters():
        p.requires_grad = False
    prompts, chosen, rejected = _pair_tensors(train_pairs)
    n = prompts.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    curve: list[dict] = []

    def snapshot(step: int) -> None:
        a = evaluate_answer(model, eval_examples)
        pr = evaluate_pairs(model, eval_pairs)
        curve.append({"step": step, **a, **pr})

    snapshot(0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(batch, n),))
        batch_tensors = (prompts[idx], chosen[idx], rejected[idx])
        loss = loss_fn(model, ref, batch_tensors, **hp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            snapshot(step)
    return curve


def make_dataset(n_pairs: int, n_eval: int, seed: int, lo: int, hi: int):
    from .data import make_pairs, sample_examples
    rng = random.Random(seed)
    train_examples = sample_examples(n_pairs, rng, lo, hi)
    train_pairs = make_pairs(train_examples, rng)
    eval_rng = random.Random(seed + 9991)
    eval_examples = sample_examples(n_eval, eval_rng, lo, hi)
    eval_pairs = make_pairs(eval_examples, eval_rng)
    return train_examples, train_pairs, eval_examples, eval_pairs
