"""The four post-training objectives, implemented from their published forms.

All operate on batches of (prompt, chosen CoT, rejected CoT). ``ref`` is a frozen
copy of the base policy (only DPO uses it; ORPO and SimPO are reference-free).

  * SFT   - cross-entropy on the chosen chain (no preference signal).
  * DPO   - -log sigmoid(beta * [(pol-ch) - (pol-r) - (ref-ch) + (ref-r)])
  * ORPO  - SFT(chosen) - beta * log sigmoid(logodds(chosen) - logodds(rejected))
  * SimPO - -log sigmoid(beta/|y| * (sum logp chosen - sum logp rejected) - gamma)

length-normalised log-odds are used where the paper calls for them, so the
comparison holds the same code path for every method.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sum_logp(model, prompt, cot):
    return model.seq_logprob(prompt, cot)


def _avg_logp(model, prompt, cot):
    return model.seq_logprob(prompt, cot) / cot.shape[1]


def _unpack(batch):
    prompt, chosen, rejected = batch
    return prompt, chosen, rejected


def loss_sft(model, ref, batch, **_hp):
    prompt, chosen, _ = _unpack(batch)
    return model.sft_loss(prompt, chosen)


def loss_dpo(model, ref, batch, beta: float = 0.1, **_hp):
    prompt, chosen, rejected = _unpack(batch)
    with torch.no_grad():
        ref_c = _sum_logp(ref, prompt, chosen)
        ref_r = _sum_logp(ref, prompt, rejected)
    pol_c = _sum_logp(model, prompt, chosen)
    pol_r = _sum_logp(model, prompt, rejected)
    logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    return -F.logsigmoid(logits).mean()


def _logodds(model, prompt, cot):
    # average per-token logprob is in (-inf, 0); log-odds = x - log(1 - exp(x))
    avg = _avg_logp(model, prompt, cot)
    return avg - torch.log1p(-torch.exp(avg).clamp(max=1 - 1e-6))


def loss_orpo(model, ref, batch, beta: float = 0.5, **_hp):
    prompt, chosen, rejected = _unpack(batch)
    sft = model.sft_loss(prompt, chosen)
    odds_c = _logodds(model, prompt, chosen)
    odds_r = _logodds(model, prompt, rejected)
    return sft - beta * F.logsigmoid(odds_c - odds_r).mean()


def loss_simpo(model, ref, batch, beta: float = 2.0, gamma: float = 0.4, **_hp):
    prompt, chosen, rejected = _unpack(batch)
    avg_c = _avg_logp(model, prompt, chosen)
    avg_r = _avg_logp(model, prompt, rejected)
    logits = beta * (avg_c - avg_r) - gamma
    return -F.logsigmoid(logits).mean()


METHODS = {"sft": loss_sft, "dpo": loss_dpo, "orpo": loss_orpo, "simpo": loss_simpo}
