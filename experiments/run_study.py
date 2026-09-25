"""Run the SFT-vs-DPO-vs-ORPO-vs-SimPO comparison and write results JSON.

Usage:  PYTHONPATH=src python experiments/run_study.py [--out PATH]
Writes results/alignment.json; the README tables are rendered from that file by
make_report.py, so the numbers shipped in the README are exactly these.

The artifact records the environment it came off, and the final metric of every
individual seed. Float reduction order over a batch depends on the thread count and
the torch build, so a rerun is bit-exact *in that environment* and merely close in
another — and the per-seed rows are what let a reader see how much of each gap is
one lucky initialisation.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from alignlab.model import count_parameters
from alignlab.study import (
    BASE_STEPS,
    BATCH,
    DPO_BETAS,
    EVAL_EVERY,
    HI,
    LO,
    LR_ALIGN,
    LR_BASE,
    METHODS,
    N_EVAL,
    N_PAIRS,
    SEEDS,
    STEPS,
)
from alignlab.train import build_base, evaluate_answer, evaluate_pairs, make_dataset, run_method


def environment() -> dict:
    """The machine these numbers came off, recorded next to them.

    Within this environment a rerun is bit-exact; across environments the float
    reduction order over a batch changes with the thread count and the torch build,
    so the artifact names its condition instead of promising a reproducibility it
    cannot deliver.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def _agg(curves: list[list[dict]]) -> list[dict]:
    """Mean/std across seeds at each recorded step index."""
    out = []
    for i in range(len(curves[0])):
        step = curves[0][i]["step"]
        aa = [c[i]["answer_acc"] for c in curves]
        wr = [c[i]["win_rate"] for c in curves]
        out.append({
            "step": step,
            "answer_acc": round(statistics.fmean(aa), 4),
            "answer_acc_std": round(statistics.pstdev(aa), 4),
            "win_rate": round(statistics.fmean(wr), 4),
        })
    return out


def run_seed(seed: int) -> dict:
    train_examples, train_pairs, eval_examples, eval_pairs = make_dataset(
        N_PAIRS, N_EVAL, seed, LO, HI)
    base = build_base(steps=BASE_STEPS, batch=BATCH, lr=LR_BASE,
                      seed=seed, examples=train_examples)
    base_m = {**evaluate_answer(base, eval_examples),
              **evaluate_pairs(base, eval_pairs)}
    base_m = {k: round(v, 4) for k, v in base_m.items()}

    curves: dict[str, list[dict]] = {}
    for name in METHODS:
        curves[name] = run_method(
            name, base=base, train_pairs=train_pairs,
            eval_examples=eval_examples, eval_pairs=eval_pairs,
            steps=STEPS, batch=BATCH, lr=LR_ALIGN, seed=seed + 1,
            eval_every=EVAL_EVERY)

    dpo_beta = {}
    for beta in DPO_BETAS:
        c = run_method("dpo", base=base, train_pairs=train_pairs,
                       eval_examples=eval_examples, eval_pairs=eval_pairs,
                       steps=STEPS, batch=BATCH, lr=LR_ALIGN, seed=seed + 1,
                       eval_every=EVAL_EVERY, beta=beta)
        dpo_beta[str(beta)] = round(c[-1]["answer_acc"], 4)

    return {"seed": seed, "params": count_parameters(base),
            "base": base_m, "curves": curves, "dpo_beta": dpo_beta}


def main(out: str | None = None) -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]

    methods = {}
    for name in METHODS:
        seed_curves = [ps["curves"][name] for ps in per_seed]
        curve = _agg(seed_curves)
        finals = [ps["curves"][name][-1] for ps in per_seed]
        methods[name] = {
            "curve": curve,
            "final_answer_acc": round(statistics.fmean(f["answer_acc"] for f in finals), 4),
            "final_answer_acc_std": round(statistics.pstdev(
                [f["answer_acc"] for f in finals]), 4),
            "final_win_rate": round(statistics.fmean(f["win_rate"] for f in finals), 4),
            "final_margin": round(statistics.fmean(f["margin"] for f in finals), 4),
            # Per-seed answer accuracy at every checkpoint: the mean curve above is
            # what the tables print, and a test checks it is really the mean of this.
            "answer_acc_per_seed": {
                str(ps["seed"]): [round(p["answer_acc"], 4) for p in ps["curves"][name]]
                for ps in per_seed},
        }

    dpo_beta = {b: round(statistics.fmean(ps["dpo_beta"][b] for ps in per_seed), 4)
                for b in map(str, DPO_BETAS)}

    artifact = {
        "config": {
            "methods": list(METHODS), "seeds": list(SEEDS), "n_pairs": N_PAIRS,
            "n_eval": N_EVAL, "operand_lo": LO, "operand_hi": HI,
            "base_steps": BASE_STEPS, "align_steps": STEPS, "eval_every": EVAL_EVERY,
            "lr_base": LR_BASE, "lr_align": LR_ALIGN,
            "params": per_seed[0]["params"],
        },
        "base": {
            "answer_acc": round(statistics.fmean(ps["base"]["answer_acc"] for ps in per_seed), 4),
            "answer_acc_std": round(statistics.pstdev(
                [ps["base"]["answer_acc"] for ps in per_seed]), 4),
            "answer_acc_per_seed": {str(ps["seed"]): ps["base"]["answer_acc"]
                                    for ps in per_seed},
            "win_rate": round(statistics.fmean(ps["base"]["win_rate"] for ps in per_seed), 4),
            "margin": round(statistics.fmean(ps["base"]["margin"] for ps in per_seed), 4),
        },
        "methods": methods,
        "dpo_beta_ablation": dpo_beta,
        "environment": environment(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    dest = Path(out or "results/alignment.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {dest} in {artifact['runtime_sec']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_study")
    parser.add_argument("--out", default=None,
                        help="where to write the artifact (default: results/alignment.json)")
    main(parser.parse_args().out)
