# alignlab — SFT vs DPO vs ORPO vs SimPO on a verifiable task, on a CPU

**alignlab** is a ~575-line, dependency-light reproduction harness for the four
post-training objectives that dominate open-LLM alignment today, run honestly at
toy scale so that every curve is real, seeded, and reproducible in minutes rather
than borrowed from a paper. The same ~103k-parameter transformer, the same frozen
SFT base, and the same preference pairs are fed to each objective, and the result
is a genuine, non-obvious finding about **what win-rate metrics hide**.

![ci](https://github.com/Lesereingrape/align-lab/actions/workflows/ci.yml/badge.svg)

- **SFT** — cross-entropy on the chosen chain (reference-free; no preference term).
- **DPO** (Rafailov et al., 2305.18298) — reference-model margin on (chosen, rejected).
- **ORPO** (Hong et al., 2403.07691) — SFT loss **plus** an odds-ratio preference
  term, with **no reference model**.
- **SimPO** (Meng et al., 2405.14734) — **length-normalised** margin, **no reference**.

## The setup

- **Task:** a from-scratch decoder-only transformer (~103k params) emits the
  digit-by-digit column-addition carry chain for two 2-digit operands. The answer
  is reconstructed from the chain and checked against `a + b` by an **exact
  verifier**, so accuracy is measured on *real generated arithmetic*, never on
  teacher forcing.
- **Preference pairs:** chosen = the correct chain, rejected = a subtly wrong one
  (dropped carry, off-by-one digit) that is *guaranteed* to reconstruct to the
  wrong sum. So on this task the preferred response genuinely **is** the correct
  behaviour.
- **Two metrics, always reported together:** greedy **answer accuracy** (sample
  the model's own chain and grade it) and offline **pair win-rate** (does
  `logprob(chosen) > logprob(rejected)` on held-out pairs).

## The finding

Alignment papers often report preference win-rate as *the* success metric. Here
that is exactly the trap: **DPO and SimPO push win-rate to ~0.99 while their
greedy answer accuracy collapses below the untrained base** — the textbook
*likelihood-displacement* failure, made visible only because we also measure real
generations. ORPO's built-in SFT anchor sidesteps it. See the tables below.

## Quickstart

```bash
pip install -e .                 # torch is the only runtime dependency
python -m alignlab.cli demo      # seed 0 at the published 600-step budget
python experiments/run_study.py  # full 3-seed study -> results/alignment.json
python experiments/make_report.py --write  # splice the block into README.md
```

## Results

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and committed as [`results/alignment.json`](results/alignment.json); the tables are rendered by `experiments/make_report.py`. All four objectives start from the same 200-step SFT base and train 600 steps on 2500 preference pairs; mean over 3 seeds.*
*A rerun is expected to reproduce these numbers bit for bit only under the environment the artifact records — Python 3.13.7, torch 2.14.0+cpu, 8 CPU threads on Windows-11-10.0.26200-SP0 — because float reduction order over a batch follows the thread count and the torch build. Elsewhere expect the same shape, not the same digits.*

- tiny model: **103,055** parameters (decoder-only transformer, CPU-only)
- weak SFT base: greedy answer accuracy **0.699** (±0.168), offline pair win-rate 0.966

### Final held-out: generation accuracy vs offline preference fit

| objective | greedy answer acc | offline win-rate | mean margin |
|-----------|------------------:|-----------------:|------------:|
| SFT base (starting point) | 0.699 | 0.966 | 4.65 |
| **SFT** | **1.000** (±0.000) | 1.000 | 12.10 |
| **ORPO** | **1.000** (±0.000) | 1.000 | 14.02 |
| **DPO** | **0.097** (±0.065) | 0.990 | 61.83 |
| **SIMPO** | **0.096** (±0.032) | 0.994 | 26.85 |

Read the two accuracy/win-rate columns *together*. On a task where the preferred response is the fully correct chain, plain **SFT** and the SFT-coupled **ORPO** stay at ceiling (1.000). The pure preference objectives do the opposite: **DPO** drives the offline win-rate to 0.990 and the margin to 61.8 yet its *greedy* answer accuracy collapses to 0.097 - **below the 0.70 base** - and SimPO shows the same (0.096 acc at win-rate 0.994).

Per-seed final greedy accuracy — DPO 0.025, DPO 0.182, DPO 0.085 against ORPO 1.000, ORPO 1.000, ORPO 1.000.
Every one of the 3 DPO seeds finishes below the 0.699 base, so the collapse is a property of the objective on this task rather than one unlucky initialisation.

### Held-out greedy answer accuracy vs training step

| step | sft acc | orpo acc | dpo acc | simpo acc |
|-----:|-------:|-------:|-------:|-------:|
| 0 | 0.699 | 0.699 | 0.699 | 0.699 |
| 150 | 0.991 | 0.989 | 0.089 | 0.127 |
| 300 | 0.992 | 0.993 | 0.090 | 0.098 |
| 450 | 1.000 | 0.969 | 0.079 | 0.076 |
| 600 | 1.000 | 1.000 | 0.097 | 0.096 |

The collapse is immediate: DPO is already at 0.089 by the first eval. This is the documented *likelihood-displacement* pathology - a margin-only objective widens chosen-vs-rejected by pushing both absolute log-probs around, so it can raise the pair win-rate while the argmax chain drifts off the correct answer. Win-rate alone is a misleading alignment metric; measuring real generations is what exposes it. ORPO's built-in SFT term anchors the chosen likelihood and avoids the collapse.

### Ablation: DPO preference strength (beta) vs accuracy

| beta | greedy answer acc |
|-----:|------------------:|
| 0.05 | 0.111 |
| 0.1 | 0.097 |
| 0.5 | 0.077 |

Stronger preference pressure (larger beta) is not gentler here - it displaces the policy *more* (0.111 -> 0.077), the opposite of the usual alignment intuition, again because the base is already near-saturated on win-rate and the only room left is to inflate the margin.

### Honest limitations

- This is a deliberately toy, fully-verifiable task. Because the preferred response *is* the correct chain, SFT/ORPO have a structural advantage on raw accuracy that real human preference data (correct-but-verbose vs better) would not give them.
- The DPO/SimPO collapse is a real, reproducible failure mode, not a claim that these methods are generally bad: with a less-saturated base, shorter schedules, or an SFT anchor they behave normally.
- It is an elicitation/behavioural study of the objectives on a tiny model, not a claim about frontier-scale alignment.
<!-- RESULTS:END -->

## Repository layout

```
src/alignlab/
  data.py     digit-addition examples, gold CoT, guaranteed-wrong preference pairs
  model.py    TinyTransformer + seq_logprob / sft_loss / generate_cot
  align.py    the four objective losses + METHODS registry
  study.py    the published budget, shared by the study script and the demo
  train.py    base pre-training, per-method training loop, both evaluations
  cli.py      `alignlab demo`
experiments/
  run_study.py     multi-seed study + DPO-beta ablation -> results/alignment.json
  make_report.py   renders the exact Results block above from the JSON
tests/         data / model / objective + README-drift, demo-vs-study and artifact self-consistency guards
```

## Reproducing and honesty

The `Results` block is generated, not typed: a CI test
(`tests/test_readme_matches_results.py`) asserts the README equals
`make_report.build(results/alignment.json)`, so no number can drift from the
committed artifact — and the same test checks that the "~575-line" claim above still
matches the package. A second guard (`tests/test_demo_matches_study.py`) pins
`alignlab demo` to `alignlab.study`, so the live command cannot quietly become a
smaller, differently-tuned experiment than the table it is printed above.
The artifact also keeps the per-seed curves behind every mean — a third guard
(`tests/test_artifact_is_internally_consistent.py`) recomputes each published point
from them, and the Results block states per seed whether the DPO collapse is
universal or one unlucky initialisation. Std-devs are across the 3 seeds; where a gap
sits inside the noise it is called out rather than sold. A rerun is bit-exact only
under the Python/torch/thread environment the JSON records; elsewhere expect the same
shape, not the same digits. The limitations section names the one
structural bias in this setup (chosen == gold favours SFT/ORPO on raw accuracy)
instead of hiding it.

## References

- DPO: Rafailov et al., *Direct Preference Optimization*, 2305.18298
- ORPO: Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model*, 2403.07691
- SimPO: Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, 2405.14734
- Likelihood displacement: Razin et al., *Unintended Likelihood Alignment in DPO*, 2410.08847

## License

MIT
