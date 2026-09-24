"""Column addition as a verifiable CoT task, plus chosen/rejected preference pairs.

The model sees two 2-digit operands and must emit the digit-by-digit column
addition, right to left, interleaving each result digit with the carry it makes:

    prompt:  a1 a0 + b1 b0 =
    target:  o0 c1 o1 c2 o2

with a0+b0 = o0 + 10*c1, a1+b1+c1 = o1 + 10*c2, and o2 = c2. Reading o2 o1 o0
reconstructs the answer, which we can check exactly against a+b without trusting
the model. The whole alignment study is powered by that free, exact verifier.

A *preference pair* is (chosen, rejected): the chosen chain is the gold one, the
rejected chain is a plausible wrong derivation (a mistaken digit or carry) that
still has valid shape. This is the supervision DPO / ORPO / SimPO learn from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD, BOS, EOS, PLUS, EQ = 0, 1, 2, 3, 4
DIGIT0 = 5
VOCAB = ["<pad>", "<bos>", "<eos>", "+", "="] + [str(d) for d in range(10)]
NVOCAB = len(VOCAB)
COT_LEN = 5


def digit_token(d: int) -> int:
    return DIGIT0 + d


def token_to_digit(tok: int) -> int:
    return tok - DIGIT0


@dataclass(frozen=True)
class Example:
    a: int
    b: int

    @property
    def target(self) -> int:
        return self.a + self.b

    def prompt(self) -> list[int]:
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        return [digit_token(a1), digit_token(a0), PLUS,
                digit_token(b1), digit_token(b0), EQ]

    def cot(self) -> list[int]:
        """Ground-truth o0 c1 o1 c2 o2 (5 tokens)."""
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        s0 = a0 + b0
        o0, c1 = s0 % 10, s0 // 10
        s1 = a1 + b1 + c1
        o1, c2 = s1 % 10, s1 // 10
        return [digit_token(x) for x in (o0, c1, o1, c2, c2)]


def sample_examples(n: int, rng: random.Random, lo: int, hi: int) -> list[Example]:
    lo = max(lo, 10)
    hi = min(hi, 99)
    return [Example(a=rng.randint(lo, hi), b=rng.randint(lo, hi)) for _ in range(n)]


def parse_answer(cot_tokens: list[int]) -> int | None:
    if len(cot_tokens) < COT_LEN:
        return None
    if any(not (DIGIT0 <= t <= DIGIT0 + 9) for t in cot_tokens[:COT_LEN]):
        return None
    o0, _c1, o1, _c2, o2 = (token_to_digit(t) for t in cot_tokens[:COT_LEN])
    return 100 * o2 + 10 * o1 + o0


def score_solution(ex: Example, cot_tokens: list[int]) -> dict:
    ans = parse_answer(cot_tokens)
    gold = ex.cot()
    return {
        "answer_correct": ans is not None and ans == ex.target,
        "cot_correct": len(cot_tokens) >= COT_LEN and cot_tokens[:COT_LEN] == gold,
        "pred_sum": ans,
        "true_sum": ex.target,
    }


def _bump(tok: int, delta: int) -> int:
    """Move a digit token by ``delta`` staying in digit range (a plausible slip)."""
    d = token_to_digit(tok) + delta
    d %= 10
    return digit_token(d)


def make_rejected(ex: Example, rng: random.Random) -> list[int]:
    """A wrong chain of valid shape: flips one derivation step into an error.

    Three plausible failure modes are tried, chosen at random:
      * forget the carry into the tens column,
      * drop the final carry (answer wraps mod 100),
      * mis-add the units column by one.
    A candidate is only usable if it reconstructs a *different* sum than gold;
    when a zero carry makes a mode accidentally correct we fall back to shifting
    the units digit, which always changes the answer.
    """
    gold = ex.cot()
    a1, a0 = divmod(ex.a, 10)
    b1, b0 = divmod(ex.b, 10)

    def no_carry() -> list[int]:
        o0 = (a0 + b0) % 10
        c1 = (a0 + b0) // 10
        o1 = (a1 + b1) % 10  # ignores c1
        return [digit_token(x) for x in (o0, c1, o1, 0, 0)]

    def drop_final() -> list[int]:
        return [gold[0], gold[1], gold[2], gold[3], digit_token(0)]

    def units_off() -> list[int]:
        bad = list(gold)
        bad[0] = _bump(bad[0], rng.choice((-1, 1)))
        return bad

    candidates = [rng.choice([no_carry, drop_final, units_off])()]
    candidates.append(units_off())  # guaranteed-wrong fallback
    for cand in candidates:
        parsed = parse_answer(cand)
        if parsed is not None and parsed != ex.target:
            return cand
    return units_off()


@dataclass(frozen=True)
class PreferencePair:
    ex: Example
    chosen: list[int]
    rejected: list[int]

    def tensors(self) -> tuple[list[int], list[int], list[int]]:
        return self.ex.prompt(), self.chosen, self.rejected


def make_pairs(examples: list[Example], rng: random.Random) -> list[PreferencePair]:
    """One (gold, plausible-wrong) pair per example."""
    return [PreferencePair(ex=ex, chosen=ex.cot(), rejected=make_rejected(ex, rng))
            for ex in examples]


def render(ex: Example, cot_tokens: list[int]) -> str:
    return " ".join(VOCAB[t] for t in ex.prompt() + list(cot_tokens))
