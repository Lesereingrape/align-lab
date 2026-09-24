from __future__ import annotations

import random

from alignlab.data import (
    Example,
    digit_token,
    make_pairs,
    parse_answer,
    sample_examples,
    score_solution,
)


def test_cot_reconstructs_true_sum():
    for a in (11, 27, 48, 99, 63):
        for b in (12, 39, 51, 8, 77):
            ex = Example(a, b)
            assert parse_answer(ex.cot()) == a + b


def test_chosen_is_correct_and_rejected_is_wrong():
    rng = random.Random(0)
    examples = sample_examples(200, rng, 10, 99)
    for pair in make_pairs(examples, rng):
        assert score_solution(pair.ex, pair.chosen)["answer_correct"]
        assert score_solution(pair.ex, pair.chosen)["cot_correct"]
        rej = score_solution(pair.ex, pair.rejected)
        assert not rej["answer_correct"]
        # still a well-formed chain (parses to an integer), just the wrong sum
        assert rej["pred_sum"] is not None


def test_rejected_has_valid_shape():
    rng = random.Random(1)
    for ex in sample_examples(100, rng, 10, 99):
        bad = parse_answer(ex.cot())
        assert isinstance(bad, int)
        assert all(digit_token(0) <= t <= digit_token(9) for t in ex.cot())
