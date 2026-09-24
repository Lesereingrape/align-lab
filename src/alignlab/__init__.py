"""alignlab: a CPU-only comparison of preference post-training objectives.

Four alignment methods (plain SFT, DPO, ORPO, SimPO) are trained from the same
weakly-supervised base model on the *same* chosen/rejected preference pairs and
scored on the *same* held-out set. The task has an exact verifier, so every
reported number is a real measurement, not a borrowed figure.
"""

from . import align, data, model, train

__all__ = ["align", "data", "model", "train"]
