"""
M3 — Text metrics for radiology report generation.

compute_text_metrics(predictions, references) returns a dict with:
  BLEU-1..4, ROUGE-L (F), METEOR.

These are the standard captioning metrics. They measure word/phrase overlap
with the doctor's report. (Clinical metrics like CheXbert-F1 are a later add-on.)

Reusable for BOTH the Phase-1 baseline and the Phase-2 QLoRA model.
"""

from __future__ import annotations

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def _ensure_nltk():
    for pkg, path in [("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4"),
                      ("punkt", "tokenizers/punkt")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def compute_text_metrics(predictions: list[str], references: list[str]) -> dict:
    assert len(predictions) == len(references)
    _ensure_nltk()

    pred_tok = [p.lower().split() for p in predictions]
    ref_tok = [[r.lower().split()] for r in references]   # corpus_bleu wants list-of-refs

    smooth = SmoothingFunction().method1
    bleu = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        bleu[f"BLEU-{n}"] = corpus_bleu(ref_tok, pred_tok, weights=weights,
                                        smoothing_function=smooth)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure
                  for p, r in zip(predictions, references)) / len(predictions)

    meteor = sum(meteor_score([r.lower().split()], p.lower().split())
                 for p, r in zip(predictions, references)) / len(predictions)

    return {**bleu, "ROUGE-L": rouge_l, "METEOR": meteor}


def format_metrics(metrics: dict) -> str:
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
