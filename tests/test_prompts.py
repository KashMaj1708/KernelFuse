"""Tests for token-accurate prompts (mock path; no HF download required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.prompts import make_prompt, prompt_token_len


def test_mock_word_prompt_length():
    p = make_prompt(32, seed=0, idx=1, model_id="mock-toy")
    assert len(p.split()) == 32


def test_max_prompt_cap():
    p = make_prompt(100, seed=0, idx=0, model_id="mock-toy", max_prompt_tokens=10)
    assert len(p.split()) == 10


def test_prompt_token_len_mock():
    p = make_prompt(16, seed=1, idx=2, model_id="mock")
    assert prompt_token_len(p, "mock") == 16


if __name__ == "__main__":
    test_mock_word_prompt_length()
    test_max_prompt_cap()
    test_prompt_token_len_mock()
    print("ok")
