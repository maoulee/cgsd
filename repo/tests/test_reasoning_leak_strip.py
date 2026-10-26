"""Thinking boundary must survive history sanitization (2026-10-22 ruling).

`strip_reasoning_leak` removes the vLLM-injected transition phrase. The default
also drops ``</think>`` (legacy action-only history), but the rollout transcript
must replay the sampled token stream, so it calls with ``keep_think_tags=True``.
"""

from kgqa.core.utils import strip_reasoning_leak

_LEAK = "I will now emit the tool call based on the reasoning above."


def test_default_strips_think_tag():
    assert strip_reasoning_leak("a</think>b") == "ab"


def test_keep_think_tags_preserves_boundary():
    text = "<think>reasoning</think>\nACTION"
    assert strip_reasoning_leak(text, keep_think_tags=True) == text


def test_leak_phrase_removed_in_both_modes():
    text = f"plan {_LEAK} action"
    assert strip_reasoning_leak(text) == "plan  action"
    assert strip_reasoning_leak(text, keep_think_tags=True) == "plan  action"


def test_turn_terminator_stripped_from_text_and_ids():
    from kgqa.rl.verl_plugins.opsd_pair import strip_turn_terminator
    text, ids = strip_turn_terminator("answer<|im_end|>", [11, 22, 248046])
    assert text == "answer"
    assert ids == [11, 22]
    # no terminator → untouched
    text2, ids2 = strip_turn_terminator("a</think>\nanswer", [11, 22])
    assert text2 == "a</think>\nanswer"
    assert ids2 == [11, 22]
