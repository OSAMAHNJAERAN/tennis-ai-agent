import pytest

from scripts.evaluate.benchmark_wasb_head_external import advance_gate


def model(precision, recall, f1, clip_precision=.96, clip_recall=.96):
    return dict(pooled=dict(precision=precision, recall=recall, f1=f1),
                per_clip={'clip':dict(precision=clip_precision, recall=clip_recall)})


@pytest.mark.parametrize('candidate,expected', [
    (model(.97, .97, .97), True),
    (model(.95, .95, .95), False),
    (model(.94, .99, .964), False),
    (model(.99, .94, .964), False),
    (model(.97, .97, .97, .94, .96), False),
    (model(.97, .97, .97, .95, .95), True),
    (model(.97, .97, .97, None, None), False),
])
def test_f1_gain_cannot_hide_pooled_or_clip_regression(candidate, expected):
    advance, passing = advance_gate(dict(original=model(.95, .95, .95), adapted=candidate))
    assert advance is expected
    assert passing['original'] == 1
