import pytest
from altwer import wer


def test_altwer_wer():
    references = [
        '["jenta","jenten"] ["jogga","jogget"] på ["broa","broen","brua","bruen"]',
        '["katten","katta"] ligger på ["matta","matten"]',
        "Det var en fin dag."
    ]
    hypotheses = [
        "jenta jogga på broa",
        "katten ligger på matta",
        "Det var en fin dag."
    ]

    # Default behavior
    score = wer(references, hypotheses, verbose=False)
    assert pytest.approx(score, 0.001) == 0.0

    # Test with lowercase enabled
    score_lower = wer(references, [h.upper() for h in hypotheses], lowercase=True)
    assert pytest.approx(score_lower, 0.001) == 0.0

    # Test with punctuation removal
    references_with_punct = [
        '["jenta","jenten"] jogga, på ["broa","broen","brua","bruen"]',
        '["katten","katta"], ligger på ["matta","matten"]',
        "Det var en fin dag!"
    ]
    hypotheses_with_punct = [
        "jenta jogga på broa",
        "katten ligger på matta",
        "Det var en fin dag"
    ]
    score_punct = wer(references_with_punct, hypotheses_with_punct, remove_punctuation=True)
    assert pytest.approx(score_punct, 0.001) == 0.0
