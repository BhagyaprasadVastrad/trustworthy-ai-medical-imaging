import pandas as pd
import pytest

from src.alignment import align_by_study_id


def test_alignment_uses_identifier_not_row_order():
    predictions = pd.DataFrame(
        {"StudyInstanceUID": ["B", "A"], "probability": [0.8, 0.2]}
    )
    labels = pd.DataFrame({"StudyInstanceUID": ["A", "B"], "label": [0, 1]})

    out = align_by_study_id(predictions, labels)

    assert list(out["StudyInstanceUID"]) == ["B", "A"]
    assert list(out["label"]) == [1, 0]


def test_duplicate_ids_are_rejected():
    predictions = pd.DataFrame(
        {"StudyInstanceUID": ["A", "A"], "probability": [0.2, 0.3]}
    )
    labels = pd.DataFrame({"StudyInstanceUID": ["A"], "label": [0]})

    with pytest.raises(ValueError, match="duplicate"):
        align_by_study_id(predictions, labels)
