import pandas as pd
import pytest

from src.time_splits import chronological_match_splits


def test_keeps_all_states_for_a_match_in_one_chronological_partition():
    dataset = pd.DataFrame(
        {
            "match_id": ["early", "early", "validation", "test"],
            "date": pd.to_datetime(
                ["2019-05-01", "2019-05-01", "2022-05-01", "2024-05-01"]
            ),
            "state": [1, 2, 1, 1],
        }
    )

    splits = chronological_match_splits(dataset)

    assert len(splits["train"]) == 2
    assert set(splits["validation"]["match_id"]) == {"validation"}
    assert set(splits["test"]["match_id"]) == {"test"}
    assert set(splits["train"]["match_id"]).isdisjoint(splits["validation"]["match_id"])
    assert set(splits["validation"]["match_id"]).isdisjoint(splits["test"]["match_id"])


def test_rejects_one_match_with_multiple_dates():
    dataset = pd.DataFrame(
        {
            "match_id": ["match-a", "match-a"],
            "date": pd.to_datetime(["2023-01-01", "2024-01-01"]),
        }
    )

    with pytest.raises(ValueError, match="exactly one date"):
        chronological_match_splits(dataset)


def test_rejects_overlapping_or_reversed_cutoffs():
    dataset = pd.DataFrame(
        {"match_id": ["match-a"], "date": pd.to_datetime(["2022-01-01"])}
    )

    with pytest.raises(ValueError, match="earlier"):
        chronological_match_splits(
            dataset,
            validation_start="2024-01-01",
            test_start="2020-01-01",
        )