"""Tests for the item bank: vocabulary, schema, duplicates, CSV round trip."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from hpotential.core.itembank import (
    DOMAINS,
    HAEMOPHILIA_SUBFACETS,
    LIFE_STAGES,
    SCHEMA,
    HPotentialItemBank,
    Item,
    ItemBankError,
    example_bank,
    write_data_dictionary,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _valid(**overrides) -> dict:
    base = {
        "item_id": "BP99",
        "domain": "Biomedical Potential",
        "life_stage": "adult",
        "stem": "Climbing one flight of stairs",
    }
    base.update(overrides)
    return base


def test_domains_match_the_manuscripts():
    assert DOMAINS == ("Biomedical Potential", "Social Modifiers", "Systemic Burdens")
    assert LIFE_STAGES == ("paediatric", "transition", "adult", "ageing")


def test_draft_names_survive_only_as_subfacets():
    # The draft's three labels are sub-facets, each under one domain.
    assert set(HAEMOPHILIA_SUBFACETS) == {
        "Functional Capacity",
        "Structural Support",
        "Autonomy and Agency",
    }
    assert set(HAEMOPHILIA_SUBFACETS.values()) == set(DOMAINS)
    # and none of them is a domain
    assert not set(HAEMOPHILIA_SUBFACETS) & set(DOMAINS)


def test_add_and_frame_shape():
    bank = HPotentialItemBank().add(**_valid())
    frame = bank.to_frame()
    assert frame.shape == (1, len(SCHEMA))
    assert list(frame.columns) == [name for name, *_ in SCHEMA]


def test_duplicate_id_rejected():
    bank = HPotentialItemBank().add(**_valid())
    with pytest.raises(ItemBankError, match="duplicate"):
        bank.add(**_valid())
    assert len(bank) == 1


def test_duplicate_inside_a_batch_leaves_the_bank_unchanged():
    bank = HPotentialItemBank()
    with pytest.raises(ItemBankError, match="duplicate"):
        bank.extend([_valid(item_id="A"), _valid(item_id="A")])
    assert len(bank) == 0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"domain": "Functional Capacity"}, "unknown domain"),
        ({"life_stage": "elderly"}, "unknown life_stage"),
        ({"stem": ""}, "empty stem"),
        ({"n_categories": 1}, "at least 2"),
        ({"direction": 0}, "direction"),
        ({"respondent": "nurse"}, "unknown respondent"),
        ({"subfacet": "Structural Support"}, "belongs to"),
        ({"subfacet": "Something Else"}, "unknown subfacet"),
        ({"item_id": ""}, "item_id"),
    ],
)
def test_schema_violations_raise(overrides, message):
    with pytest.raises(ItemBankError, match=message):
        HPotentialItemBank().add(**_valid(**overrides))


def test_unknown_field_rejected():
    with pytest.raises(ItemBankError, match="unknown item fields"):
        HPotentialItemBank().add(_valid(weight=1.0))


def test_csv_round_trip(tmp_path):
    bank = example_bank()
    path = bank.to_csv(tmp_path / "bank.csv")
    reloaded = HPotentialItemBank.from_csv(path)
    pd.testing.assert_frame_equal(bank.to_frame(), reloaded.to_frame())
    assert reloaded.item_ids == bank.item_ids


def test_from_csv_validates_rows(tmp_path):
    frame = example_bank().to_frame()
    frame.loc[0, "domain"] = "Functional Capacity"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ItemBankError, match="unknown domain"):
        HPotentialItemBank.from_csv(path)


def test_coverage_shows_empty_cells_as_zero():
    table = example_bank().coverage()
    assert list(table.index) == list(DOMAINS)
    assert list(table.columns) == list(LIFE_STAGES) + ["total"]
    assert int(table.loc["Systemic Burdens", "paediatric"]) == 0
    assert int(table["total"].sum()) == len(example_bank())


def test_filter_returns_an_independent_bank():
    bank = example_bank()
    adults = bank.filter(life_stage="adult")
    assert len(adults) == 6
    adults.add(**_valid())
    assert len(bank) == 12


def test_filter_rejects_unknown_values():
    with pytest.raises(ItemBankError, match="unknown domain"):
        example_bank().filter(domain="Autonomy and Agency")


def test_shipped_dictionary_matches_the_code(tmp_path):
    shipped = pd.read_csv(DATA_DIR / "dictionary.csv")
    regenerated = pd.read_csv(write_data_dictionary(tmp_path / "dictionary.csv"))
    pd.testing.assert_frame_equal(shipped, regenerated)


def test_building_a_large_bank_is_linear_enough():
    # The draft concatenated a frame per item, which is quadratic: 5000 items
    # took minutes. Accumulating records and building once takes well under a
    # second, so a generous bound still fails loudly if the loop regresses.
    start = time.perf_counter()
    bank = HPotentialItemBank()
    bank.extend(
        Item(
            item_id=f"X{i:05d}",
            domain=DOMAINS[i % 3],
            life_stage=LIFE_STAGES[i % 4],
            stem=f"Synthetic stem {i}",
        )
        for i in range(5000)
    )
    frame = bank.to_frame()
    elapsed = time.perf_counter() - start
    assert frame.shape[0] == 5000
    assert elapsed < 10.0, f"building 5000 items took {elapsed:.1f}s"
