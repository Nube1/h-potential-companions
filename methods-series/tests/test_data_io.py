"""Tests for real-data validation and de-identified analysis snapshots."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hpotential.data import DatasetSchema, SchemaError, load_dataset, write_analysis_bundle
from hpotential.data.schema import validate_dataframe


def schema() -> DatasetSchema:
    return DatasetSchema.from_dict(
        {
            "name": "test",
            "version": "1",
            "unique_key": ["id", "visit"],
            "max_row_missing_fraction": 0.5,
            "columns": [
                {"name": "id", "kind": "string", "nullable": False, "role": "id"},
                {"name": "visit", "kind": "date", "nullable": False, "role": "time"},
                {"name": "access", "kind": "number", "minimum": 0, "maximum": 1},
                {
                    "name": "group",
                    "kind": "category",
                    "categories": ["a", "b"],
                    "role": "group",
                },
            ],
        }
    )


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["P1", "P2"],
            "visit": ["2026-01-01", "2026-01-02"],
            "access": [0.2, 0.8],
            "group": ["a", "b"],
        }
    )


def test_validation_coerces_and_accepts_valid_data() -> None:
    clean, report = validate_dataframe(valid_frame(), schema())
    assert report.valid
    assert str(clean["access"].dtype) == "float64"
    assert str(clean["visit"].dtype).startswith("datetime64")


def test_validation_detects_range_category_and_duplicate() -> None:
    bad = valid_frame()
    bad.loc[1, "id"] = "P1"
    bad.loc[1, "visit"] = "2026-01-01"
    bad.loc[0, "access"] = 1.2
    bad.loc[1, "group"] = "unknown"
    _, report = validate_dataframe(bad, schema())
    assert not report.valid
    joined = " ".join(report.errors)
    assert "above 1" in joined
    assert "failed category coercion" in joined
    assert "duplicated" in joined


def test_load_and_write_removes_identifiers_and_records_hash(tmp_path) -> None:
    data = tmp_path / "cohort.csv"
    schema_file = tmp_path / "schema.json"
    valid_frame().to_csv(data, index=False)
    schema_file.write_text(json.dumps(schema().to_dict()), encoding="utf-8")
    bundle = load_dataset(data, schema_file)
    files = write_analysis_bundle(bundle, tmp_path / "snapshot")
    output = pd.read_csv(files["data"])
    manifest = json.loads(files["manifest"].read_text())
    assert "id" not in output
    assert len(manifest["source_sha256"]) == 64
    assert manifest["validation"]["valid"] is True
    with pytest.raises(FileExistsError):
        write_analysis_bundle(bundle, tmp_path / "snapshot")


def test_strict_load_rejects_invalid_data(tmp_path) -> None:
    data = tmp_path / "cohort.csv"
    schema_file = tmp_path / "schema.json"
    frame = valid_frame()
    frame.loc[0, "access"] = -1
    frame.to_csv(data, index=False)
    schema_file.write_text(json.dumps(schema().to_dict()), encoding="utf-8")
    with pytest.raises(SchemaError):
        load_dataset(data, schema_file)
