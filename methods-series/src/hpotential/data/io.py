"""Secure, auditable input/output for real cohort data."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import DatasetSchema, SchemaError, ValidationReport, validate_dataframe


@dataclass(frozen=True)
class DataBundle:
    """Validated data plus non-identifying provenance metadata."""

    frame: pd.DataFrame
    report: ValidationReport
    source_sha256: str
    source_format: str
    schema: DatasetSchema


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format {suffix}; use CSV, TSV, XLSX/XLS, or Parquet")


def load_schema(path: str | Path) -> DatasetSchema:
    """Load a version-controlled JSON schema."""

    with Path(path).open(encoding="utf-8") as stream:
        return DatasetSchema.from_dict(json.load(stream))


def load_dataset(
    data_path: str | Path,
    schema_path: str | Path,
    *,
    sheet_name: str | int | None = None,
    strict: bool = True,
) -> DataBundle:
    """Read, coerce, validate and fingerprint a real dataset.

    Raw values are never written by this function. In strict mode any validation error
    raises :class:`SchemaError`; otherwise the caller receives the report for remediation.
    """

    source = Path(data_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    schema = load_schema(schema_path)
    clean, report = validate_dataframe(_read(source, sheet_name), schema)
    if strict and not report.valid:
        raise SchemaError("Dataset validation failed:\n- " + "\n- ".join(report.errors))
    return DataBundle(clean, report, _sha256(source), source.suffix.lower().lstrip("."), schema)


def _manifest(bundle: DataBundle) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": {"name": bundle.schema.name, "version": bundle.schema.version},
        "source_sha256": bundle.source_sha256,
        "source_format": bundle.source_format,
        "rows": len(bundle.frame),
        "columns": len(bundle.frame.columns),
        "validation": bundle.report.to_dict(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }


def write_analysis_bundle(
    bundle: DataBundle,
    output_dir: str | Path,
    *,
    include_identifiers: bool = False,
    output_format: str = "csv",
) -> dict[str, Path]:
    """Write a validated analysis dataset, manifest and missingness table.

    Identifier-role columns are removed by default. The function refuses invalid data and
    never overwrites an existing output directory, protecting prior analysis snapshots.
    """

    if not bundle.report.valid:
        raise SchemaError("Cannot write an analysis bundle from invalid data")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.mkdir(parents=True)
    frame = bundle.frame.copy()
    identifier_columns = {c.name for c in bundle.schema.columns if c.role == "id"}
    if not include_identifiers:
        frame = frame.drop(columns=sorted(identifier_columns & set(frame.columns)))

    if output_format == "csv":
        data_file = destination / "analysis_data.csv"
        frame.to_csv(data_file, index=False)
    elif output_format == "parquet":
        data_file = destination / "analysis_data.parquet"
        frame.to_parquet(data_file, index=False)
    else:
        raise ValueError("output_format must be 'csv' or 'parquet'")

    manifest_file = destination / "manifest.json"
    manifest_file.write_text(json.dumps(_manifest(bundle), indent=2), encoding="utf-8")
    missing_file = destination / "missingness.csv"
    pd.DataFrame(
        {
            "column": list(bundle.report.missing_fraction),
            "missing_fraction": list(bundle.report.missing_fraction.values()),
        }
    ).to_csv(missing_file, index=False)
    return {"data": data_file, "manifest": manifest_file, "missingness": missing_file}
