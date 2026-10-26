"""Declarative schemas and validation for participant-level data.

The module deliberately avoids disease-specific assumptions. A study supplies a JSON
schema defining names, types, ranges, categories, identifiers and missingness limits.
Validation returns a report and raises only when a caller requests strict operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

Kind = Literal["string", "integer", "number", "boolean", "date", "category"]


class SchemaError(ValueError):
    """Raised when a dataset violates a schema in strict mode."""


@dataclass(frozen=True)
class ColumnSpec:
    """Validation contract for one input column."""

    name: str
    kind: Kind
    required: bool = True
    nullable: bool = True
    minimum: float | None = None
    maximum: float | None = None
    categories: tuple[str, ...] = ()
    role: Literal["id", "time", "predictor", "outcome", "group", "weight", "other"] = "other"
    description: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ColumnSpec:
        data = dict(value)
        data["categories"] = tuple(data.get("categories", ()))
        return cls(**data)


@dataclass(frozen=True)
class DatasetSchema:
    """Study-level schema loaded from a version-controlled JSON document."""

    name: str
    version: str
    columns: tuple[ColumnSpec, ...]
    unique_key: tuple[str, ...] = ()
    max_row_missing_fraction: float = 1.0
    allow_extra_columns: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetSchema:
        return cls(
            name=value["name"],
            version=str(value["version"]),
            columns=tuple(ColumnSpec.from_dict(v) for v in value["columns"]),
            unique_key=tuple(value.get("unique_key", ())),
            max_row_missing_fraction=float(value.get("max_row_missing_fraction", 1.0)),
            allow_extra_columns=bool(value.get("allow_extra_columns", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Machine-readable validation result without participant values."""

    schema_name: str
    schema_version: str
    rows: int
    columns: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_fraction: dict[str, float] = field(default_factory=dict)
    coercion_failures: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "rows": self.rows,
            "columns": self.columns,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_fraction": self.missing_fraction,
            "coercion_failures": self.coercion_failures,
        }


def _coerce(series: pd.Series, spec: ColumnSpec) -> tuple[pd.Series, int]:
    original_nonmissing = series.notna()
    if spec.kind == "string":
        result = series.astype("string")
    elif spec.kind == "integer":
        result = pd.to_numeric(series, errors="coerce").astype("Int64")
    elif spec.kind == "number":
        result = pd.to_numeric(series, errors="coerce").astype(float)
    elif spec.kind == "boolean":
        mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
        result = series.map(lambda x: mapping.get(str(x).strip().lower()) if pd.notna(x) else pd.NA)
        result = result.astype("boolean")
    elif spec.kind == "date":
        result = pd.to_datetime(series, errors="coerce", utc=True)
    elif spec.kind == "category":
        normalized = series.astype("string")
        allowed = normalized.isin(spec.categories) | normalized.isna()
        result = normalized.where(allowed).astype(
            pd.CategoricalDtype(categories=list(spec.categories))
        )
    else:  # pragma: no cover - protected by the Literal and schema construction
        raise ValueError(f"Unsupported kind: {spec.kind}")
    failures = int((original_nonmissing & result.isna()).sum())
    return result, failures


def validate_dataframe(
    frame: pd.DataFrame, schema: DatasetSchema
) -> tuple[pd.DataFrame, ValidationReport]:
    """Coerce and validate a data frame, returning no participant values in the report."""

    report = ValidationReport(schema.name, schema.version, len(frame), len(frame.columns))
    expected = {c.name for c in schema.columns}
    required = {c.name for c in schema.columns if c.required}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        report.errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    extra = sorted(set(frame.columns) - expected)
    if extra and not schema.allow_extra_columns:
        report.errors.append(f"Unexpected columns: {', '.join(extra)}")
    elif extra:
        report.warnings.append(f"Unmodelled columns retained: {', '.join(extra)}")

    clean = frame.copy()
    for spec in schema.columns:
        if spec.name not in clean:
            continue
        converted, failures = _coerce(clean[spec.name], spec)
        clean[spec.name] = converted
        if failures:
            report.coercion_failures[spec.name] = failures
            report.errors.append(f"{spec.name}: {failures} value(s) failed {spec.kind} coercion")
        missing = float(converted.isna().mean()) if len(converted) else 0.0
        report.missing_fraction[spec.name] = missing
        if not spec.nullable and converted.isna().any():
            report.errors.append(f"{spec.name}: missing values are not permitted")
        if spec.kind in {"integer", "number"}:
            numeric = pd.to_numeric(converted, errors="coerce")
            if spec.minimum is not None:
                n = int((numeric < spec.minimum).sum())
                if n:
                    report.errors.append(f"{spec.name}: {n} value(s) below {spec.minimum}")
            if spec.maximum is not None:
                n = int((numeric > spec.maximum).sum())
                if n:
                    report.errors.append(f"{spec.name}: {n} value(s) above {spec.maximum}")
        if spec.kind == "category" and not spec.categories:
            report.errors.append(f"{spec.name}: category columns require allowed categories")

    if schema.unique_key and set(schema.unique_key).issubset(clean.columns):
        duplicates = int(clean.duplicated(list(schema.unique_key), keep=False).sum())
        if duplicates:
            report.errors.append(
                f"Unique key {list(schema.unique_key)} has {duplicates} duplicated row(s)"
            )

    model_columns = [c.name for c in schema.columns if c.name in clean]
    if model_columns and len(clean):
        row_missing = clean[model_columns].isna().mean(axis=1)
        excessive = int((row_missing > schema.max_row_missing_fraction).sum())
        if excessive:
            report.errors.append(
                f"{excessive} row(s) exceed missing fraction "
                f"{schema.max_row_missing_fraction:.3f}"
            )

    for name, fraction in report.missing_fraction.items():
        if fraction > 0.30:
            report.warnings.append(f"{name}: high missingness ({fraction:.1%})")
    return clean, report
