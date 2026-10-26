"""Item bank for the H-potential construct.

The bank is a typed table of candidate indicators. It carries the vocabulary the
manuscripts use, checks each row against a schema, refuses duplicate item
identifiers, and round trips through CSV without losing types.

Domain vocabulary
-----------------
The three top-level domains are those of the manuscripts:

    Biomedical Potential, Social Modifiers, Systemic Burdens

An earlier draft of this package used a different set of names (Functional
Capacity, Structural Support, Autonomy and Agency). Those names are kept here as
haemophilia-specific sub-facets under :data:`HAEMOPHILIA_SUBFACETS`, so older
item spreadsheets can still be read, but they are not domains and must not be
reported as such. One mapping needs care: ``Structural Support`` sits under
``Systemic Burdens`` and is scored in the opposite direction, because support
present is burden absent. The ``direction`` column records that for each item.

Life stages
-----------
paediatric, transition, adult, ageing. A life stage marks the population an item
was written for. Items shared across stages are listed once per stage they apply
to, because wording and recall period usually differ.

Limitations
-----------
The bank stores metadata about items. It does not store responses, does not
score anything, and cannot establish content validity. Whether an item belongs
in a domain is a judgement made by people, recorded here for audit; the Delphi
module records what a panel decided about it. The example bank shipped with the
package is synthetic and written only to exercise the code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "DOMAINS",
    "LIFE_STAGES",
    "RESPONDENTS",
    "HAEMOPHILIA_SUBFACETS",
    "SCHEMA",
    "Item",
    "HPotentialItemBank",
    "example_bank",
]

DOMAINS: tuple[str, ...] = (
    "Biomedical Potential",
    "Social Modifiers",
    "Systemic Burdens",
)

LIFE_STAGES: tuple[str, ...] = ("paediatric", "transition", "adult", "ageing")

RESPONDENTS: tuple[str, ...] = ("self", "proxy", "clinician", "record")

#: Draft sub-facet names, kept for backward compatibility with early item
#: spreadsheets. Keys are sub-facets, values are the domain each one sits under.
#: ``Structural Support`` runs opposite in direction to its domain: an item that
#: measures support present measures burden absent, so it is entered with
#: ``direction = -1`` when the domain is scored as burden.
HAEMOPHILIA_SUBFACETS: dict[str, str] = {
    "Functional Capacity": "Biomedical Potential",
    "Autonomy and Agency": "Social Modifiers",
    "Structural Support": "Systemic Burdens",
}


@dataclass(frozen=True)
class Item:
    """One candidate indicator.

    Parameters
    ----------
    item_id : str
        Stable identifier, unique within a bank. Used as the column name when
        responses are collected.
    domain : str
        One of :data:`DOMAINS`.
    life_stage : str
        One of :data:`LIFE_STAGES`.
    stem : str
        The question as it is put to the respondent.
    n_categories : int
        Number of ordered response options, at least 2.
    direction : int
        ``+1`` when a higher raw response means greater capability, ``-1`` when
        it means less. Scoring code uses this; it is not cosmetic.
    respondent : str
        One of :data:`RESPONDENTS`.
    recall_period : str
        Free text, for example ``"past 4 weeks"``.
    subfacet : str
        Optional sub-facet. When given it must be a key of
        :data:`HAEMOPHILIA_SUBFACETS` whose value is ``domain``.
    source : str
        Where the item came from: an existing instrument, an interview, or new
        drafting.
    """

    item_id: str
    domain: str
    life_stage: str
    stem: str
    n_categories: int = 5
    direction: int = 1
    respondent: str = "self"
    recall_period: str = "past 4 weeks"
    subfacet: str = ""
    source: str = "new"


#: Column schema of the item data dictionary. Each entry is
#: ``(field, dtype, required, allowed values or "", description)``.
SCHEMA: tuple[tuple[str, str, bool, str, str], ...] = (
    ("item_id", "string", True, "", "Stable unique identifier for the item"),
    ("domain", "string", True, "|".join(DOMAINS), "Top-level H-potential domain"),
    ("life_stage", "string", True, "|".join(LIFE_STAGES), "Population the wording targets"),
    ("stem", "string", True, "", "Question as put to the respondent"),
    ("n_categories", "integer", True, ">=2", "Number of ordered response options"),
    ("direction", "integer", True, "-1|1", "+1 if higher response means greater capability"),
    ("respondent", "string", True, "|".join(RESPONDENTS), "Who answers the item"),
    ("recall_period", "string", True, "", "Recall window stated in the item"),
    (
        "subfacet",
        "string",
        False,
        "|".join(HAEMOPHILIA_SUBFACETS),
        "Haemophilia sub-facet, may be blank",
    ),
    ("source", "string", False, "", "Origin: instrument name, interview, or new"),
)

_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Item))
_STRING_COLUMNS: tuple[str, ...] = tuple(name for name, dtype, *_ in SCHEMA if dtype == "string")


class ItemBankError(ValueError):
    """Raised when an item or a frame fails the schema."""


class HPotentialItemBank:
    """A validated collection of candidate indicators.

    The bank keeps items in a list and builds a :class:`pandas.DataFrame` only
    when one is asked for. An earlier draft appended each row with
    ``pd.concat`` inside the loop, which copies the whole frame on every
    addition and costs O(n^2) time and memory; adding a few thousand items took
    minutes. Building once from a list of records is linear.

    Parameters
    ----------
    items : iterable of Item or mapping, optional
        Initial contents. Each element is validated on entry.

    Examples
    --------
    >>> bank = HPotentialItemBank()
    >>> _ = bank.add(item_id="BP01", domain="Biomedical Potential",
    ...              life_stage="adult", stem="Walking a short distance")
    >>> bank.to_frame().shape
    (1, 10)

    Limitations
    -----------
    Validation is structural. It checks vocabulary, types and uniqueness. It
    cannot check that an item measures what its domain label claims.
    """

    def __init__(self, items: Iterable[Item | Mapping[str, Any]] | None = None) -> None:
        self._items: list[Item] = []
        self._ids: set[str] = set()
        if items is not None:
            self.extend(items)

    # ------------------------------------------------------------------ size
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"HPotentialItemBank(n_items={len(self._items)})"

    @property
    def item_ids(self) -> list[str]:
        """Identifiers in insertion order."""
        return [it.item_id for it in self._items]

    # ------------------------------------------------------------- building
    def add(
        self, item: Item | Mapping[str, Any] | None = None, **kwargs: Any
    ) -> HPotentialItemBank:
        """Add one validated item and return the bank, so calls can chain.

        Parameters
        ----------
        item : Item or mapping, optional
            The item. When omitted, the item is built from keyword arguments.
        **kwargs
            Field values, used when ``item`` is omitted.

        Returns
        -------
        HPotentialItemBank
            The same bank, mutated.

        Raises
        ------
        ItemBankError
            If the item breaks the schema or repeats an existing ``item_id``.
        """
        if item is None:
            item = kwargs
        elif kwargs:
            raise ItemBankError("pass either an item or keyword fields, not both")
        obj = _coerce_item(item)
        _validate_item(obj)
        if obj.item_id in self._ids:
            raise ItemBankError(f"duplicate item_id: {obj.item_id!r}")
        self._items.append(obj)
        self._ids.add(obj.item_id)
        return self

    def extend(self, items: Iterable[Item | Mapping[str, Any]]) -> HPotentialItemBank:
        """Add many items, validating each. All or nothing.

        Parameters
        ----------
        items : iterable of Item or mapping
            Items to add.

        Returns
        -------
        HPotentialItemBank
            The same bank, mutated.

        Raises
        ------
        ItemBankError
            If any item fails. The bank is left unchanged in that case.
        """
        staged: list[Item] = []
        seen: set[str] = set()
        for raw in items:
            obj = _coerce_item(raw)
            _validate_item(obj)
            if obj.item_id in self._ids or obj.item_id in seen:
                raise ItemBankError(f"duplicate item_id: {obj.item_id!r}")
            seen.add(obj.item_id)
            staged.append(obj)
        self._items.extend(staged)
        self._ids.update(seen)
        return self

    # -------------------------------------------------------------- reading
    def to_frame(self) -> pd.DataFrame:
        """Return the bank as a frame with one row per item.

        Returns
        -------
        pandas.DataFrame
            Columns in schema order. String columns use pandas ``string``
            dtype, counts and direction use ``int64``.
        """
        frame = pd.DataFrame([asdict(it) for it in self._items], columns=list(_COLUMNS))
        return _apply_dtypes(frame)

    def filter(
        self,
        domain: str | Sequence[str] | None = None,
        life_stage: str | Sequence[str] | None = None,
    ) -> HPotentialItemBank:
        """Return a new bank holding the items that match.

        Parameters
        ----------
        domain : str or sequence of str, optional
            Domains to keep. ``None`` keeps all.
        life_stage : str or sequence of str, optional
            Life stages to keep. ``None`` keeps all.

        Returns
        -------
        HPotentialItemBank
            A new bank. The original is untouched.
        """
        doms = _as_set(domain, DOMAINS, "domain")
        stages = _as_set(life_stage, LIFE_STAGES, "life_stage")
        keep = [
            it
            for it in self._items
            if (doms is None or it.domain in doms) and (stages is None or it.life_stage in stages)
        ]
        return HPotentialItemBank(keep)

    def coverage(self) -> pd.DataFrame:
        """Count items by domain and life stage.

        Returns
        -------
        pandas.DataFrame
            Rows are the three domains in fixed order, columns the four life
            stages in fixed order, plus a ``total`` column. Empty cells are
            zeros, so gaps in the blueprint are visible rather than missing.
        """
        frame = self.to_frame()
        table = pd.crosstab(
            pd.Categorical(frame["domain"], categories=list(DOMAINS)),
            pd.Categorical(frame["life_stage"], categories=list(LIFE_STAGES)),
            dropna=False,
        )
        table.index.name = "domain"
        table.columns.name = "life_stage"
        table["total"] = table.sum(axis=1)
        return table

    # -------------------------------------------------------------- storage
    def to_csv(self, path: str | Path) -> Path:
        """Write the bank to CSV.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file. Parent directories must exist.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        out = Path(path)
        self.to_frame().to_csv(out, index=False)
        return out

    @classmethod
    def from_csv(cls, path: str | Path) -> HPotentialItemBank:
        """Read a bank from CSV, validating every row.

        Parameters
        ----------
        path : str or pathlib.Path
            Source file, with the columns of :data:`SCHEMA`.

        Returns
        -------
        HPotentialItemBank
            The bank.

        Raises
        ------
        ItemBankError
            If columns are missing or any row fails validation.
        """
        frame = pd.read_csv(Path(path), dtype=str, keep_default_na=False)
        return cls.from_frame(frame)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> HPotentialItemBank:
        """Build a bank from a frame, validating every row.

        Parameters
        ----------
        frame : pandas.DataFrame
            One row per item, columns as in :data:`SCHEMA`.

        Returns
        -------
        HPotentialItemBank
            The bank.

        Raises
        ------
        ItemBankError
            If required columns are missing or a row fails validation.
        """
        missing = [
            c for c in _COLUMNS if c not in frame.columns and c not in ("subfacet", "source")
        ]
        if missing:
            raise ItemBankError(f"missing columns: {missing}")
        records = frame.to_dict(orient="records")
        return cls(records)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _coerce_item(raw: Item | Mapping[str, Any]) -> Item:
    """Turn a mapping into an :class:`Item`, dropping nothing silently."""
    if isinstance(raw, Item):
        return raw
    if not isinstance(raw, Mapping):
        raise ItemBankError(f"cannot read an item from {type(raw).__name__}")
    unknown = set(raw) - set(_COLUMNS)
    if unknown:
        raise ItemBankError(f"unknown item fields: {sorted(unknown)}")
    data = dict(raw)
    for key in ("n_categories", "direction"):
        if key in data and data[key] is not None and not isinstance(data[key], bool):
            try:
                data[key] = int(data[key])
            except (TypeError, ValueError) as exc:
                raise ItemBankError(f"{key} must be an integer, got {data[key]!r}") from exc
    for key in _STRING_COLUMNS:
        if key in data and data[key] is not None:
            data[key] = str(data[key]).strip()
    return Item(**data)


def _validate_item(item: Item) -> None:
    """Check one item against the schema, raising on the first problem."""
    if not item.item_id:
        raise ItemBankError("item_id must not be empty")
    if item.domain not in DOMAINS:
        raise ItemBankError(f"unknown domain {item.domain!r}; expected one of {list(DOMAINS)}")
    if item.life_stage not in LIFE_STAGES:
        raise ItemBankError(
            f"unknown life_stage {item.life_stage!r}; expected one of {list(LIFE_STAGES)}"
        )
    if not item.stem:
        raise ItemBankError(f"item {item.item_id!r} has an empty stem")
    if not isinstance(item.n_categories, int) or item.n_categories < 2:
        raise ItemBankError(f"item {item.item_id!r}: n_categories must be an integer of at least 2")
    if item.direction not in (-1, 1):
        raise ItemBankError(f"item {item.item_id!r}: direction must be -1 or 1")
    if item.respondent not in RESPONDENTS:
        raise ItemBankError(f"item {item.item_id!r}: unknown respondent {item.respondent!r}")
    if item.subfacet:
        parent = HAEMOPHILIA_SUBFACETS.get(item.subfacet)
        if parent is None:
            raise ItemBankError(
                f"item {item.item_id!r}: unknown subfacet {item.subfacet!r}; "
                f"expected one of {sorted(HAEMOPHILIA_SUBFACETS)}"
            )
        if parent != item.domain:
            raise ItemBankError(
                f"item {item.item_id!r}: subfacet {item.subfacet!r} belongs to "
                f"{parent!r}, not {item.domain!r}"
            )


def _apply_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Give the frame stable dtypes so a CSV round trip compares equal."""
    out = frame.copy()
    for name in _STRING_COLUMNS:
        out[name] = out[name].astype("string")
    out["n_categories"] = out["n_categories"].astype("int64")
    out["direction"] = out["direction"].astype("int64")
    return out


def _as_set(
    value: str | Sequence[str] | None, allowed: Sequence[str], label: str
) -> set[str] | None:
    """Normalise a filter argument to a set, checking the vocabulary."""
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ItemBankError(f"unknown {label}: {bad}; expected from {list(allowed)}")
    return set(values)


def write_data_dictionary(path: str | Path) -> Path:
    """Write the schema of the item bank to CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file.

    Returns
    -------
    pathlib.Path
        The path written.

    Notes
    -----
    The file describes the columns an item table must have. It is the schema,
    not the items. ``data/dictionary.csv`` in the repository is produced by this
    function, and a test checks that the two agree.
    """
    out = Path(path)
    frame = pd.DataFrame(
        list(SCHEMA),
        columns=["field", "dtype", "required", "allowed_values", "description"],
    )
    frame.to_csv(out, index=False)
    return out


def example_bank() -> HPotentialItemBank:
    """Return a small synthetic bank used by the tests and the example script.

    Returns
    -------
    HPotentialItemBank
        Twelve invented items, four per domain, spread over the life stages.

    Notes
    -----
    The stems are written for this supplement. They are not validated items,
    have not been through cognitive interviewing, and must not be fielded.
    """
    rows = [
        (
            "BP01",
            "Biomedical Potential",
            "adult",
            "Difficulty walking 500 metres",
            "Functional Capacity",
            -1,
        ),
        (
            "BP02",
            "Biomedical Potential",
            "adult",
            "Joint pain on most days",
            "Functional Capacity",
            -1,
        ),
        (
            "BP03",
            "Biomedical Potential",
            "ageing",
            "Tiredness limiting daily activity",
            "Functional Capacity",
            -1,
        ),
        (
            "BP04",
            "Biomedical Potential",
            "paediatric",
            "Keeping up with peers in play",
            "Functional Capacity",
            1,
        ),
        (
            "SM01",
            "Social Modifiers",
            "adult",
            "Someone to help after a bleed",
            "Autonomy and Agency",
            1,
        ),
        (
            "SM02",
            "Social Modifiers",
            "transition",
            "Confidence managing own treatment",
            "Autonomy and Agency",
            1,
        ),
        (
            "SM03",
            "Social Modifiers",
            "adult",
            "Decisions made together with the care team",
            "Autonomy and Agency",
            1,
        ),
        (
            "SM04",
            "Social Modifiers",
            "paediatric",
            "School makes allowance for treatment",
            "Autonomy and Agency",
            1,
        ),
        (
            "SB01",
            "Systemic Burdens",
            "adult",
            "Treatment unavailable when needed",
            "Structural Support",
            -1,
        ),
        (
            "SB02",
            "Systemic Burdens",
            "adult",
            "Travel time to the treatment centre",
            "Structural Support",
            -1,
        ),
        (
            "SB03",
            "Systemic Burdens",
            "ageing",
            "Cost of care met out of pocket",
            "Structural Support",
            -1,
        ),
        (
            "SB04",
            "Systemic Burdens",
            "transition",
            "Referrals arranged without delay",
            "Structural Support",
            1,
        ),
    ]
    bank = HPotentialItemBank()
    bank.extend(
        Item(
            item_id=item_id,
            domain=domain,
            life_stage=stage,
            stem=stem,
            subfacet=subfacet,
            direction=direction,
            n_categories=5,
            respondent="self" if stage != "paediatric" else "proxy",
            source="synthetic example",
        )
        for item_id, domain, stage, stem, subfacet, direction in rows
    )
    return bank
