"""The committed results regenerate, and the paper tables quote them correctly."""

import csv
import importlib.util
import json
import math
import re
import tempfile
import unittest
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPERS = ROOT / "papers"

_spec = importlib.util.spec_from_file_location("reproduce", ROOT / "analysis/reproduce.py")
reproduce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reproduce)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paper_table(name, header_start):
    """Rows of the first Markdown table whose header begins with ``header_start``."""
    lines = (PAPERS / name).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| " + header_start))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def close(a, b, rel=1e-9, abs_=1e-12):
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    except ValueError:
        return a == b


class ReproduceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.manifest = reproduce.main(["--out", str(cls.out), "--quiet"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_regenerated_tables_match_committed(self):
        committed = json.loads((RESULTS / "manifest.json").read_text())
        self.assertEqual(sorted(committed["csv_sha256"]), sorted(self.manifest["csv_sha256"]))
        for name in committed["csv_sha256"]:
            new, old = read_csv(self.out / name), read_csv(RESULTS / name)
            self.assertEqual(len(new), len(old), name)
            for a, b in zip(new, old):
                self.assertEqual(list(a), list(b), name)
                for key in a:
                    self.assertTrue(close(a[key], b[key]), f"{name}:{key} {a[key]} != {b[key]}")

    def test_committed_hashes_are_current(self):
        committed = json.loads((RESULTS / "manifest.json").read_text())
        self.assertEqual(committed["seed"], reproduce.SEED)
        import hashlib

        for name, digest in committed["csv_sha256"].items():
            self.assertEqual(
                hashlib.sha256((RESULTS / name).read_bytes()).hexdigest(), digest, name
            )


class PaperTableTests(unittest.TestCase):
    """Each number in a manuscript table equals the result rounded to 4 dp."""

    def assert_quoted(self, quoted, value):
        # Round half up on the decimal string, as the manuscripts do
        # (coverage 3815/4000 = 0.95375 is quoted as 0.9538).
        expected = Decimal(value).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        self.assertEqual(quoted, str(expected))

    def test_paper1_aggregation_table(self):
        rows = paper_table("01_capability_realisation.md", "S | R")
        data = read_csv(RESULTS / "aggregation.csv")
        self.assertEqual(len(rows), len(data))
        for quoted, row in zip(rows, data):
            self.assertEqual(float(quoted[0]), float(row["S"]))
            self.assertEqual(float(quoted[1]), float(row["R"]))
            for cell, key in zip(quoted[2:], ("arithmetic", "geometric", "harmonic", "bottleneck")):
                self.assert_quoted(cell, row[key])

    def test_paper2_measurement_table(self):
        rows = paper_table("02_measurement_uncertainty.md", "Condition")
        data = read_csv(RESULTS / "measurement.csv")
        self.assertEqual(len(rows), len(data))
        for quoted, row in zip(rows, data):
            self.assertEqual(quoted[1], row["domain"])
            for cell, key in zip(quoted[2:], ("correlation", "rmse", "bias", "coverage95")):
                self.assert_quoted(cell, row[key])

    def test_paper3_policy_table(self):
        rows = paper_table("03_equity_policy.md", "Aversion")
        data = read_csv(RESULTS / "policy.csv")
        by_eps = {}
        for row in data:
            by_eps.setdefault(float(row["epsilon"]), {})[row["action"]] = row
        self.assertEqual(len(rows), len(by_eps))
        for quoted in rows:
            entry = by_eps[float(quoted[0])]
            self.assert_quoted(quoted[1], entry["support_low"]["ede"])
            self.assert_quoted(quoted[2], entry["support_high"]["ede"])
            chosen = "L" if entry["support_low"]["optimal"] == "True" else "H"
            self.assertEqual(quoted[3], chosen)

    def test_prose_figures(self):
        text = (PAPERS / "02_measurement_uncertainty.md").read_text(encoding="utf-8")
        drift = next(
            r
            for r in read_csv(RESULTS / "measurement.csv")
            if r["scenario"] == "social_intercept_shift" and r["domain"] == "2"
        )
        self.assertIn(f"{float(drift['coverage95']):.4f}", text)
        abstract = (PAPERS / "01_capability_realisation.md").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"0\.3742 and 0\.4500", abstract))


if __name__ == "__main__":
    unittest.main()
