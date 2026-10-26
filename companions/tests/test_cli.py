import csv
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from hpotential.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


def run(*argv):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_score_then_ede_matches_committed_example(self):
        scored = self.dir / "scores.csv"
        code, out, _ = run("score", str(ROOT / "examples/profiles.csv"), str(scored))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["rows"], 3)
        with scored.open() as f, (ROOT / "results/example_scores.csv").open() as g:
            got, want = list(csv.DictReader(f)), list(csv.DictReader(g))
        for a, b in zip(got, want):
            self.assertEqual(a["id"], b["id"])
            self.assertAlmostEqual(float(a["H"]), float(b["H"]), places=12)
        code, out, _ = run("ede", str(scored), "--epsilon", "1")
        self.assertEqual(code, 0)
        self.assertAlmostEqual(json.loads(out)["ede"], 0.2513760195071536, places=12)

    def test_weighted_ede(self):
        path = self.write("w.csv", "id,H,weight\na,1,3\nb,4,1\n")
        code, out, _ = run("ede", str(path))
        self.assertEqual(code, 0)
        self.assertAlmostEqual(json.loads(out)["ede"], 1.75)

    def test_excel_bom_is_accepted(self):
        path = self.write("bom.csv", "﻿id,P,S,R\nx,1,0.5,0.5\n")
        code, _, err = run("score", str(path), str(self.dir / "o.csv"))
        self.assertEqual(code, 0, err)

    def test_clean_errors(self):
        cases = [
            ("id,P,S\nx,1,0.5\n", "columns"),
            ("id,P,S,R\n", "no data rows"),
            ("id,P,S,R\nx,1,abc,0.5\n", "not a number"),
            ("id,P,S,R\nx,1,1.5,0.5\n", "S must lie in [0, 1]"),
            ("id,P,S,R\nx,1,nan,0.5\n", "finite"),
        ]
        for text, message in cases:
            path = self.write("bad.csv", text)
            code, _, err = run("score", str(path), str(self.dir / "o.csv"))
            self.assertEqual(code, 2, text)
            self.assertIn(message, err)
            self.assertNotIn("Traceback", err)

    def test_output_must_differ(self):
        path = self.write("p.csv", "id,P,S,R\nx,1,0.5,0.5\n")
        code, _, err = run("score", str(path), str(path))
        self.assertEqual(code, 2)
        self.assertIn("differ", err)

    def test_missing_file(self):
        code, _, err = run("ede", str(self.dir / "nope.csv"))
        self.assertEqual(code, 2)
        self.assertIn("Cannot read", err)


if __name__ == "__main__":
    unittest.main()
