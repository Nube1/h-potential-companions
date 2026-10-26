import math
import unittest

import numpy as np

from hpotential.measurement import GaussianMeasurement, anchored


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.model = GaussianMeasurement(["x1", "x2"], [[1], [1]], [[1]], [1, 1])

    def test_gaussian_analytic(self):
        m, v = self.model.score({"x1": [1.0, np.nan, np.nan], "x2": [2.0, 2.0, np.nan]})
        np.testing.assert_allclose(m[:, 0], [1, 1, 0])
        np.testing.assert_allclose(v[:, 0, 0], [1 / 3, 1 / 2, 1])
        self.assertAlmostEqual(self.model.implied_determinacy()[0], math.sqrt(2 / 3))

    def test_outcome_rejected(self):
        with self.assertRaises(ValueError):
            self.model.score({"x1": [1], "x2": [2], "death": [1]})

    def test_missingness_cannot_add_information(self):
        _, v = self.model.score({"x1": [1, np.nan, np.nan], "x2": [2, 2, np.nan]})
        self.assertLess(v[0, 0, 0], v[1, 0, 0])
        self.assertLess(v[1, 0, 0], v[2, 0, 0])

    def test_constant_drift_preserves_correlation_but_biases(self):
        x = np.arange(20.0)
        baseline, _ = self.model.score({"x1": x, "x2": x})
        shifted, _ = self.model.score({"x1": x + 0.5, "x2": x + 0.5})
        np.testing.assert_allclose(shifted - baseline, 1 / 3)
        self.assertAlmostEqual(np.corrcoef(baseline[:, 0], shifted[:, 0])[0, 1], 1)

    def test_anchors_frozen(self):
        np.testing.assert_allclose(anchored([0, 5, 10], 0, 10), [0, 0.5, 1])
        np.testing.assert_allclose(anchored([0, 10], 0, 10, False), [1, 0])
        with self.assertRaises(ValueError):
            anchored([11], 0, 10)

    def test_pattern_grouping_matches_rowwise(self):
        rng = np.random.default_rng(1)
        model = GaussianMeasurement(
            ["a", "b", "c", "d"],
            [[0.8, 0], [0.6, 0], [0, 0.7], [0, 0.9]],
            [[1, 0.3], [0.3, 1]],
            [0.36, 0.64, 0.51, 0.19],
            [0.1, 0, -0.2, 0],
        )
        x = rng.normal(size=(50, 4))
        x[rng.random(x.shape) < 0.4] = np.nan
        means, covs = model.score(dict(zip("abcd", x.T)))
        for i in range(50):
            m1, c1 = model.score({n: [x[i, j]] for j, n in enumerate("abcd")})
            np.testing.assert_allclose(means[i], m1[0], atol=1e-12)
            np.testing.assert_allclose(covs[i], c1[0], atol=1e-12)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            GaussianMeasurement(["x"], [[1]], [[-1]], [1])
        with self.assertRaises(ValueError):
            GaussianMeasurement(["x"], [[1]], [[1]], [0])
        with self.assertRaises(ValueError):
            GaussianMeasurement(["x", "x"], [[1], [1]], [[1]], [1, 1])
        with self.assertRaises(ValueError):
            GaussianMeasurement(["x"], [[np.nan]], [[1]], [1])

    def test_anchor_nan_passes_through(self):
        self.assertTrue(np.isnan(anchored([np.nan], 0, 1))[0])
        with self.assertRaises(ValueError):
            anchored([0.5], 1, 1)


if __name__ == "__main__":
    unittest.main()
