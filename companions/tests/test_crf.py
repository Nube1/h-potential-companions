import math
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from hpotential.crf import (
    Action,
    H,
    allocate,
    contextual_mean,
    ede,
    ede_gradient,
    gap_decomposition,
    gradients,
)

unit = st.floats(0, 1)
positive_unit = st.floats(1e-6, 1)
capacity = st.floats(0, 1e3)
alphas = st.floats(0.01, 0.99)
rhos = st.one_of(st.just(-math.inf), st.floats(-50, 1))


class CRFTests(unittest.TestCase):
    def test_axioms_grid(self):
        for p in (0, 0.1, 1, 30):
            for s in (0, 0.01, 0.4, 1):
                for r in (0, 0.01, 0.7, 1):
                    for rho in (0, -1, -800, -math.inf):
                        h = H(p, s, r, rho)
                        self.assertGreaterEqual(h, 0)
                        self.assertLessEqual(h, p + 1e-12)
                        self.assertLessEqual(h, H(p, min(1, s + 0.01), r, rho) + 1e-12)
                        if s * r == 0:
                            self.assertEqual(h, 0)
            self.assertAlmostEqual(H(p, 1, 1), p)

    def test_generalized_mean_and_limits(self):
        for s, r in ((0.7, 0.2), (0.001, 0.99), (0.4, 0.4)):
            vals = [contextual_mean(s, r, rho) for rho in (-math.inf, -800, -2, -1, 0, 1)]
            self.assertTrue(all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])))
            self.assertAlmostEqual(contextual_mean(s, r, -1), 1 / (0.5 / s + 0.5 / r))
            self.assertAlmostEqual(contextual_mean(s, r, 1e-10), math.sqrt(s * r), places=9)
        self.assertTrue(math.isfinite(contextual_mean(1e-250, 0.8, -800)))

    def test_geometric_gradient_finite_difference(self):
        x = [0.55, 0.35, 0.4]
        d = 1e-6
        for j, g in enumerate(gradients(*x)):
            a = x.copy()
            b = x.copy()
            a[j] += d
            b[j] -= d
            self.assertAlmostEqual(g, (H(*a) - H(*b)) / (2 * d), places=8)

    def test_gap_and_units(self):
        a = (0.3, 0.7, 0.85)
        b = (0.3, 0.7, 0.2)
        *parts, total = gap_decomposition(a, b)
        self.assertAlmostEqual(total, math.log(H(*a) / H(*b)))
        self.assertAlmostEqual(H(30, 0.7, 0.2), 100 * H(0.3, 0.7, 0.2))

    def test_boundary_and_invalid(self):
        for args in ((-1, 0.2, 0.3), (1, 1.1, 0.3), (1, float("nan"), 0.3)):
            with self.assertRaises(ValueError):
                H(*args)
        with self.assertRaises(ValueError):
            gradients(1, 0, 0.2)
        with self.assertRaises(ValueError):
            gap_decomposition((0, 0.3, 0.4), (1, 0.3, 0.4))
        self.assertEqual(contextual_mean(0, 1), 0)

    def test_ede_analytic_and_zero(self):
        self.assertAlmostEqual(ede([1, 4], 1), 2)
        self.assertAlmostEqual(ede([1, 4], 2), 1.6)
        self.assertAlmostEqual(ede([0, 4], 0.5), 1)
        self.assertEqual(ede([0, 4], 1), 0)
        self.assertEqual(ede([0, 4], 2), 0)
        self.assertAlmostEqual(ede([1, 4], 0, [3, 1]), 1.75)
        self.assertAlmostEqual(ede([1, 4], 1 + 1e-9), 2, places=7)
        vals = [ede([0.1, 0.4, 0.9], e) for e in (0, 0.5, 1, 2, 3)]
        self.assertTrue(all(a >= b for a, b in zip(vals, vals[1:])))

    def test_population_gradient(self):
        x = [0.1, 0.4, 0.9]
        d = 1e-6
        for j, g in enumerate(ede_gradient(x, 2, [1, 2, 3])):
            a = x.copy()
            b = x.copy()
            a[j] += d
            b[j] -= d
            self.assertAlmostEqual(
                g, (ede(a, 2, [1, 2, 3]) - ede(b, 2, [1, 2, 3])) / (2 * d), places=7
            )

    def test_allocator_exact(self):
        p = [(1, 0.2, 0.2), (1, 0.8, 0.8)]
        acts = [
            Action("low", 1, ((0, 0.1, 0.1), (0, 0, 0))),
            Action("high", 1, ((0, 0, 0), (0, 0.2, 0.2))),
        ]
        self.assertEqual(allocate(p, acts, 1, 0)["selected"], ["high"])
        self.assertEqual(allocate(p, acts, 1, 2)["selected"], ["low"])
        self.assertEqual(allocate(p, acts, 0)["selected"], [])

    def test_allocator_tolerates_round_off(self):
        # 0.1 + (0.34 + 0.56) evaluates to 1.0000000000000002 in floating point.
        # The best portfolio reaches S = R = 1 exactly and must stay feasible.
        acts = [Action("a", 1, ((0, 0.34, 0.34),)), Action("b", 1, ((0, 0.56, 0.56),))]
        best = allocate([(1, 0.1, 0.1)], acts, 2)
        self.assertEqual(best["selected"], ["a", "b"])
        self.assertAlmostEqual(best["ede"], 1.0)

    def test_allocator_rejects_real_bound_violation(self):
        acts = [Action("too_much", 1, ((0, 0.5, 0),))]
        self.assertEqual(allocate([(1, 0.6, 0.5)], acts, 1)["selected"], [])

    def test_allocator_tie_prefers_cheaper(self):
        acts = [Action("dear", 2, ((0, 0.1, 0.1),)), Action("cheap", 1, ((0, 0.1, 0.1),))]
        self.assertEqual(allocate([(1, 0.2, 0.2)], acts, 2)["selected"], ["cheap"])

    def test_allocator_input_checks(self):
        act = Action("a", 1, ((0, 0.1, 0.1),))
        with self.assertRaises(ValueError):
            allocate([(1, 0.2, 0.2)], [act, act], 2)
        with self.assertRaises(ValueError):
            allocate([(1, 0.2, 0.2)], [Action("a", -1, ((0, 0, 0),))], 2)
        with self.assertRaises(ValueError):
            allocate([(1, 0.2, 0.2)], [Action("a", 1, ((0, 0, 0), (0, 0, 0)))], 2)
        with self.assertRaises(ValueError):
            allocate([(1, 0.2, 0.2)], [Action(str(j), 0, ((0, 0, 0),)) for j in range(21)], 1)
        with self.assertRaises(ValueError):
            allocate([(1, 0.2, 0.2)], [act], float("inf"))

    def test_ede_input_checks(self):
        for bad in (
            ([], 0, None),
            ([1], -1, None),
            ([-1], 0, None),
            ([1, 2], 0, [1]),
            ([1, 2], 0, [0, 0]),
            ([1], float("nan"), None),
        ):
            with self.assertRaises(ValueError):
                ede(*bad)

    def test_zero_weight_member_is_ignored(self):
        self.assertEqual(ede([0, 4], 2, [0, 1]), 4)


class CRFProperties(unittest.TestCase):
    """Randomised checks of the paper 1 axioms and the paper 3 EDE bounds."""

    @settings(max_examples=300, deadline=None)
    @given(capacity, unit, unit, rhos, alphas)
    def test_ceiling_and_nonnegativity(self, p, s, r, rho, alpha):
        h = H(p, s, r, rho, alpha)
        self.assertGreaterEqual(h, 0)
        self.assertLessEqual(h, p * (1 + 1e-12))
        if (s == 0 or r == 0) and rho <= 0:  # rho > 0 is the compensatory comparator
            self.assertEqual(h, 0)

    @settings(max_examples=300, deadline=None)
    @given(positive_unit, positive_unit, alphas)
    def test_mean_between_min_and_max(self, s, r, alpha):
        for rho in (-math.inf, -10, -1, 0, 0.5, 1):
            m = contextual_mean(s, r, rho, alpha)
            self.assertLessEqual(min(s, r) * (1 - 1e-12), m)
            self.assertLessEqual(m, max(s, r) * (1 + 1e-12))

    @settings(max_examples=200, deadline=None)
    @given(positive_unit, positive_unit, st.floats(-5, 1), st.floats(0.01, 0.5), alphas)
    def test_mean_increasing_in_rho(self, s, r, rho, step, alpha):
        lower = contextual_mean(s, r, rho - step, alpha)
        upper = contextual_mean(s, r, rho, alpha)
        self.assertLessEqual(lower, upper * (1 + 1e-12))

    @settings(max_examples=200, deadline=None)
    @given(st.floats(0.01, 100), st.floats(0.01, 100), unit, unit, alphas)
    def test_unit_equivariance(self, c, p, s, r, alpha):
        self.assertTrue(
            math.isclose(
                H(c * p, s, r, 0, alpha), c * H(p, s, r, 0, alpha), rel_tol=1e-12, abs_tol=1e-300
            )
        )

    @settings(max_examples=200, deadline=None)
    @given(
        st.tuples(st.floats(0.01, 10), positive_unit, positive_unit),
        st.tuples(st.floats(0.01, 10), positive_unit, positive_unit),
        alphas,
    )
    def test_gap_parts_sum_to_log_ratio(self, a, b, alpha):
        *parts, total = gap_decomposition(a, b, alpha)
        self.assertAlmostEqual(math.fsum(parts), total, places=12)
        self.assertAlmostEqual(total, math.log(H(*a, alpha=alpha) / H(*b, alpha=alpha)), places=9)

    @settings(max_examples=200, deadline=None)
    @given(st.lists(st.floats(0, 10), min_size=1, max_size=20), st.floats(0, 5))
    def test_ede_between_min_and_mean(self, values, epsilon):
        e = ede(values, epsilon)
        mean = sum(values) / len(values)
        self.assertLessEqual(min(values) * (1 - 1e-9), e)
        self.assertLessEqual(e, mean * (1 + 1e-9) + 1e-300)

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.floats(0.01, 10), min_size=2, max_size=10), st.floats(0, 4))
    def test_ede_gradient_sums_to_one_at_equality(self, values, epsilon):
        equal = [values[0]] * len(values)
        self.assertAlmostEqual(sum(ede_gradient(equal, epsilon)), 1.0, places=9)
