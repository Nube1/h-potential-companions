import math
import unittest
import numpy as np
from hpotential.crf import H, contextual_mean, gradients, gap_decomposition, ede, ede_gradient, Action, allocate
from hpotential.measurement import GaussianMeasurement, anchored


class CRFTests(unittest.TestCase):
    def test_axioms_grid(self):
        for p in (0, .1, 1, 30):
            for s in (0, .01, .4, 1):
                for r in (0, .01, .7, 1):
                    for rho in (0, -1, -800, -math.inf):
                        h=H(p,s,r,rho)
                        self.assertGreaterEqual(h,0)
                        self.assertLessEqual(h,p+1e-12)
                        self.assertLessEqual(h,H(p,min(1,s+.01),r,rho)+1e-12)
                        if s*r==0: self.assertEqual(h,0)
            self.assertAlmostEqual(H(p,1,1),p)

    def test_generalized_mean_and_limits(self):
        for s,r in ((.7,.2),(.001,.99),(.4,.4)):
            vals=[contextual_mean(s,r,rho) for rho in (-math.inf,-800,-2,-1,0,1)]
            self.assertTrue(all(a<=b+1e-12 for a,b in zip(vals,vals[1:])))
            self.assertAlmostEqual(contextual_mean(s,r,-1),1/(.5/s+.5/r))
            self.assertAlmostEqual(contextual_mean(s,r,1e-10),math.sqrt(s*r),places=9)
        self.assertTrue(math.isfinite(contextual_mean(1e-250,.8,-800)))

    def test_geometric_gradient_finite_difference(self):
        x=[.55,.35,.4]; d=1e-6
        for j,g in enumerate(gradients(*x)):
            a=x.copy(); b=x.copy(); a[j]+=d; b[j]-=d
            self.assertAlmostEqual(g,(H(*a)-H(*b))/(2*d),places=8)

    def test_gap_and_units(self):
        a=(.3,.7,.85); b=(.3,.7,.2)
        *parts,total=gap_decomposition(a,b)
        self.assertAlmostEqual(total,math.log(H(*a)/H(*b)))
        self.assertAlmostEqual(H(30,.7,.2),100*H(.3,.7,.2))

    def test_boundary_and_invalid(self):
        for args in ((-1,.2,.3),(1,1.1,.3),(1,float('nan'),.3)):
            with self.assertRaises(ValueError): H(*args)
        with self.assertRaises(ValueError): gradients(1,0,.2)
        with self.assertRaises(ValueError): gap_decomposition((0,.3,.4),(1,.3,.4))
        self.assertEqual(contextual_mean(0,1),0)

    def test_ede_analytic_and_zero(self):
        self.assertAlmostEqual(ede([1,4],1),2)
        self.assertAlmostEqual(ede([1,4],2),1.6)
        self.assertAlmostEqual(ede([0,4],.5),1)
        self.assertEqual(ede([0,4],1),0)
        self.assertEqual(ede([0,4],2),0)
        self.assertAlmostEqual(ede([1,4],0,[3,1]),1.75)
        self.assertAlmostEqual(ede([1,4],1+1e-9),2,places=7)
        vals=[ede([.1,.4,.9],e) for e in (0,.5,1,2,3)]
        self.assertTrue(all(a>=b for a,b in zip(vals,vals[1:])))

    def test_population_gradient(self):
        x=[.1,.4,.9]; d=1e-6
        for j,g in enumerate(ede_gradient(x,2,[1,2,3])):
            a=x.copy();b=x.copy();a[j]+=d;b[j]-=d
            self.assertAlmostEqual(g,(ede(a,2,[1,2,3])-ede(b,2,[1,2,3]))/(2*d),places=7)

    def test_allocator_exact(self):
        p=[(1,.2,.2),(1,.8,.8)]
        acts=[Action('low',1,((0,.1,.1),(0,0,0))),Action('high',1,((0,0,0),(0,.2,.2)))]
        self.assertEqual(allocate(p,acts,1,0)['selected'],['high'])
        self.assertEqual(allocate(p,acts,1,2)['selected'],['low'])
        self.assertEqual(allocate(p,acts,0)['selected'],[])


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.model=GaussianMeasurement(['x1','x2'],[[1],[1]],[[1]],[1,1])

    def test_gaussian_analytic(self):
        m,v=self.model.score({'x1':[1.,np.nan, np.nan],'x2':[2.,2.,np.nan]})
        np.testing.assert_allclose(m[:,0],[1,1,0])
        np.testing.assert_allclose(v[:,0,0],[1/3,1/2,1])
        self.assertAlmostEqual(self.model.implied_determinacy()[0],math.sqrt(2/3))

    def test_outcome_rejected(self):
        with self.assertRaises(ValueError): self.model.score({'x1':[1],'x2':[2],'death':[1]})

    def test_missingness_cannot_add_information(self):
        _,v=self.model.score({'x1':[1,np.nan,np.nan],'x2':[2,2,np.nan]})
        self.assertLess(v[0,0,0],v[1,0,0])
        self.assertLess(v[1,0,0],v[2,0,0])

    def test_constant_drift_preserves_correlation_but_biases(self):
        x=np.arange(20.)
        baseline,_=self.model.score({'x1':x,'x2':x})
        shifted,_=self.model.score({'x1':x+.5,'x2':x+.5})
        np.testing.assert_allclose(shifted-baseline,1/3)
        self.assertAlmostEqual(np.corrcoef(baseline[:,0],shifted[:,0])[0,1],1)

    def test_anchors_frozen(self):
        np.testing.assert_allclose(anchored([0,5,10],0,10),[0,.5,1])
        np.testing.assert_allclose(anchored([0,10],0,10,False),[1,0])
        with self.assertRaises(ValueError): anchored([11],0,10)


if __name__=='__main__': unittest.main()
