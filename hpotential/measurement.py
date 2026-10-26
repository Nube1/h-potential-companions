"""Gaussian posterior scoring for an externally specified measurement model.

This is not CFA estimation or a NHATS analysis. No outcomes are accepted.
"""
import numpy as np


class GaussianMeasurement:
    def __init__(self, names, loadings, phi, uniqueness, intercept=None):
        self.names=tuple(names)
        self.loadings=np.asarray(loadings,dtype=float)
        self.phi=np.asarray(phi,dtype=float)
        self.uniqueness=np.asarray(uniqueness,dtype=float)
        self.intercept=np.zeros(len(names)) if intercept is None else np.asarray(intercept,dtype=float)
        p,k=self.loadings.shape
        if p!=len(names) or len(set(names))!=p or self.phi.shape!=(k,k) or self.uniqueness.shape!=(p,) or self.intercept.shape!=(p,):
            raise ValueError('Incompatible measurement dimensions')
        if any(not np.all(np.isfinite(a)) for a in (self.loadings,self.phi,self.uniqueness,self.intercept)):
            raise ValueError('Nonfinite measurement parameter')
        if not np.allclose(self.phi,self.phi.T) or np.any(np.linalg.eigvalsh(self.phi)<=0) or np.any(self.uniqueness<=0):
            raise ValueError('Need positive-definite latent covariance and positive residual variances')

    def score(self, indicators):
        """Returns posterior mean and covariance per row, NaN = missing.

        Dictionary must contain exactly the frozen indicator names: accidental
        outcome columns fail, rather than entering a model silently.
        """
        if set(indicators)!=set(self.names):
            raise ValueError('Input columns must exactly equal frozen indicators')
        x=np.column_stack([indicators[n] for n in self.names]).astype(float)
        if np.any(np.isinf(x)):
            raise ValueError('Infinite indicator values')
        means=[]; covariances=[]
        for row in x:
            obs=~np.isnan(row)
            if not obs.any():
                means.append(np.zeros(self.phi.shape[0])); covariances.append(self.phi.copy()); continue
            l=self.loadings[obs]; psi=self.uniqueness[obs]
            sigma=l@self.phi@l.T+np.diag(psi)
            # solve rather than invert the observed covariance.
            gain=np.linalg.solve(sigma,l@self.phi).T
            means.append(gain@(row[obs]-self.intercept[obs]))
            v=self.phi-gain@l@self.phi
            covariances.append((v+v.T)/2)
        return np.asarray(means),np.asarray(covariances)

    def implied_determinacy(self):
        """Complete-data model-implied correlations, not empirical validation."""
        _,v=self.score({n:[0.] for n in self.names})
        return np.sqrt(np.maximum(0,1-np.diag(v[0])/np.diag(self.phi)))


def anchored(value, lower, upper, higher_is_better=True):
    """Explicit fixed-anchor scaling; rejects extrapolation; preserves NaN."""
    if not np.isfinite(lower) or not np.isfinite(upper) or upper<=lower:
        raise ValueError('Invalid fixed anchors')
    x=np.asarray(value,dtype=float)
    if np.any(np.isinf(x)) or np.any((x<lower)|(x>upper)):
        raise ValueError('Values outside frozen anchors')
    z=(x-lower)/(upper-lower)
    return z if higher_is_better else 1-z
