"""Dimensionless context, capacity in declared units; no clinical calibration."""
import math
from dataclasses import dataclass


def _finite(x, name):
    if not math.isfinite(x):
        raise ValueError(f'{name} must be finite')


def _profile(p, s, r, alpha):
    for x, name in zip((p, s, r, alpha), ('P', 'S', 'R', 'alpha')):
        _finite(x, name)
    if p < 0 or not (0 <= s <= 1 and 0 <= r <= 1) or not 0 < alpha < 1:
        raise ValueError('Require P >= 0, S,R in [0,1], alpha in (0,1)')


def contextual_mean(s, r, rho=0.0, alpha=0.5):
    """Stable CES mean. Positive rho is an explicit compensatory comparator.

    rho=-inf is the exact bottleneck limit. No hidden epsilon at zero.
    """
    _profile(1, s, r, alpha)
    if math.isnan(rho) or rho == math.inf:
        raise ValueError('rho must be finite or -infinity')
    if rho == -math.inf:
        return min(s, r)
    if s == 0 or r == 0:
        if rho <= 0:
            return 0.0
        return (alpha * s**rho + (1-alpha) * r**rho)**(1/rho)
    ls, lr = math.log(s), math.log(r)
    mu = alpha*ls + (1-alpha)*lr
    # Cumulant expansion avoids cancellation as rho -> 0.
    if abs(rho) * abs(ls-lr) < 1e-6:
        variance = alpha*(1-alpha)*(ls-lr)**2
        return math.exp(mu + rho*variance/2)
    a, b = rho*ls, rho*lr
    m = max(a, b)
    return math.exp((m + math.log(alpha*math.exp(a-m)+(1-alpha)*math.exp(b-m)))/rho)


def H(p, s, r, rho=0.0, alpha=0.5):
    _profile(p, s, r, alpha)
    return p * contextual_mean(s, r, rho, alpha)


def gradients(p, s, r, alpha=0.5):
    """Geometric CRF derivatives, only on the strictly positive interior."""
    _profile(p, s, r, alpha)
    if min(p, s, r) <= 0:
        raise ValueError('Interior derivatives require positive P,S,R')
    h = H(p, s, r, alpha=alpha)
    return h/p, alpha*h/s, (1-alpha)*h/r


def gap_decomposition(a, b, alpha=0.5):
    """Log ratio of two positive comparable profiles, not group mean scores."""
    _profile(*a, alpha)
    _profile(*b, alpha)
    if min(*a, *b) <= 0:
        raise ValueError('Log decomposition needs strictly positive profiles')
    parts = (math.log(a[0])-math.log(b[0]),
             alpha*(math.log(a[1])-math.log(b[1])),
             (1-alpha)*(math.log(a[2])-math.log(b[2])))
    return (*parts, math.fsum(parts))


def ede(values, epsilon=0, weights=None):
    """Weighted Atkinson equally distributed equivalent, with exact zero limits."""
    v = list(values)
    if not v or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Require a nonempty population and finite epsilon >= 0')
    if any(not math.isfinite(x) or x < 0 for x in v):
        raise ValueError('Scores must be finite and nonnegative')
    w = [1.0]*len(v) if weights is None else list(weights)
    if len(w) != len(v) or any(not math.isfinite(x) or x < 0 for x in w) or sum(w) <= 0:
        raise ValueError('Invalid population weights')
    pairs = [(x, q/sum(w)) for x,q in zip(v,w) if q > 0]
    if all(x == 0 for x,q in pairs):
        return 0.0
    if epsilon >= 1 and any(x == 0 for x,q in pairs):
        return 0.0
    if epsilon == 1:
        return math.exp(math.fsum(q*math.log(x) for x,q in pairs))
    power = 1-epsilon
    if abs(power) < 1e-6 and all(x > 0 for x,q in pairs):
        mu = math.fsum(q*math.log(x) for x,q in pairs)
        var = math.fsum(q*(math.log(x)-mu)**2 for x,q in pairs)
        return math.exp(mu+power*var/2)
    terms = [(power*math.log(x),q) for x,q in pairs if x > 0]
    m = max(t for t,q in terms)
    return math.exp((m+math.log(math.fsum(q*math.exp(t-m) for t,q in terms)))/power)


def ede_gradient(values, epsilon=0, weights=None):
    v=list(values)
    if any(x <= 0 for x in v):
        raise ValueError('Population gradient requires strictly positive scores')
    total=ede(v, epsilon, weights)
    w=[1.0]*len(v) if weights is None else list(weights)
    z=sum(w)
    return [q/z*(total/x)**epsilon for x,q in zip(v,w)]


@dataclass(frozen=True)
class Action:
    name: str
    cost: float
    changes: tuple  # one (dP,dS,dR) for each population member


def allocate(profiles, actions, budget, epsilon=0, alpha=0.5, weights=None):
    """Exact finite portfolio enumeration, additive specified domain responses.

    Actions are binary, jointly feasible, not automatically clipped to a bound.
    Exponential complexity; intentionally capped at 20 candidate actions.
    """
    if not math.isfinite(budget) or budget < 0 or len(actions)>20:
        raise ValueError('Invalid budget or too many actions')
    if len({a.name for a in actions}) != len(actions):
        raise ValueError('Action names must be unique')
    for p in profiles:
        _profile(*p,alpha)
    for a in actions:
        if not math.isfinite(a.cost) or a.cost<0 or len(a.changes)!=len(profiles):
            raise ValueError('Invalid action cost or population response shape')
        if any(len(d)!=3 or any(not math.isfinite(x) for x in d) for d in a.changes):
            raise ValueError('Invalid action response')
    best=None
    for mask in range(1<<len(actions)):
        chosen=[a for j,a in enumerate(actions) if mask & (1<<j)]
        cost=math.fsum(a.cost for a in chosen)
        if cost>budget:
            continue
        post=[tuple(p[k]+sum(a.changes[i][k] for a in chosen) for k in range(3))
              for i,p in enumerate(profiles)]
        if any(p<0 or not(0<=s<=1 and 0<=r<=1) for p,s,r in post):
            continue
        scores=[H(*p,alpha=alpha) for p in post]
        obj=ede(scores,epsilon,weights)
        row={'selected':[a.name for a in chosen], 'cost':cost, 'ede':obj, 'scores':scores}
        if best is None or obj>best['ede']+1e-14 or (abs(obj-best['ede'])<=1e-14 and cost<best['cost']):
            best=row
    return best
