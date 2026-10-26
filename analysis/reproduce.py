"""Regenerate all reported synthetic results. Run from repository root."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv, hashlib, json, platform, math
import numpy as np
from hpotential.crf import H, contextual_mean, gap_decomposition, ede, Action, allocate
from hpotential.measurement import GaussianMeasurement

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)


def write_csv(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader();w.writerows(rows)


def main():
    profiles=[('balanced',1,.45,.45),('moderate',1,.60,.30),('unbalanced',1,.70,.20),('extreme',1,.89,.01)]
    rows=[]
    for name,p,s,r in profiles:
        rows.append(dict(profile=name,P=p,S=s,R=r,arithmetic=H(p,s,r,1),geometric=H(p,s,r),harmonic=H(p,s,r,-1),bottleneck=H(p,s,r,-math.inf)))
    write_csv('aggregation.csv',rows)
    sensitivity=[]
    for a in (.25,.5,.75):
        for rho in (-20,-5,-2,-1,-.5,0,1):
            for name,p,s,r in profiles:
                sensitivity.append(dict(profile=name,alpha=a,rho=rho,H=H(p,s,r,rho,a)))
    write_csv('sensitivity.csv',sensitivity)

    # Monte Carlo demonstration: fixed, known model; no estimated CFA.
    rng=np.random.default_rng(20261025)
    phi=np.array([[1,.3,.2],[.3,1,.4],[.2,.4,1]])
    load=np.zeros((9,3))
    for j in range(9): load[j,j//3]=(.6,.75,.9)[j%3]
    psi=1-np.sum(load**2,axis=1)
    names=[f'indicator_{j+1}' for j in range(9)]
    model=GaussianMeasurement(names,load,phi,psi)
    # Repeat common latent draws under missingness and intercept drift.
    z=rng.multivariate_normal(np.zeros(3),phi,size=4000)
    x=z@load.T+rng.normal(size=(4000,9))*np.sqrt(psi)
    miss=rng.random(x.shape)<.3
    shifted=x.copy(); shifted[:,3:6]+=.5
    measure=[]
    for label,xx in [('complete',x),('missing_30pct',np.where(miss,np.nan,x)),('social_intercept_shift',shifted)]:
        means,covs=model.score(dict(zip(names,xx.T)))
        for k in range(3):
            errors=means[:,k]-z[:,k]
            measure.append(dict(scenario=label,domain=k+1,n=len(z),correlation=float(np.corrcoef(means[:,k],z[:,k])[0,1]),rmse=float(np.sqrt(np.mean(errors**2))),bias=float(np.mean(errors)),posterior_sd=float(np.mean(np.sqrt(covs[:,k,k]))),coverage95=float(np.mean(np.abs(errors)<=1.95996398454*np.sqrt(covs[:,k,k])))))
    write_csv('measurement.csv',measure)
    write_csv('determinacy.csv',[dict(domain=k+1,implied=float(d)) for k,d in enumerate(model.implied_determinacy())])

    population=[(1,.2,.2),(1,.8,.8)]
    actions=[Action('support_low',1,((0,.1,.1),(0,0,0))),Action('support_high',1,((0,0,0),(0,.2,.2)))]
    policy=[]
    for eps in (0,.5,1,2,3):
        best=allocate(population,actions,1,eps)
        for a in actions:
            post=[tuple(p[k]+a.changes[i][k] for k in range(3)) for i,p in enumerate(population)]
            values=[H(*p) for p in post]
            policy.append(dict(epsilon=eps,action=a.name,cost=a.cost,mean=sum(values)/len(values),ede=ede(values,eps),optimal=a.name in best['selected']))
    write_csv('policy.csv',policy)
    write_csv('gap.csv',[dict(zip(('biomedical','social','structural','total'),gap_decomposition((.3,.7,.85),(.3,.7,.2))))])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))}
    manifest={'seed':20261025,'numpy':np.__version__,'python':platform.python_version(),'platform':platform.platform(),'n_synthetic':4000,'clinical_data':False,'command':'python analysis/reproduce.py','csv_sha256':hashes}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'tables':len(hashes),'synthetic_n':4000,'measurement':measure,'policy':policy},indent=2))


if __name__=='__main__': main()
