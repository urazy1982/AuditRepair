import sys
from pathlib import Path
import os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REF=Path(os.environ.get('V83_REFERENCE',str(ROOT.parent/'AuditRepair-v8.3'))).resolve()
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(REF/'study/corrected'))
from fast_pac import pac_labels_fast
from audit_repair.pac import pac_labels

def check_pair(p,u,g,already=None):
    for seed in [0,1,3]:
        a,ai=pac_labels(p,u,g,eps=.03,alpha=.1,m=150,already=already,seed=seed)
        b,bi=pac_labels_fast(p,u,g,eps=.03,alpha=.1,m=150,already=already,seed=seed)
        np.testing.assert_array_equal(a,b)
        assert ai==bi,(ai,bi)

def test_random():
    for n in [30,200,1000]:
        for trial in range(10):
            r=np.random.default_rng(123+trial)
            g=r.integers(0,3,n);p=g.copy();ix=r.choice(n,int(.17*n),replace=False);p[ix]=(p[ix]+1)%3
            u=r.random(n)
            check_pair(p,u,g)

def test_ties():
    for n in [30,150]:
        r=np.random.default_rng(n);g=r.integers(0,3,n);p=r.integers(0,3,n)
        u=r.choice([0,.1,.1,.5,.8,.99],size=n)
        check_pair(p,u,g,dict(zip(range(10),g[:10])))

def test_degenerate():
    r=np.random.default_rng(6)
    n=100;g=r.integers(0,2,n)
    for p in [g.copy(),1-g,np.zeros(n,dtype=int)]:
        for u in [np.zeros(n),np.ones(n)*.5,np.full(n,-1.),np.arange(n)/n]:
            check_pair(p,u,g)

if __name__=='__main__':
    for f in [test_random,test_ties,test_degenerate]:
        f();print('PASS',f.__name__)
