#!/usr/bin/env python3
"""Independent small-domain numerical audit of the mathematical betting contracts.

Uses scalar Decimal/Fraction reference calculations (no calls into PAC/CS core
inside the reference). Reports finite exhaustive and adversarial checks; does NOT
claim a general floating-point proof or independent reproduction of the ICML code.
"""
from __future__ import annotations
from collections import Counter
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib,itertools,json,math,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REF=Path(os.environ.get('V83_REFERENCE',str(ROOT.parent/'AuditRepair-v8.3')))
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(REF/'study/corrected'))
from fast_pac import pac_labels_fast
from audit_repair.pac import pac_labels
from audit_repair.core import WorLowerCS


def fractional_capital(xs,N,m0,c=Fraction(1,2)):
    """Scalar rational e-process, implementing the specified predictable rule."""
    s=Fraction(0);ss=Fraction(0);capital=Fraction(1)
    history=[]
    for t,x in enumerate(xs):
        rem=N-t; mw=(N*m0-s)/rem
        if mw<0:
            capital=None
        elif mw==0:
            if x>0:capital=None
        elif mw < 1 and capital is not None:
            mu=(Fraction(1,2)+s)/(t+1)
            var=(Fraction(1,4)+ss)/(t+1)
            gap=mu-mw
            lam=min(max(gap/(var+gap*gap),0),c/mw) if gap>0 else Fraction(0)
            mul=1+lam*(x-mw)
            if mul<0:raise AssertionError('Negative e-process factor')
            capital*=mul
        # mw>=1: freeze capital; null cannot be rejected on such step.
        mu=(Fraction(1,2)+s)/(t+1)
        ss+=(x-mu)**2;s+=x
        history.append(capital)
    return history


def exact_wor_checks():
    counts={'paths':0,'pointwise_null_paths':0,'null_exceedances':0,'violations':0,
            'float_integer_gate_disagreements':0,'max_abs_logcapital_diff':0.0}
    # All 3-valued finite populations of size <=4, every distinct ordering;
    # exact fractional null grid points near 0/1 and at 1/2.
    grid=[Fraction(0),Fraction(1,400),Fraction(1,2),Fraction(399,400),Fraction(1)]
    failures=[]
    for N in range(2,5):
        for population in itertools.combinations_with_replacement([Fraction(0),Fraction(1,2),Fraction(1)],N):
            mean=sum(population,Fraction(0))/N
            orders=set(itertools.permutations(population))
            for m0 in grid:
                under_null=mean<=m0
                rejected_count=0
                for order in orders:
                    ref=fractional_capital(order,N,m0)
                    if under_null:
                        counts['pointwise_null_paths']+=1
                        exceed=any(cap is None or cap>=10 for cap in ref)
                        rejected_count+=int(exceed)
                    counts['paths']+=1
                    # Compare finite capital of a directly computed sample path
                    # to actual vectorized process at same hypothesis grid index.
                    if m0 in (Fraction(0),Fraction(1,2),Fraction(1)):
                        cs=WorLowerCS(N,alpha=.1,grid=401)
                        j=int(400*m0)
                        for i,x in enumerate(order):
                            cs.update(float(x));cap=ref[i]
                            if cap is None:expected=float('inf')
                            else:expected=math.log(float(cap))
                            actual=float(cs.log_cap[j])
                            if math.isinf(actual)!=math.isinf(expected) or (math.isfinite(actual) and abs(actual-expected)>1e-10):
                                counts['violations']+=1
                                if len(failures)<10: failures.append({'N':N,'m0':str(m0),'order':[str(z) for z in order], 't':i,'actual':actual,'expected':expected})
                            if math.isfinite(actual) and math.isfinite(expected):
                                counts['max_abs_logcapital_diff']=max(counts['max_abs_logcapital_diff'],abs(actual-expected))
                        # Check exact integer decision vs a separate Fraction formula.
                        j=cs.lower_grid_index
                        audited_gain=sum(int(2*x)-1 for x in order)
                        nominal=N*(Fraction(2*j,400)-1)-audited_gain
                        if cs.remainder_is_positive()!=(cs.t<N and nominal>0 and Fraction(*cs.lower.as_integer_ratio())*2*N-N-audited_gain>0):
                            counts['float_integer_gate_disagreements']+=1
                if under_null:
                    counts['null_exceedances']+=rejected_count
                    if Fraction(rejected_count,len(orders))>Fraction(1,10):
                        counts['violations']+=1
                        if len(failures)<10:failures.append({'N':N,'m0':str(m0),'population':[str(z) for z in population],'type':'Ville exceedance'})
    return counts,failures


def decimal_pac(pred,U,gold,draws,already,eps=Decimal('.03'),alpha=Decimal('.10'),c=Decimal('.50')):
    """Scalar high-precision terminal-capital PAC reference for fixed draws."""
    n=len(pred)
    p=list(map(int,pred));u=list(map(float,U))
    for i,y in already.items():p[i]=int(y);u[i]=-1.0
    us=sorted(set(v for v in u if v>=0.0))
    threshold=math.inf
    with localcontext() as ctx:
        ctx.prec=70
        boundary=(Decimal(1)/alpha).ln()
        for candidate in us:
            s=Decimal(0);ss=Decimal(0);logcap=Decimal(0)
            for t,d in enumerate(draws):
                z=Decimal(int(p[d]!=gold[d] and u[d]<=candidate))
                mu=(eps/2+s)/Decimal(t+1)
                var=(Decimal(1)/4+ss)/Decimal(t+1)
                gap=eps-mu
                lam=max(Decimal(0),min(gap/(var+gap*gap),c/(1-eps))) if gap>0 else Decimal(0)
                logcap+=(Decimal(1)+lam*(eps-z)).ln()
                ss+=(z-mu)**2;s+=z
            if logcap<boundary:
                threshold=float(candidate)
                break
    draws_set=set(map(int,draws))
    reviewed={i for i,value in enumerate(u) if value>=threshold}
    bought=(draws_set|reviewed)-set(already)
    f=p.copy()
    for i in draws_set|reviewed:f[i]=int(gold[i])
    return threshold,len(bought),float(sum(int(f[i]!=gold[i]) for i in range(n))/n)


def terminal_pac_checks():
    stats={'scenarios':0,'matched':0,'violations':0,'max_abs_cutoff_diff':0.0,
           'degenerate_cases':0,'ties':0,'repeated_draw_sequences':0}
    failures=[]
    for n in (16,35,80):
        for trial in range(12):
            rng=np.random.default_rng([20261104,n,trial])
            gold=rng.integers(0,3,n)
            pred=gold.copy();pred[rng.choice(n,max(1,n//6),replace=False)]=(pred[rng.choice(n,max(1,n//6),replace=False)]+1)%3
            if trial%4==0:U=np.full(n,.5);stats['degenerate_cases']+=1
            elif trial%4==1:U=rng.choice([0.0,.2,.8,1.0],n);stats['ties']+=1
            elif trial%4==2:U=rng.random(n)
            else:U=np.full(n,-1.)
            owned={int(i):int(gold[i]) for i in range(min(5,n))} if trial%2 else {}
            m=80
            draw_seed=np.random.SeedSequence([20261104,n,trial,71])
            draws=np.random.default_rng(draw_seed).integers(0,n,m)
            if len(set(draws))<m:stats['repeated_draw_sequences']+=1
            threshold,cost,err=decimal_pac(pred,U,gold,draws,owned)
            fa,info=pac_labels_fast(pred,U,gold,eps=.03,alpha=.1,m=m,already=owned,seed=draw_seed)
            oa,orig=pac_labels(pred,U,gold,eps=.03,alpha=.1,m=m,already=owned,seed=draw_seed)
            stats['scenarios']+=1
            np.testing.assert_array_equal(fa,oa)
            if info!=orig:raise AssertionError('Optimized-original PAC implementation mismatch')
            if info['expert_new']==cost and info['error']==err and (info['u_hat']==threshold or (math.isinf(info['u_hat']) and math.isinf(threshold))):
                stats['matched']+=1
            else:
                stats['violations']+=1
                failures.append({'n':n,'trial':trial,'decimal_threshold':threshold,'float_threshold':info['u_hat'],
                                'decimal_new_labels':cost,'float_new_labels':info['expert_new']})
    return stats,failures


def main():
    a,b=exact_wor_checks();c,d=terminal_pac_checks()
    report={'source':'reference PAC Labels and WoR betting mathematical contracts; independent Fraction/Decimal implementations',
            'scope':'small exhaustive and adversarial numerical cases; no general guarantee proof',
            'wor':a,'terminal_pac':c,'failures':b+d,
            'status':'passed' if a['violations']==0 and a['float_integer_gate_disagreements']==0 and c['violations']==0 else 'failed',
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=ROOT/'results'/'E8';out.mkdir(exist_ok=True,parents=True)
    (out/'e8_independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__': main()
