#!/usr/bin/env python3
"""Comparison of the eight frozen analysis outputs, including cross-platform tolerances."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--actual',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
names=['E1_effects.csv','E1_observed_error.csv','E1_reach.csv','E2_E3_effects.csv','E2_E3_observed_error.csv','E2_E3_primary_contrasts.csv','E2_E3_reach.csv','E4_observed_error.csv']
r={};errors=[]
for n in names:
    ref=a.reference/n;act=a.actual/n
    if not ref.exists() or not act.exists():errors.append('Missing: '+n);continue
    exact=ref.read_bytes()==act.read_bytes()
    if exact:
        r[n]={'byte_identical':True,'rows':len(pd.read_csv(act))};continue
    x=pd.read_csv(ref);y=pd.read_csv(act)
    aligned=(list(x.columns)==list(y.columns) and len(x)==len(y))
    if not aligned:errors.append('Shape/column mismatch: '+n);r[n]={'byte_identical':False,'compatible':False};continue
    diffs={}
    for col in x.columns:
        try:
            v=pd.to_numeric(x[col],errors='raise').to_numpy(float)
            w=pd.to_numeric(y[col],errors='raise').to_numpy(float)
            if not np.allclose(v,w,atol=1e-8,rtol=1e-10,equal_nan=True):
                diffs[col]=float(np.nanmax(np.abs(v-w)))
        except (ValueError,TypeError,OverflowError):
            if not x[col].fillna('NA').astype(str).equals(y[col].fillna('NA').astype(str)):
                diffs[col]='text mismatch'
    if diffs:errors.append('Scientific columns differ: '+n)
    r[n]={'byte_identical':False,'compatible':not bool(diffs),'diff_columns':diffs,'rows':len(x)}
status='exact' if all(x.get('byte_identical') for x in r.values()) else 'numerically_consistent' if not errors else 'failed'
rep={'status':status,'tables':r,'problems':errors,'qualification':'Comparison reuses frozen inputs/seeds; not new inferential samples. Numerical closeness alone is not a guarantee for uninspected downstream decisions.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rep,indent=2)+'\n')
print(json.dumps(rep,indent=2))
if errors:raise SystemExit(2)
