#!/usr/bin/env python3
"""Targeted PAC boundary implementation stress, separate from study results.

This adds two fixed synthetic N=5000 fixtures because m=500 can query most of a
small N=240 pool. It does not rerun or replace the exhaustive/core verification.
The fixtures are fixed before sampling and have baseline error just above eps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from unittest.mock import patch
import verify_guarantees as verify

HERE=Path(__file__).resolve().parent


def frozen_large_pools(_seed):
    n=5000
    pred=np.zeros(n,dtype=int)
    scores=(np.arange(n)*32//n)/32.
    balanced_gold=np.zeros(n,dtype=int)
    fixture_rng=np.random.default_rng(38017)
    balanced_gold[fixture_rng.choice(n,170,replace=False)]=1
    misranked_gold=np.zeros(n,dtype=int)
    misranked_gold[:170]=1
    already={int(i):0 for i in range(500,900)}
    return [
        ("large_32levels_distributed_risk_034",pred.copy(),scores.copy(),balanced_gold,{}),
        ("large_32levels_misranked_risk_034_owned400",pred.copy(),scores.copy(),misranked_gold,already),
    ]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=HERE.parent)
    parser.add_argument("--output",type=Path,default=HERE/"large_pool_pac_report.json")
    parser.add_argument("--reps",type=int,default=10000)
    args=parser.parse_args()
    if args.reps<=0:
        parser.error("--reps must be positive")
    checks=verify.Checks()
    _,pac=verify.load_source(args.source_root)
    files=[Path(__file__),Path(verify.__file__),args.source_root/"audit_repair"/"pac.py"]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    start=time.monotonic()
    report={"schema_version":1,"artifact_kind":"targeted_implementation_verification_not_study_results",
            "status":"in_progress","configuration":{"n":5000,"m":500,"epsilon":.03,
            "alpha_final":.1,"seed":20261106,"repetitions_per_fixture":args.reps,
            "fixture_count":2},"source_and_script_sha256":hashes,
            "motivation":"m=500 labels most of the N=240 pools in the main verification. These two frozen larger pools retain a substantial unqueried set and start just above the final-error tolerance.",
            "scope":"Conditional PAC calls on frozen synthetic predictions, gold and scores; no real study data or E1-E4 outcome is loaded.",
            "fixture_design":[{"name":"large_32levels_distributed_risk_034","initial_error_count":170,
                               "n":5000,"scores":"32 equally sized levels from 0 through 31/32",
                               "already_owned":0,"error_placement":"170 fixed indices from rng seed 38017"},
                              {"name":"large_32levels_misranked_risk_034_owned400","initial_error_count":170,
                               "n":5000,"scores":"32 equally sized levels; all errors at lowest indices",
                               "already_owned":400,"owned_placement":"truthful zeros at indices 500 through 899"}],
            "limitations":["Two deliberately specified fixed-input conditions do not represent the seven real study pools or all annotation-panel distributions.",
                           "This is an implementation regression check, not a proof of the general guarantee.",
                           "One-sided 95% Clopper-Pearson intervals are individual, not simultaneous. A lower bound above alpha is a statistical regression signal, not a deterministic proof of a defect.",
                           "Repair-stage familywise failure has its separate .05 budget in the main report. The final-error event here is compared only with alpha_final=.10."]}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("START PAC boundary stress: two N=5000, 32-score-level fixtures, epsilon=.03, alpha_final=.10, m=500",flush=True)
    try:
        with patch.object(verify,"make_pac_pools",frozen_large_pools):
            report["results"]=verify.pac_mc_check(pac,checks,args.reps,20261106,.10,.03,500)
    except Exception as exc:
        import traceback
        checks.check(False,"large_pool_execution_error",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
    checks.check(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
                 "large_pool_sources_and_scripts_unchanged")
    report.update({"status":"failed" if checks.failures else "passed_under_tested_conditions",
                   "elapsed_seconds":round(time.monotonic()-start,3),"assertion_count":checks.assertion_count,
                   "failures":checks.failures,"failure_count_retained":len(checks.failures)})
    args.output.write_text(json.dumps(verify.json_safe(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"elapsed_seconds":report["elapsed_seconds"],
                      "assertions":report["assertion_count"],"failures":len(checks.failures),"report":str(args.output)}),flush=True)
    return 1 if checks.failures else 0


if __name__=="__main__":
    raise SystemExit(main())
