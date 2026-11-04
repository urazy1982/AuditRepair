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


def frozen_boundary_pools(_seed):
    n=5000
    pred=np.zeros(n,dtype=int)
    tied_gold=np.zeros(n,dtype=int)
    tied_gold[:170]=1  # 3.4% before calibration; labels are fixed, not redrawn.
    low_score_gold=np.zeros(n,dtype=int)
    low_score_gold[:180]=1  # 3.6%, concentrated in the first tied score bucket.
    low_scores=np.full(n,.5)
    low_scores[:250]=0.
    already={int(i):0 for i in range(250,500)}
    return [
        ("boundary_tied_risk_034",pred.copy(),np.full(n,.5),tied_gold,{}),
        ("boundary_low_score_risk_036_owned250",pred.copy(),low_scores,low_score_gold,already),
    ]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=HERE.parent)
    parser.add_argument("--output",type=Path,default=HERE/"pac_boundary_report.json")
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
            "alpha_final":.1,"seed":20261105,"repetitions_per_fixture":args.reps,
            "fixture_count":2},"source_and_script_sha256":hashes,
            "motivation":"m=500 labels most of the N=240 pools in the main verification. These two frozen larger pools retain a substantial unqueried set and start just above the final-error tolerance.",
            "scope":"Conditional PAC calls on frozen synthetic predictions, gold and scores; no real study data or E1-E4 outcome is loaded.",
            "fixture_design":[{"name":"boundary_tied_risk_034","initial_error_count":170,
                               "n":5000,"scores":"all .5","already_owned":0},
                              {"name":"boundary_low_score_risk_036_owned250","initial_error_count":180,
                               "n":5000,"scores":"first 250 at 0, remaining at .5; owned indices 250..499 are truthful zeros",
                               "already_owned":250}],
            "limitations":["Two deliberately specified fixed-input conditions do not represent the seven real study pools or all annotation-panel distributions.",
                           "This is an implementation regression check, not a proof of the general guarantee.",
                           "One-sided 95% Clopper-Pearson intervals are individual, not simultaneous. A lower bound above alpha is a statistical regression signal, not a deterministic proof of a defect.",
                           "Repair-stage familywise failure has its separate .05 budget in the main report. The final-error event here is compared only with alpha_final=.10."]}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("START PAC boundary stress: two N=5000 fixtures, epsilon=.03, alpha_final=.10, m=500",flush=True)
    try:
        with patch.object(verify,"make_pac_pools",frozen_boundary_pools):
            report["results"]=verify.pac_mc_check(pac,checks,args.reps,20261105,.10,.03,500)
    except Exception as exc:
        import traceback
        checks.check(False,"boundary_execution_error",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
    checks.check(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
                 "boundary_sources_and_scripts_unchanged")
    report.update({"status":"failed" if checks.failures else "passed_under_tested_conditions",
                   "elapsed_seconds":round(time.monotonic()-start,3),"assertion_count":checks.assertion_count,
                   "failures":checks.failures,"failure_count_retained":len(checks.failures)})
    args.output.write_text(json.dumps(verify.json_safe(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"elapsed_seconds":report["elapsed_seconds"],
                      "assertions":report["assertion_count"],"failures":len(checks.failures),"report":str(args.output)}),flush=True)
    return 1 if checks.failures else 0


if __name__=="__main__":
    raise SystemExit(main())
