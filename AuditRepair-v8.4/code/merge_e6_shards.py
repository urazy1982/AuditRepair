#!/usr/bin/env python3
"""Consolidate five disjoint E6 full-run shards after 100/100 seeds per cell."""
from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import e6_synthetic as e6

def main():
    expected=hashlib.sha256((ROOT/'protocol/frozen_v84.json').read_bytes()).hexdigest()
    sources=[(0,11,ROOT/'results/E6'),(11,22,ROOT/'results/E6_shard_11_22'),
             (22,33,ROOT/'results/E6_shard_22_33'),(33,44,ROOT/'results/E6_shard_33_44'),
             (44,54,ROOT/'results/E6_shard_44_54')]
    allcells=e6.cells();target=ROOT/'results/E6_full';target.mkdir(exist_ok=True,parents=True)
    count=0
    for lo,hi,src in sources:
        run=json.loads((src/'e6_manifest.json').read_text())
        if run['status']!='pilot_partial' or run['seed_range']!=[0,100] or run['cells']!=hi-lo:
            raise AssertionError('Shard missing or incomplete')
        if run['protocol_sha256']!=expected:raise AssertionError('Shard protocol changed')
        for ci in range(lo,hi):
            for seed in range(100):
                path=src/'checkpoints'/f'cell_{ci:02d}'/f'seed_{seed:03d}.json'
                obj=json.loads(path.read_text())
                if obj['protocol_sha256']!=expected or obj['cell']!=allcells[ci] or obj['seed']!=seed:
                    raise AssertionError(f'Bad checkpoint {ci} {seed}')
                if len(obj['rows'])!=15 or len(obj['states'])!=3:raise AssertionError('Wrong PAC nesting')
                folder=target/'checkpoints'/f'cell_{ci:02d}';folder.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,folder/path.name);count+=1
    nrows,nstates,s=e6.summarize(target,allcells,0,100)
    status={'status':'complete','cells':54,'seed_range':[0,100],
            'per_cell_independent_seeds':100,'checkpoints':count,'PAC_pairs':nrows,
            'policy_states':nstates,'protocol_sha256':expected,
            'code_sha256':hashlib.sha256((ROOT/'code/e6_synthetic.py').read_bytes()).hexdigest(),
            'merge_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'fresh controlled synthetic populations, NOT new LLM annotations'}
    (target/'e6_manifest.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))
if __name__=='__main__':main()
