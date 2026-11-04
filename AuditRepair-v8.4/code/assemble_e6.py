#!/usr/bin/env python3
"""Merge shard checkpoints into a complete read-only analytical E6 release.

Run only after the five independent shard jobs terminate. Existing checkpoints
are never overwritten, and every expected frozen cell/seed identity is checked.
"""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from e6_synthetic import cells,summarize

def main():
    ph=hashlib.sha256((ROOT/'protocol/frozen_v84.json').read_bytes()).hexdigest()
    shards=[ROOT/'results/E6',*[ROOT/f'results/E6_shard_{i}_{j}' for i,j in ((11,22),(22,33),(33,44),(44,54))]]
    dest=ROOT/'results/E6_full'
    for shard in shards:
        man=json.loads((shard/'e6_manifest.json').read_text())
        assert man['protocol_sha256']==ph,shard
        assert man['per_cell_independent_seeds']==100,shard
    dest.mkdir(parents=True,exist_ok=True)
    total=0
    for cell in cells():
        cid=cell['cell_id']
        src=shards[0] if cid<11 else shards[1] if cid<22 else shards[2] if cid<33 else shards[3] if cid<44 else shards[4]
        srcdir=src/'checkpoints'/f'cell_{cid:02d}'
        target=dest/'checkpoints'/f'cell_{cid:02d}'
        target.mkdir(parents=True,exist_ok=True)
        for seed in range(100):
            file=srcdir/f'seed_{seed:03d}.json';blob=file.read_bytes();obj=json.loads(blob)
            assert obj['protocol_sha256']==ph and obj['cell']==cell and obj['seed']==seed and obj['status']=='complete',file
            assert len(obj['rows'])==15 and len(obj['states'])==3,file
            dst=target/file.name
            if dst.exists():
                assert dst.read_bytes()==blob,dst
            else:dst.write_bytes(blob)
            total+=1
    assert total==5400
    pairs,states,summary=summarize(dest,cells(),0,100)
    assert pairs==81000 and states==16200
    man={'status':'complete','cells':54,'per_cell_independent_seeds':100,'PAC_pairs':pairs,'policy_states':states,
         'protocol_sha256':ph,'source_shards':[str(s.relative_to(ROOT)) for s in shards],
         'reconciliation':'exact canonical per-cell per-seed checkpoint merge; all 5400 identities verified',
         'scientific_scope':'controlled new synthetic populations, not fresh human or LLM labels'}
    (dest/'e6_manifest.json').write_text(json.dumps(man,indent=2)+'\n')
    print(json.dumps(man,indent=2))
if __name__=='__main__':main()
