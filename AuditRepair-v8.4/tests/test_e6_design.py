from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import e6_synthetic as e6

def test_cells():
    cs=e6.cells()
    assert len(cs)==54
    assert len({(c['size'],c['gain'],c['position'],c['groups']) for c in cs})==54

def test_groups_and_gains():
    for c in e6.cells():
        p=e6.make_pool(c,0)
        assert len(p['gold'])==5000
        assert len(p['groups'])==c['groups']
        assert len(np.unique(np.concatenate(p['groups'])))==c['groups']*c['size']
        assert all(g==round(c['gain']*c['size']) for g in p['true_group_gains'])
        assert len(np.unique(p['warm']))==120

def test_no_gold_ranking():
    c=[z for z in e6.cells() if z['groups']==10][0]
    p=e6.make_pool(c,7)
    a={x:e6.ranking(p,x) for x in ('size','cost_aware_no_gold','random')}
    p['gold'][:]=2  # hostile gold replacement has no path into rank policy
    b={x:e6.ranking(p,x) for x in a}
    assert a==b

def test_policy_budget():
    WorLowerCS,_=e6.setup_import(e6.REF)
    for c in [e6.cells()[0],e6.cells()[23],e6.cells()[53]]:
        p=e6.make_pool(c,0)
        for policy in ('size','cost_aware_no_gold','random'):
            s=e6.audit_policy(p,policy,WorLowerCS)
            assert len(s['owned'])==400
            assert s['group_audits']<=196
            assert s['tested_count']<=c['groups']
            assert s['successful_gain']==sum(t['true_unaudited_gain'] for t in s['tested'] if t['decision']=='repair')
            for i,v in s['owned'].items():
                assert v==p['gold'][i]

if __name__=='__main__':
    for t in [test_cells,test_groups_and_gains,test_no_gold_ranking,test_policy_budget]:
        t();print('PASS',t.__name__)
