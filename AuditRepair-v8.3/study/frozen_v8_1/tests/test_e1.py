import unittest
from unittest.mock import patch
import numpy as np

from audit_repair.pac import pac_labels
from experiment_common import check_pac_accounting, changed_diagnostics
from run_e1 import e1_masks


class E1Tests(unittest.TestCase):
    def test_full_gold_attains_fixed_query_floor(self):
        n=3000
        gold=np.arange(n)%4
        u=np.linspace(0,1,n)
        already={int(i):int(gold[i]) for i in range(400)}
        final,info=pac_labels(gold.copy(),u,gold,eps=.03,alpha=.1,m=500,already=already,seed=812)
        check=check_pac_accounting(info,u,already,n,500,812)
        self.assertEqual(info['threshold_new'],0)
        self.assertEqual(info['u_hat'],np.inf)
        self.assertEqual(len(already)+info['expert_new'],check['minimum_fixed_cost'])
        np.testing.assert_array_equal(final,gold)

    def test_prefix_oracle_is_not_unrestricted_ceiling(self):
        n=2000
        base=np.zeros(n,dtype=int)
        gold=base.copy();gold[:20]=1;gold[1000:1020]=1
        u=np.r_[np.full(1000,.1),np.full(1000,.8)]
        draws=np.r_[np.arange(20),np.arange(1000,1020),np.arange(200,660)]
        class FixedDraws:
            def integers(self,low,high,size):
                assert low==0 and high==n and size==500
                return draws.copy()
        with patch('audit_repair.pac.np.random.default_rng',return_value=FixedDraws()):
            _,b=pac_labels(base,u,gold,eps=.03,alpha=.1,m=500)
            o4=base.copy();mask=(u<=b['u_hat'])&(base!=gold);o4[mask]=gold[mask]
            _,p=pac_labels(o4,u,gold,eps=.03,alpha=.1,m=500)
            _,full=pac_labels(gold.copy(),u,gold,eps=.03,alpha=.1,m=500)
        self.assertEqual(b['u_hat'],.1)
        self.assertEqual(p['u_hat'],.8)
        self.assertEqual(full['u_hat'],np.inf)
        self.assertGreater(p['expert_new'],full['expert_new'])
        self.assertEqual(full['expert_new'],500)

    def test_threshold_ties_and_empty_denominators(self):
        base=np.zeros(4,dtype=int);gold=np.ones(4,dtype=int)
        u=np.array([.1,.2,.2,.8]);pred=gold.copy()
        d=changed_diagnostics(pred,base,gold,u,.2,{})
        self.assertEqual((d['changed_below'],d['changed_at'],d['changed_above']),(1,2,1))
        self.assertEqual(d['share_changed_le'],.75)
        z=changed_diagnostics(base,base,gold,u,.2,{})
        self.assertIsNone(z['share_changed_le'])

    def test_missing_votes_are_not_complete_unanimity(self):
        L=np.array([[0,0,0,0],[0,0,0,-1],[0,1,0,-1],[-1,-1,-1,-1]])
        wrong,dis,uni=e1_masks(L,np.zeros(4,dtype=int),np.ones(4,dtype=int),{})
        np.testing.assert_array_equal(dis,[False,False,True,False])
        np.testing.assert_array_equal(uni,[True,False,False,False])
        self.assertEqual(int((wrong&~dis&~uni).sum()),2)

    def test_paired_stream_replay(self):
        ss=np.random.SeedSequence([20261030,0]).spawn(3)[2].spawn(10)[0]
        a=np.random.default_rng(ss).integers(0,3000,500)
        b=np.random.default_rng(ss).integers(0,3000,500)
        np.testing.assert_array_equal(a,b)


if __name__=='__main__': unittest.main()
