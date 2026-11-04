from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
fig=root/'results'/'figures';fig.mkdir(parents=True,exist_ok=True)
e5=pd.read_csv(root/'results/E5/e5_seedcluster_summary.csv')
sel=[('ag_news_5000','G0_uniform','S1'),('ag_news_5000','G1','S1'),('ag_news_holdout','G1','S1')]
d=pd.DataFrame([e5[(e5.pool==a)&(e5.generator==b)&(e5.score==c)].iloc[0] for a,b,c in sel])
labels=['AG News dev G0/S1','AG News dev G1/S1','AG News holdout G1/S1']
fig1,ax=plt.subplots(figsize=(9,4.5))
means=d.mean_joint_saving.to_numpy();lo=d.joint_seedboot95_low.to_numpy();hi=d.joint_seedboot95_high.to_numpy();xs=np.arange(3)
ax.errorbar(xs,means,yerr=np.vstack([means-lo,hi-means]),fmt='o',capsize=5)
ax.axhline(0,linewidth=1,linestyle='--');ax.set_xticks(xs,labels);ax.set_ylabel('Mean expert labels saved (A minus C)');ax.set_title('E5 — joint rescoring, conditional seed-bootstrap intervals');fig1.tight_layout();fig1.savefig(fig/'e5_joint_rescoring.png',dpi=180);plt.close(fig1)
e6=pd.read_csv(root/'results/E6_full/e6_seedcluster_inference.csv')
e6=e6[(e6.policy=='size')&(e6.gain==0.4)]
fig2,ax=plt.subplots(figsize=(9,5))
for (pos,sl),p in e6.groupby(['position','shortlist']):
    p=p.sort_values('size');ax.plot(p['size'],p['mean_savings'],marker='o',label=f'{pos}, {sl} groups')
ax.axhline(0,linewidth=1,linestyle='--');ax.set_xticks([50,100,300]);ax.set_xlabel('Planted group size');ax.set_ylabel('Expert labels saved per run');ax.set_title('E6 — conversion vs score location and shared-budget shortlist');ax.legend(ncol=2,fontsize=9);fig2.tight_layout();fig2.savefig(fig/'e6_conversion_by_position.png',dpi=180);plt.close(fig2)
print(fig)
