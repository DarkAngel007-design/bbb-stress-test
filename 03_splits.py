import pandas as pd, numpy as np, json
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import train_test_split
RDLogger.DisableLog('rdApp.*')
c=pd.read_csv('data/kumar_clean.csv').reset_index(drop=True); c['id']=c.index
def flat(s):
    m=Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
c['flat']=c.smi.map(flat); c['scaf']=c.scaf.fillna('')
fl=c.flat.unique().tolist(); fidx={f:i for i,f in enumerate(fl)}; c['g']=c.flat.map(fidx)
fp={f:AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(f),2,2048) for f in fl}
F=[fp[f] for f in fl]
# Butina clusters on flat structures
n=len(F); dist=[]
for i in range(1,n):
    s=DataStructs.BulkTanimotoSimilarity(F[i],F[:i]); dist.extend([1-x for x in s])
cl=Butina.ClusterData(dist,n,0.6,isDistData=True); cmap={}
for k,mem in enumerate(cl):
    for m in mem: cmap[m]=k
c['clu']=c.g.map(cmap); print('flat structures',n,'butina clusters',len(cl))
TF=0.25; rows=[]
def group_split(key,seed):
    rng=np.random.RandomState(seed); keys=np.array(c[key].unique().tolist(),dtype=object); rng.shuffle(keys)
    sizes=c.groupby(key).size(); te=[];tot=0
    for k in keys:
        if tot>=TF*len(c): break
        if tot+sizes[k]>1.15*TF*len(c): continue   # skip oversized groups
        te.append(k); tot+=sizes[k]
    return c[c[key].isin(te)].id.values
splits={}
for s in range(5):
    tr,te=train_test_split(c.id.values,test_size=TF,stratify=c.y,random_state=s); splits[('random_rows',s)]=te
    splits[('random_groups',s)]=group_split('g',s)
    splits[('scaffold_rand',s)]=group_split('scaf',s)
    splits[('cluster_butina',s)]=group_split('clu',s)
for i,src in enumerate(['Roy','Zhao','Shen']):
    te=c[c.src==src].id.values; splits[('source_holdout_'+src,0)]=te
res=[]; sim_store={}
for (nm,s),te in splits.items():
    te=np.array(sorted(te)); tr=np.array(sorted(set(c.id)-set(te)))
    if nm.startswith('source'):  # remove test compounds whose flat structure also occurs in train
        trg=set(c.g[tr]); te=np.array([i for i in te if c.g[i] not in trg]); 
    trf=sorted(set(c.g[tr])); trF=[F[g] for g in trf]
    ms=np.array([max(DataStructs.BulkTanimotoSimilarity(F[g],trF)) for g in c.g[te]])
    sim_store[(nm,s)]=ms
    res.append(dict(split=nm,seed=s,n_train=len(tr),n_test=len(te),test_pos=round(c.y[te].mean(),3),
      med_maxsim=round(float(np.median(ms)),3),frac_ge_0_9=round(float((ms>=.9).mean()),3),frac_ge_0_7=round(float((ms>=.7).mean()),3),
      test_scaf_in_train=round(float(c.scaf[te].isin(set(c.scaf[tr])).mean()),3),
      test_flat_in_train=round(float(c.g[te].isin(set(c.g[tr])).mean()),3)))
    for i in te: rows.append((nm,s,int(i),'test'))
    for i in tr: rows.append((nm,s,int(i),'train'))
r=pd.DataFrame(res); r.to_csv('results/split_summary_per_seed.csv',index=False)
agg=r.groupby('split').agg(n_test=('n_test','mean'),test_pos=('test_pos','mean'),med_maxsim=('med_maxsim','mean'),sim_sd=('med_maxsim','std'),
  ge_0_9=('frac_ge_0_9','mean'),ge_0_7=('frac_ge_0_7','mean'),scaf_in_train=('test_scaf_in_train','mean'),flat_in_train=('test_flat_in_train','mean')).round(3)
agg.to_csv('results/split_summary.csv'); print(agg.to_string())
pd.DataFrame(rows,columns=['split','seed','id','role']).merge(c[['id','smi','y','src']],on='id').to_csv('results/split_indices.csv',index=False)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,5)); names=['random_rows','random_groups','scaffold_rand','cluster_butina','source_holdout_Roy','source_holdout_Zhao','source_holdout_Shen']
data=[np.concatenate([v for (nm,s),v in sim_store.items() if nm==x]) for x in names]
ax.boxplot(data,vert=False,tick_labels=names,showfliers=False); ax.set_xlabel('max Tanimoto (ECFP4) of each test compound to the training set'); ax.set_title('How novel is the test set under each split? (Kumar 3,482 compounds)'); ax.invert_yaxis()
fig.tight_layout(); fig.savefig('results/02_split_similarity.png',dpi=130)
