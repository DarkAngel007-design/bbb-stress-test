import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score
RDLogger.DisableLog('rdApp.*')
c=pd.read_csv('data/kumar_clean.csv').reset_index(drop=True)
def flat(s):
    m=Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
c['flat']=c.smi.map(flat)
ms=[Chem.MolFromSmiles(s) for s in c.smi]
D=np.clip(np.nan_to_num(np.array([[f(m) for _,f in Descriptors.descList] for m in ms],dtype=float),nan=0,posinf=0,neginf=0),-1e6,1e6)
fp=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(f),2,2048) for f in c.flat]
c['tpsa']=[rdMolDescriptors.CalcTPSA(m) for m in ms]; c['mw']=[Descriptors.MolWt(m) for m in ms]
y=c.y.values; sp=pd.read_csv('results/split_indices.csv'); sp=sp[~sp.split.str.startswith('source')]
rows=[]
for (nm,seed),g in sp.groupby(['split','seed']):
    tr=g[g.role=='train'].id.values; te=g[g.role=='test'].id.values
    m=LGBMClassifier(n_estimators=300,learning_rate=0.05,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=seed,verbose=-1,n_jobs=8).fit(D[tr],y[tr])
    p=m.predict_proba(D[te])[:,1]; trf=[fp[i] for i in tr]
    ms_=[max(DataStructs.BulkTanimotoSimilarity(fp[i],trf)) for i in te]
    rows.append(pd.DataFrame(dict(split=nm,seed=seed,id=te,y=y[te],p=p,maxsim=ms_)))
r=pd.concat(rows).merge(c[['id' if 'id' in c else 'smi','src','tpsa','mw','smi']].reset_index().rename(columns={'index':'id'})[['id','src','tpsa','mw','smi']],on='id')
r['pred']=(r.p>=.5).astype(int); r['err']=(r.pred!=r.y).astype(int)
r['bin']=pd.cut(r.maxsim,[0,.3,.4,.5,.6,.7,.8,.9,1.001],right=False)
r.to_csv('results/test_predictions_lgbm.csv',index=False)
def summ(d):
    return pd.Series(dict(n=len(d),err_rate=d.err.mean(),pos=d.y.mean(),auroc=roc_auc_score(d.y,d.p) if d.y.nunique()>1 else np.nan,bacc=balanced_accuracy_score(d.y,d.pred)))
t=r.groupby('bin').apply(summ).round(3); print('POOLED by similarity-to-train bin'); print(t.to_string()); t.to_csv('results/error_by_similarity_bin.csv')
print('\nBY SPLIT'); print(r.groupby('split').apply(summ).round(3).to_string())
print('\nBY SOURCE (cluster+scaffold splits)'); print(r[r.split.isin(['scaffold_rand','cluster_butina'])].groupby('src').apply(summ).round(3).to_string())
hard=r[r.split=='cluster_butina']
fn=hard[(hard.y==1)&(hard.pred==0)]; fp_=hard[(hard.y==0)&(hard.pred==1)]; tp=hard[(hard.y==1)&(hard.pred==1)]; tn=hard[(hard.y==0)&(hard.pred==0)]
print('\nCLUSTER SPLIT error types: FN',len(fn),'FP',len(fp_),'of',len(hard))
for nm,d in [('TP',tp),('FN',fn),('TN',tn),('FP',fp_)]: print(nm,'median TPSA',round(d.tpsa.median(),1),'median MW',round(d.mw.median(),1),'median maxsim',round(d.maxsim.median(),2))
# compound-level: repeatedly misclassified
cm=hard.groupby('id').agg(n=('err','size'),errs=('err','sum')).reset_index(); print('\ncompounds in cluster-split test sets:',len(cm),'| wrong in every appearance (n>=2):',int(((cm.n>=2)&(cm.errs==cm.n)).sum()),'of',int((cm.n>=2).sum()))
hard.groupby('id').filter(lambda d: len(d)>=2 and d.err.all()).drop_duplicates('id')[['smi','y','src','tpsa','maxsim']].head(15).to_csv('results/persistent_errors_cluster.csv',index=False)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(12,4.5)); tt=t.reset_index()
ax[0].bar(range(len(tt)),tt.err_rate,color='#c44'); ax[0].set_xticks(range(len(tt))); ax[0].set_xticklabels([str(b) for b in tt.bin],rotation=40,ha='right',fontsize=8)
for i,(e,n) in enumerate(zip(tt.err_rate,tt.n)): ax[0].text(i,e+.003,f'n={int(n)}',ha='center',fontsize=7)
ax[0].set_ylabel('error rate'); ax[0].set_xlabel('max Tanimoto to training set'); ax[0].set_title('Error rate vs similarity to training (LightGBM, pooled over splits)')
ax[1].plot(tt.index,tt.auroc,'o-',color='#48a'); ax[1].set_xticks(range(len(tt))); ax[1].set_xticklabels([str(b) for b in tt.bin],rotation=40,ha='right',fontsize=8); ax[1].set_ylabel('AUROC'); ax[1].set_title('AUROC vs similarity bin'); ax[1].set_xlabel('max Tanimoto to training set')
fig.tight_layout(); fig.savefig('05_error_vs_similarity.png',dpi=130)
