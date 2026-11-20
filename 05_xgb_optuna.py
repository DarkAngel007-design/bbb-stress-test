import pandas as pd, numpy as np, warnings, time, sys
warnings.filterwarnings('ignore')
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Cluster import Butina
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score
RDLogger.DisableLog('rdApp.*')
NTRIALS=int(sys.argv[1]) if len(sys.argv)>1 else 20
c=pd.read_csv('data/kumar_clean.csv').reset_index(drop=True); c['scaf']=c.scaf.fillna('')
def flat(s):
    m=Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
c['flat']=c.smi.map(flat); fl=c.flat.unique().tolist(); c['g']=c.flat.map({f:i for i,f in enumerate(fl)})
F=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(f),2,2048) for f in fl]; n=len(F); dist=[]
for i in range(1,n): dist.extend([1-x for x in DataStructs.BulkTanimotoSimilarity(F[i],F[:i])])
cmap={m:k for k,mem in enumerate(Butina.ClusterData(dist,n,0.6,isDistData=True)) for m in mem}; c['clu']=c.g.map(cmap)
GK={'random_rows':None,'random_groups':'g','scaffold_rand':'scaf','cluster_butina':'clu'}
D=[];E=[]
for s in c.smi:
    m=Chem.MolFromSmiles(s); D.append([f(m) for _,f in Descriptors.descList])
    a=np.zeros(2048); DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),a); E.append(a)
D=np.clip(np.nan_to_num(np.array(D,dtype=float),nan=0,posinf=0,neginf=0),-1e6,1e6); E=np.array(E)
X={'desc':D,'desc+ecfp':np.hstack([D,E])}; y=c.y.values
sp=pd.read_csv('results/split_indices.csv'); sp=sp[~sp.split.str.startswith('source')]
def metr(yt,p):
    yh=(p>=.5).astype(int)
    return dict(auroc=roc_auc_score(yt,p),auprc=average_precision_score(yt,p),mcc=matthews_corrcoef(yt,yh),bacc=balanced_accuracy_score(yt,yh),acc=accuracy_score(yt,yh))
def run(nm,seed,view):
    g=sp[(sp.split==nm)&(sp.seed==seed)]; tr=g[g.role=='train'].id.values; te=g[g.role=='test'].id.values
    Xt=X[view]; res=[]
    d=XGBClassifier(n_jobs=2,random_state=seed,verbosity=0).fit(Xt[tr],y[tr])
    res.append(dict(split=nm,seed=seed,view=view,model='XGB_default',**metr(y[te],d.predict_proba(Xt[te])[:,1])))
    key=GK[nm]; grp=c[key].values[tr] if key else np.arange(len(tr))
    itr,iva=next(GroupShuffleSplit(1,test_size=0.2,random_state=seed).split(tr,groups=grp)); a,b=tr[itr],tr[iva]
    def obj(t):
        p=dict(max_depth=t.suggest_int('max_depth',3,10),learning_rate=t.suggest_float('learning_rate',0.01,0.3,log=True),
          subsample=t.suggest_float('subsample',0.5,1.0),colsample_bytree=t.suggest_float('colsample_bytree',0.2,1.0),
          min_child_weight=t.suggest_int('min_child_weight',1,10),reg_lambda=t.suggest_float('reg_lambda',1e-2,10,log=True),
          reg_alpha=t.suggest_float('reg_alpha',1e-3,1,log=True),gamma=t.suggest_float('gamma',0,2))
        m=XGBClassifier(n_estimators=1000,early_stopping_rounds=50,eval_metric='auc',n_jobs=2,random_state=seed,verbosity=0,tree_method='hist',**p).fit(Xt[a],y[a],eval_set=[(Xt[b],y[b])],verbose=False)
        t.set_user_attr('it',int(m.best_iteration)); return m.best_score
    st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed)); st.optimize(obj,n_trials=NTRIALS)
    bp=st.best_params; ne=max(20,int(st.best_trial.user_attrs['it']*1.1))
    m=XGBClassifier(n_estimators=ne,n_jobs=2,random_state=seed,verbosity=0,tree_method='hist',**bp).fit(Xt[tr],y[tr])
    res.append(dict(split=nm,seed=seed,view=view,model='XGB_optuna',inner_auc=st.best_value,n_est=ne,**metr(y[te],m.predict_proba(Xt[te])[:,1])))
    return res
jobs=[(nm,s,v) for nm in GK for s in range(5) for v in X]
t=time.time(); out=Parallel(n_jobs=9)(delayed(run)(*j) for j in jobs); print('time',round(time.time()-t))
r=pd.DataFrame([x for o in out for x in o]); r.to_csv('results/xgb_results_per_run.csv',index=False)
print(r.groupby(['split','view','model'])[['auroc','mcc','bacc','acc']].mean().round(3).to_string())
