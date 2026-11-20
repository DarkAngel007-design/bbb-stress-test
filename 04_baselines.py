import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score
from lightgbm import LGBMClassifier
RDLogger.DisableLog('rdApp.*')
c=pd.read_csv('data/kumar_clean.csv').reset_index(drop=True)
names=[n for n,_ in Descriptors.descList]
D=[];E=[];M=[]
for s in c.smi:
    m=Chem.MolFromSmiles(s)
    D.append([f(m) for _,f in Descriptors.descList])
    a=np.zeros(2048); DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),a); E.append(a)
    b=np.zeros(167); DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m),b); M.append(b)
D=np.nan_to_num(np.array(D,dtype=float),nan=0,posinf=0,neginf=0); D=np.clip(D,-1e6,1e6); E=np.array(E); M=np.array(M)
y=c.y.values; print('features',D.shape,E.shape,M.shape,flush=True)
X={'desc':D,'ecfp':E,'desc+ecfp':np.hstack([D,E]),'desc+maccs':np.hstack([D,M])}
models={
 'LR_desc':('desc',lambda s:make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=0.1))),
 'RF_ecfp':('ecfp',lambda s:RandomForestClassifier(500,n_jobs=-1,random_state=s)),
 'RF_desc+ecfp':('desc+ecfp',lambda s:RandomForestClassifier(500,n_jobs=-1,random_state=s)),
 'LGBM_desc':('desc',lambda s:LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=s,verbose=-1,n_jobs=8)),
 'MLP_desc+maccs':('desc+maccs',lambda s:make_pipeline(StandardScaler(),MLPClassifier((256,128),alpha=1e-3,early_stopping=True,max_iter=300,random_state=s))),
}
sp=pd.read_csv('results/split_indices.csv')
out=[]
for (nm,seed),g in sp.groupby(['split','seed']):
    if nm.startswith('source'): continue
    tr=g[g.role=='train'].id.values; te=g[g.role=='test'].id.values
    out.append(dict(split=nm,seed=seed,model='majority',auroc=0.5,auprc=y[te].mean(),mcc=0.0,bacc=0.5,acc=max(y[te].mean(),1-y[te].mean()) if False else float(y[te].mean()>=.5)*y[te].mean()+float(y[te].mean()<.5)*(1-y[te].mean())))
    for mn,(fx,mk) in models.items():
        t=time.time(); clf=mk(seed).fit(X[fx][tr],y[tr]); p=clf.predict_proba(X[fx][te])[:,1]; yh=(p>=.5).astype(int)
        out.append(dict(split=nm,seed=seed,model=mn,auroc=roc_auc_score(y[te],p),auprc=average_precision_score(y[te],p),mcc=matthews_corrcoef(y[te],yh),bacc=balanced_accuracy_score(y[te],yh),acc=accuracy_score(y[te],yh)))
    print(nm,seed,'done',flush=True)
r=pd.DataFrame(out); r.to_csv('results/baseline_results_per_run.csv',index=False)
s=r.groupby(['split','model'])[['auroc','auprc','mcc','bacc','acc']].agg(['mean','std']).round(3)
s.to_csv('results/baseline_summary.csv'); print(s.to_string())
