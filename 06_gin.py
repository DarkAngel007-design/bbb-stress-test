import pandas as pd, numpy as np, warnings, time, sys, itertools
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as Fn
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score
RDLogger.DisableLog('rdApp.*')
SMOKE=len(sys.argv)>1 and sys.argv[1]=='smoke'
c=pd.read_csv('data/kumar_clean.csv').reset_index(drop=True); c['scaf']=c.scaf.fillna('')
def flat(s):
    m=Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
c['flat']=c.smi.map(flat); fl=c.flat.unique().tolist(); c['g']=c.flat.map({f:i for i,f in enumerate(fl)})
F=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(f),2,2048) for f in fl]; n=len(F); dist=[]
for i in range(1,n): dist.extend([1-x for x in DataStructs.BulkTanimotoSimilarity(F[i],F[:i])])
cmap={m:k for k,mem in enumerate(Butina.ClusterData(dist,n,0.6,isDistData=True)) for m in mem}; c['clu']=c.g.map(cmap)
GK={'random_rows':None,'random_groups':'g','scaffold_rand':'scaf','cluster_butina':'clu'}
ATOMS=[6,7,8,9,15,16,17,35,53,5,14,34]
HYB=[Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3,Chem.HybridizationType.SP3D,Chem.HybridizationType.SP3D2]
def oh(v,l): return [float(v==x) for x in l]+[float(v not in l)]
def graph(smi):
    m=Chem.MolFromSmiles(smi); xs=[]
    for a in m.GetAtoms():
        xs.append(oh(a.GetAtomicNum(),ATOMS)+oh(a.GetDegree(),[0,1,2,3,4,5])+oh(a.GetFormalCharge(),[-1,0,1])+oh(a.GetHybridization(),HYB)+[float(a.GetIsAromatic()),float(a.IsInRing())]+oh(a.GetTotalNumHs(),[0,1,2,3]))
    e=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in m.GetBonds()]; e+= [(j,i) for i,j in e]
    ei=torch.tensor(e,dtype=torch.long).t() if e else torch.zeros((2,0),dtype=torch.long)
    return torch.tensor(xs,dtype=torch.float),ei
G=[graph(s) for s in c.smi]; NF=G[0][0].shape[1]
D=[[f(Chem.MolFromSmiles(s)) for _,f in Descriptors.descList] for s in c.smi]
D=np.clip(np.nan_to_num(np.array(D,dtype=float),nan=0,posinf=0,neginf=0),-1e6,1e6); y=c.y.values.astype(np.float32)
print('graphs',len(G),'node feats',NF,flush=True)
def batch(idx,Dz):
    xs=[];es=[];bs=[];off=0
    for k,i in enumerate(idx):
        x,e=G[i]; xs.append(x); es.append(e+off); bs.append(torch.full((x.shape[0],),k,dtype=torch.long)); off+=x.shape[0]
    return torch.cat(xs),torch.cat(es,1),torch.cat(bs),(torch.tensor(Dz[idx],dtype=torch.float) if Dz is not None else None)
class GIN(nn.Module):
    def __init__(s,h,L,dr,nd):
        super().__init__(); s.inp=nn.Linear(NF,h)
        s.mlps=nn.ModuleList([nn.Sequential(nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Linear(h,h)) for _ in range(L)])
        s.bns=nn.ModuleList([nn.BatchNorm1d(h) for _ in range(L)]); s.eps=nn.Parameter(torch.zeros(L)); s.dr=dr; s.nd=nd
        s.head=nn.Sequential(nn.Linear(2*h+(64 if nd else 0),h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,1))
        if nd: s.dproj=nn.Sequential(nn.Linear(nd,64),nn.ReLU())
    def forward(s,x,ei,b,d):
        h=s.inp(x)
        for l,(m,bn) in enumerate(zip(s.mlps,s.bns)):
            agg=torch.zeros_like(h); 
            if ei.shape[1]: agg.index_add_(0,ei[1],h[ei[0]])
            h=Fn.dropout(Fn.relu(bn(m((1+s.eps[l])*h+agg))),s.dr,s.training)+h
        B=int(b.max())+1; sm=torch.zeros(B,h.shape[1]).index_add_(0,b,h); cnt=torch.zeros(B).index_add_(0,b,torch.ones(len(b))).unsqueeze(1)
        z=torch.cat([sm/cnt,sm/20.],1)
        if s.nd: z=torch.cat([z,s.dproj(d)],1)
        return s.head(z).squeeze(1)
def predict(model,idx,Dz):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(idx),256):
            x,e,b,d=batch(idx[i:i+256],Dz); out.append(torch.sigmoid(model(x,e,b,d)))
    return torch.cat(out).numpy()
def fit(a,b,cfg,seed,Dz,maxep=80,pat=15):
    torch.manual_seed(seed); np.random.seed(seed); h,L,dr=cfg
    model=GIN(h,L,dr,Dz.shape[1] if Dz is not None else 0); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=(-1,None,0); rng=np.random.RandomState(seed)
    for ep in range(maxep):
        model.train(); perm=rng.permutation(a)
        for i in range(0,len(perm),64):
            ii=perm[i:i+64]
            if len(ii)<2: continue
            x,e,bb,d=batch(ii,Dz); loss=Fn.binary_cross_entropy_with_logits(model(x,e,bb,d),torch.tensor(y[ii])); opt.zero_grad(); loss.backward(); opt.step()
        auc=roc_auc_score(y[b],predict(model,b,Dz))
        if auc>best[0]: best=(auc,{k:v.clone() for k,v in model.state_dict().items()},ep)
        elif ep-best[2]>=pat: break
    model.load_state_dict(best[1]); return model,best[0],best[2]
GRID=[(64,3,0.1),(128,3,0.1),(128,4,0.3),(256,3,0.3)]
sp=pd.read_csv('results/split_indices.csv'); sp=sp[~sp.split.str.startswith('source')]
def run(nm,seed,variant):
    torch.set_num_threads(2)
    g=sp[(sp.split==nm)&(sp.seed==seed)]; tr=g[g.role=='train'].id.values; te=g[g.role=='test'].id.values
    key=GK[nm]; grp=c[key].values[tr] if key else np.arange(len(tr))
    itr,iva=next(GroupShuffleSplit(1,test_size=0.2,random_state=seed).split(tr,groups=grp)); a,b=tr[itr],tr[iva]
    Dz=None
    if variant=='GIN_desc':
        mu=D[a].mean(0); sd=D[a].std(0)+1e-6; Dz=np.clip((D-mu)/sd,-5,5)
    best=None
    for cfg in (GRID[:1] if SMOKE else GRID):
        m,auc,ep=fit(a,b,cfg,seed,Dz,maxep=5 if SMOKE else 80)
        if best is None or auc>best[1]: best=(m,auc,cfg,ep)
    m,auc,cfg,ep=best; p=predict(m,te,Dz); yh=(p>=.5).astype(int); yt=y[te]
    return dict(split=nm,seed=seed,model=variant,inner_auc=auc,cfg=str(cfg),epoch=ep,auroc=roc_auc_score(yt,p),auprc=average_precision_score(yt,p),mcc=matthews_corrcoef(yt,yh),bacc=balanced_accuracy_score(yt,yh),acc=accuracy_score(yt,yh))
jobs=[(nm,s,v) for nm in GK for s in range(5) for v in ['GIN','GIN_desc']]
if SMOKE: jobs=jobs[:2]
t=time.time(); out=Parallel(n_jobs=1 if SMOKE else 9)(delayed(run)(*j) for j in jobs); print('time',round(time.time()-t))
r=pd.DataFrame(out); r.to_csv('results/gin_results_per_run.csv' if not SMOKE else 'results/gin_smoke.csv',index=False)
print(r.groupby(['split','model'])[['auroc','mcc','bacc','acc']].mean().round(3).to_string())
