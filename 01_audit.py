import pandas as pd, numpy as np, json
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
lfc=rdMolStandardize.LargestFragmentChooser(); unc=rdMolStandardize.Uncharger()
def std(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        m=unc.uncharge(lfc.choose(m)); return Chem.MolToSmiles(m)
    except Exception: return None
def scaf(smi):
    try: return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception: return None
a=pd.read_csv('data/BBBP.csv'); a=a.rename(columns={'smiles':'raw','p_np':'y'})[['name','raw','y']]
b=pd.read_csv('data/B3DB_classification.tsv',sep='\t'); b=b.rename(columns={'SMILES':'raw'})
b['y']=(b['BBB+/BBB-']=='BBB+').astype(int); b=b[['compound_name','raw','y']].rename(columns={'compound_name':'name'})
rep={}
out={}
for nm,d in [('martins',a),('b3db',b)]:
    n0=len(d); d=d.copy(); d['smi']=d.raw.map(std); fail=int(d.smi.isna().sum()); d=d.dropna(subset=['smi'])
    g=d.groupby('smi').y.agg(['nunique','size','mean'])
    conflict=int((g['nunique']>1).sum()); dup_groups=int((g['size']>1).sum()); dup_rows=int(g['size'].sum()-len(g))
    dd=d[d.smi.isin(g.index[g['nunique']==1])].drop_duplicates('smi').copy()
    dd['scaf']=dd.smi.map(scaf)
    dd['acyclic']=(dd.scaf=='')
    sc=dd.scaf.value_counts()
    ms=[Chem.MolFromSmiles(s) for s in dd.smi]
    dd['mw']=[Descriptors.MolWt(m) for m in ms]; dd['tpsa']=[rdMolDescriptors.CalcTPSA(m) for m in ms]
    dd['hbd']=[rdMolDescriptors.CalcNumHBD(m) for m in ms]
    rep[nm]=dict(rows_raw=n0,parse_fail=fail,dup_extra_rows=dup_rows,conflicting_label_molecules=conflict,
      n_clean=len(dd),pos=int(dd.y.sum()),pos_frac=round(dd.y.mean(),4),majority_acc=round(max(dd.y.mean(),1-dd.y.mean()),4),
      n_scaffolds=int(dd.scaf.nunique()),acyclic=int(dd.acyclic.sum()),singleton_scaffolds=int((sc==1).sum()),
      largest_scaffold_group=int(sc.iloc[0]) if sc.index[0]!='' else int(sc.iloc[1]),
      top_scaffold_nonempty=[(k,int(v)) for k,v in sc.items() if k!=''][:3])
    out[nm]=dd; dd.to_csv(f'data/{nm}_clean.csv',index=False)
# overlap
sa,sb=set(out['martins'].smi),set(out['b3db'].smi)
ov=sa&sb; rep['overlap_martins_in_b3db']=dict(n=len(ov),frac_of_martins=round(len(ov)/len(sa),4))
m=out['martins'].merge(out['b3db'][['smi','y']],on='smi',suffixes=('','_b3db'))
rep['overlap_label_disagree']=int((m.y!=m.y_b3db).sum())
scm,scb=set(out['martins'].scaf),set(out['b3db'].scaf)
rep['scaffold_overlap_martins_in_b3db']=round(out['martins'].scaf.isin(scb).mean(),4)
json.dump(rep,open('results/audit.json','w'),indent=1)
print(json.dumps(rep,indent=1))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,3,figsize=(13,7))
for j,(nm,d) in enumerate(out.items()):
    ax[j,0].bar(['BBB-','BBB+'],[(d.y==0).sum(),(d.y==1).sum()],color=['#c44','#48a']); ax[j,0].set_title(f'{nm}: class balance (n={len(d)})')
    for lab,c in [(0,'#c44'),(1,'#48a')]:
        ax[j,1].hist(d[d.y==lab].tpsa,bins=40,alpha=.6,color=c,label=['BBB-','BBB+'][lab],density=True)
    ax[j,1].set_title('TPSA by class'); ax[j,1].legend(); ax[j,1].set_xlabel('TPSA')
    sc=d.scaf.value_counts(); sc=sc[sc.index!='']
    ax[j,2].hist(sc.values,bins=np.arange(1,max(sc.values)+2),log=True,color='#555'); ax[j,2].set_title('compounds per scaffold (log)'); ax[j,2].set_xlabel('group size')
fig.tight_layout(); fig.savefig('results/01_data_audit.png',dpi=130)
