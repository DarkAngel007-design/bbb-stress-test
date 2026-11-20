import pandas as pd, numpy as np, json
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
lfc=rdMolStandardize.LargestFragmentChooser(); unc=rdMolStandardize.Uncharger()
def std(s):
    m=Chem.MolFromSmiles(s)
    if m is None: return None
    try: return Chem.MolToSmiles(unc.uncharge(lfc.choose(m)))
    except Exception: return None
def scaf(s):
    try: return MurckoScaffold.MurckoScaffoldSmiles(smiles=s,includeChirality=False)
    except Exception: return None
d=pd.read_excel('data/kumar_table1.xlsx').iloc[:,:5]; d.columns=['sl','raw','pid','y','ref']
d['src']=d.ref.str.extract(r'(Roy|Zhao|Shen)')[0]
rep={'rows':len(d)}
d['smi']=d.raw.map(std); rep['parse_fail']=int(d.smi.isna().sum()); d=d.dropna(subset=['smi'])
g=d.groupby('smi').agg(n=('y','size'),nu=('y','nunique'),ns=('src','nunique'))
rep['unique_smiles']=len(g); rep['extra_dup_rows']=int(g.n.sum()-len(g))
rep['conflict_molecules']=int((g.nu>1).sum())
rep['dups_across_sources']=int(((g.n>1)&(g.ns>1)).sum())
conf=d[d.smi.isin(g.index[g.nu>1])].sort_values('smi')[['smi','y','src']]
conf.to_csv('results/kumar_label_conflicts.csv',index=False)
c=d[d.smi.isin(g.index[g.nu==1])].drop_duplicates('smi').copy()
c['scaf']=c.smi.map(scaf)
rep['n_clean']=len(c); rep['pos']=int(c.y.sum()); rep['pos_frac']=round(c.y.mean(),4)
sc=c.scaf.value_counts(); rep['n_scaffolds']=int(c.scaf.nunique()); rep['acyclic']=int((c.scaf=='').sum())
rep['singleton_scaffolds']=int((sc==1).sum()); rep['largest_nonempty_scaffold']=[(k,int(v)) for k,v in sc.items() if k!=''][:3]
rep['per_source']=c.groupby('src').y.agg(n='size',pos_frac='mean').round(3).reset_index().to_dict('records')
# scaffold overlap between sources
S={s:set(c[c.src==s].scaf) for s in c.src.unique()}
rep['scaffold_share_with_other_sources']={s:round(c[c.src==s].scaf.isin(set().union(*[S[t] for t in S if t!=s])).mean(),3) for s in S}
# overlap with Martins / B3DB (clean files from stage 1)
for nm in ['martins','b3db']:
    o=pd.read_csv(f'data/{nm}_clean.csv'); m=c.merge(o[['smi','y']],on='smi',suffixes=('','_o'))
    rep[f'overlap_{nm}']=dict(n=len(m),frac_of_kumar=round(len(m)/len(c),3),label_disagree=int((m.y!=m.y_o).sum()))
# similarity: nearest-neighbour Tanimoto within the set (leave-one-out), ECFP4
fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in c.smi]
nn=[]
for i,f in enumerate(fps):
    sims=DataStructs.BulkTanimotoSimilarity(f,fps); sims[i]=0; nn.append(max(sims))
c['nn']=nn; rep['nn_tanimoto_median']=round(float(np.median(nn)),3)
rep['frac_nn_ge_0.7']=round(float(np.mean(np.array(nn)>=0.7)),3); rep['frac_nn_ge_0.9']=round(float(np.mean(np.array(nn)>=0.9)),3)
c.drop(columns=['raw']).to_csv('data/kumar_clean.csv',index=False)
json.dump(rep,open('results/audit_kumar.json','w'),indent=1); print(json.dumps(rep,indent=1))
