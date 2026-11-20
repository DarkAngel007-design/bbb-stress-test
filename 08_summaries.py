"""Regenerates the paired-comparison tables and the two summary figures from the per-run result files."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=['random_rows','random_groups','scaffold_rand','cluster_butina']
b=pd.read_csv('results/baseline_results_per_run.csv'); x=pd.read_csv('results/xgb_results_per_run.csv'); g=pd.read_csv('results/gin_results_per_run.csv')
x['key']=x.model+'|'+x.view
def paired(a_df,refs,models,metrics=('auroc','mcc')):
    rows=[]
    for sp in S:
        for m in models:
            a=a_df[(a_df.split==sp)&(a_df.name_==m)].set_index('seed')
            for rn,rb in refs.items():
                bb=rb[rb.split==sp].set_index('seed')
                for met in metrics:
                    d=a[met]-bb[met]; rows.append(dict(split=sp,model=m,vs=rn,metric=met,mean_diff=round(d.mean(),4),se=round(d.std()/np.sqrt(len(d)),4),seeds_better=int((d>0).sum()),n_seeds=len(d)))
    return pd.DataFrame(rows)
refs={'LGBM_desc':b[b.model=='LGBM_desc'],'RF_desc+ecfp':b[b.model=='RF_desc+ecfp']}
paired(x.assign(name_=x.key),refs,['XGB_default|desc','XGB_optuna|desc','XGB_optuna|desc+ecfp']).to_csv('results/xgb_paired_vs_baselines.csv',index=False)
xd=x[(x.view=='desc')&(x.model=='XGB_default')].assign(model='XGB_default_desc')
allr=pd.concat([g,b,xd[b.columns.intersection(xd.columns)]],ignore_index=True)
paired(allr.assign(name_=allr.model),{'LGBM_desc':b[b.model=='LGBM_desc']},['GIN','GIN_desc','RF_desc+ecfp','XGB_default_desc','MLP_desc+maccs','LR_desc']).to_csv('results/gin_paired_vs_lgbm.csv',index=False)
def bars(models,metrics,fn,df,ylims):
    fig,ax=plt.subplots(1,2,figsize=(13,4.8)); w=.8/len(models)
    for a_,(k,t) in zip(ax,metrics):
        for i,m in enumerate(models):
            gg=df[df.model==m].groupby('split')[k]; a_.bar(np.arange(4)+i*w,[gg.mean()[s] for s in S],w,yerr=[gg.std()[s] for s in S],capsize=1.5,label=m)
        a_.set_xticks(np.arange(4)+w*(len(models)-1)/2); a_.set_xticklabels(S,rotation=15); a_.set_title(t+' (mean ± sd over seeds)'); a_.set_ylim(*ylims[k])
    ax[0].legend(fontsize=7,loc='lower left'); fig.tight_layout(); fig.savefig(fn,dpi=130)
bars(['LR_desc','RF_ecfp','RF_desc+ecfp','LGBM_desc','MLP_desc+maccs'],[('acc','Accuracy'),('mcc','MCC')],'results/03_baselines.png',b,{'acc':(.6,1),'mcc':(.5,1)})
bars(['LR_desc','RF_desc+ecfp','LGBM_desc','XGB_default_desc','MLP_desc+maccs','GIN','GIN_desc'],[('auroc','AUROC'),('mcc','MCC')],'results/04_gin_vs_trees.png',allr,{'auroc':(.9,1),'mcc':(.6,1)})
print('wrote paired tables and figures to results/')
