"""Reproduce the BBB stress-test pipeline.

Run from inside this folder with Table 1.XLSX placed next to this file or one level up
(or set TABLE1_PATH). Creates data/ and results/ here.
Order: 01 audit (Martins/B3DB) -> 02 audit (Kumar) -> 03 splits -> 04 baselines
-> 05 XGBoost+Optuna -> 06 GIN -> 07 error analysis -> 08 paired tables + summary figures.
Steps 05 and 06 take ~10-15 min each on 18 CPU cores; skip with --skip 05 06.
"""
import os, shutil, subprocess, sys, urllib.request
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
skip = set(sys.argv[sys.argv.index('--skip') + 1:]) if '--skip' in sys.argv else set()
os.makedirs('data', exist_ok=True); os.makedirs('results', exist_ok=True); os.makedirs('src', exist_ok=True)
for f in os.listdir('.'):  # scripts expect to live in src/ and read data/ results/
    if f[:2].isdigit() and f.endswith('.py'): shutil.copy(f, os.path.join('src', f))
table = next((p for p in [os.environ.get('TABLE1_PATH', ''), 'Table 1.XLSX', os.path.join('..', 'Table 1.XLSX')] if p and os.path.exists(p)), None)
if table is None:
    sys.exit("Table 1.XLSX not found. Put the DeePred-BBB Table 1 spreadsheet next to run_all.py (or in the parent folder), or set TABLE1_PATH=/path/to/Table 1.XLSX")
shutil.copy(table, 'data/kumar_table1.xlsx')
for url, out in [('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv', 'data/BBBP.csv'),
                 ('https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_classification.tsv', 'data/B3DB_classification.tsv')]:
    if not os.path.exists(out): urllib.request.urlretrieve(url, out)
steps = ['01_audit', '02_audit_kumar', '03_splits', '04_baselines', '05_xgb_optuna', '06_gin', '07_error_analysis', '08_summaries']
for s in steps:
    if s[:2] in skip: continue
    print('==', s, flush=True); subprocess.run([sys.executable, f'src/{s}.py'] + (['20'] if s.startswith('05') else []), check=True)
print('done; outputs in results/ (plus figures in the working directory for 05/07)')
