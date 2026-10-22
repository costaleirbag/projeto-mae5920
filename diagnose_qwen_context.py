"""Read-only CPU diagnostic of transfer between two captured representations."""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import json
ROOT=Path(__file__).resolve().parent
Q=ROOT/'results/qwen_improved'
v=np.load(ROOT/'results/qwen/steering/directions.npz')['rfm_agop']
rows=[]
for context in ['class','writer']:
    feature=np.load(Q/f'features_{context}.npz')
    for split in ['train','probe_validation']:
        x=feature[f'{split}_L24'];y=feature[split+'_labels'];p=x@v
        row={'context':context,'split':split,'n':len(y),'old_direction_auc':float(roc_auc_score(y,p)),
             'old_direction_correlation':float(np.corrcoef(y,p)[0,1]),
             'median_hidden_norm':float(np.median(np.linalg.norm(x,axis=1))),
             'mean_projection_negative':float(p[y==0].mean()),'mean_projection_positive':float(p[y==1].mean())}
        rows.append(row);print(row,flush=True)
(Q/'context_transfer_diagnosis.json').write_text(json.dumps({'rows':rows,
 'caveat':'Descriptive comparison: prompt task, captured token positions and pooling differ. It does not isolate a single causal factor or establish generative control.'},indent=2)+'\n')
