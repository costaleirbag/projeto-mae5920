"""Plots from saved auxiliary diagnostics; no model inference."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,NullFormatter
Q=ROOT/'results/qwen_improved';folder=Q/'figures';folder.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def log_ticks(axis,ticks,labels):
    axis.set_xscale('log');axis.set_xticks(ticks,labels)
    axis.xaxis.set_minor_locator(NullLocator());axis.xaxis.set_minor_formatter(NullFormatter())


def main():
    old=pd.DataFrame(json.loads((Q/'original_learning_curve.json').read_text())['rows'])
    transfer=pd.DataFrame(json.loads((Q/'context_transfer_diagnosis.json').read_text())['rows'])
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    ax[0].plot(old.n,old.validation_direction_auc,'o-',color='#007d83')
    log_ticks(ax[0],[32,128,512],['32','128','512'])
    ax[0].set(ylim=(.5,1.01),xlabel='Exemplos de treino (escala log)',ylabel='AUC da direção na validação',title='Representação original: classificação')
    for r in old.itertuples():ax[0].annotate(f'{r.validation_direction_auc:.3f}',(r.n,r.validation_direction_auc),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9)
    rows=transfer.loc[transfer.split=='probe_validation'].set_index('context').loc[['class','writer']]
    ax[1].bar([0,1],rows.old_direction_auc,color=['#0072b2','#d55e00'])
    ax[1].set(xticks=[0,1],xticklabels=['Classificação\núltima posição','Escrita\nmédia das últimas posições'],ylim=(0,1.08),ylabel='AUC da mesma direção antiga',title='Mesmas 256 resenhas de validação')
    for i,v in enumerate(rows.old_direction_auc):ax[1].text(i,v+.02,f'{v:.3f}',ha='center')
    fig.tight_layout();fig.savefig(folder/'representation_diagnosis.png',dpi=180,bbox_inches='tight');plt.close(fig)
    catalog=pd.DataFrame(json.loads((Q/'direction_catalog.json').read_text()))
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    for ax,layer in zip(axes,[16,24]):
        for context,color,label in [('class','#0072b2','Classificação'),('writer','#007d83','Escrita')]:
            rows=catalog.loc[(catalog.context==context)&(catalog.method=='rfm')&(catalog.layer==layer)].sort_values('n')
            ax.plot(rows.n,rows.projection_auc,'o-',color=color,label=label)
        log_ticks(ax,[128,512,2048],['128','512','2.048'])
        ax.set(ylim=(.5,1.01),xlabel='Exemplos de treino (escala log)',title=f'Bloco {layer}')
    axes[0].set_ylabel('AUC da direção na validação');axes[1].legend()
    fig.tight_layout();fig.savefig(folder/'new_auxiliary_learning_curves.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('Auxiliary diagnostic plots created.')

if __name__=='__main__':main()
