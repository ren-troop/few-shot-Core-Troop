"""Local review figures; all PNGs remain ignored by git."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(root):
    root = Path(root)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summaries = []
    for file in sorted(root.glob('*/summary.json')):
        s = json.loads(file.read_text(encoding='utf-8')); name=s['method']; summaries.append(s)
        v = file.parent/'validation_log.csv'
        if v.exists():
            d=pd.read_csv(v); axes[0].plot(d.episode,d.val_accuracy_mean,marker='o',label=name)
    axes[0].set(title='Held-out background validation tasks',xlabel='Episode',ylabel='Accuracy')
    if axes[0].lines: axes[0].legend()
    axes[1].bar([s['method'] for s in summaries],[s['test_accuracy_mean'] for s in summaries],
                yerr=[s['test_accuracy_ci95'] for s in summaries],capsize=3)
    axes[1].set(title='Final test accuracy: task-level 95% t CI',ylabel='Accuracy',ylim=(0,1))
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    file=root/'meta_l2/l2_trajectory.csv'
    if file.exists():
        d=pd.read_csv(file);fig,ax=plt.subplots(figsize=(8,4))
        for name,g in d.groupby('parameter'):ax.plot(g.episode,g.coefficient,label=name)
        ax.set(yscale='log',xlabel='Episode',ylabel='L2 coefficient',title='Meta-L2 coefficient trajectories')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/'l2_trajectory.png',dpi=160);plt.close(fig)
    file=root/'meta_sgd/learning_rate_signs.csv'
    if file.exists():
        d=pd.read_csv(file);fig,ax=plt.subplots(figsize=(9,4))
        bottom=0
        for kind in ['negative','zero','positive']:
            values=d[kind+'_count']/d['count'];ax.bar(d.parameter,values,bottom=bottom,label=kind);bottom=bottom+values
        ax.set(ylabel='Fraction',title='Meta-SGD inner step signs',ylim=(0,1))
        ax.tick_params(axis='x',labelrotation=70);ax.legend();fig.tight_layout()
        fig.savefig(root/'learning_rate_signs.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    plot_results(p.parse_args().root)
