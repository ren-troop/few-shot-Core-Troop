"""Export compact tracked evidence; optionally package full diagnostic logs for review."""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'meta_maml_exploration'))
from aggregate_seeds import aggregate


def export(roots, target, feedback_zip=None):
    if target.exists() and any(target.iterdir()):
        raise ValueError('Export destination must be new; avoid mixing reruns')
    # Validate comparable settings before writing the evidence package.
    seeds=[]
    for source in roots:
        summaries=list(source.glob('*/summary.json'))
        if not summaries:raise ValueError(f'Incomplete run: {source}')
        current={int(json.loads(f.read_text(encoding='utf-8'))['configuration']['seed']) for f in summaries}
        if len(current)!=1:raise ValueError(f'Mixed seeds under {source}')
        seeds.append(current.pop())
    if len(seeds)!=len(set(seeds)):raise ValueError('Repeated seed')
    aggregate(roots,target/'aggregate')
    sample_written=False
    for source,seed in zip(roots,seeds):
        out=target/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
        if (source/'comparison.csv').exists():shutil.copy2(source/'comparison.csv',out/'comparison.csv')
        for method in sorted(source.iterdir()):
            if not (method/'summary.json').exists():continue
            dst=out/method.name;dst.mkdir()
            for name in ['summary.json','configuration.json','split.json','learned_hyperparameters.csv',
                         'validation_log.csv','learning_rate_signs.csv']:
                if (method/name).exists():shutil.copy2(method/name,dst/name)
            f=method/'l2_trajectory.csv'
            if f.exists():
                d=pd.read_csv(f)
                d=d[(d.episode%20==0)|(d.episode==1)|(d.episode==d.episode.max())]
                d.to_csv(dst/'l2_trajectory_sample.csv',index=False)
            if not sample_written and method.name=='meta_l2':
                d=pd.read_csv(method/'training_log.csv')
                d=d[(d.episode%20==0)|(d.episode==1)|(d.episode==d.episode.max())]
                d.to_csv(dst/'training_log_sample.csv',index=False);sample_written=True
    if feedback_zip:
        feedback_zip.parent.mkdir(parents=True,exist_ok=True)
        allowed={'.csv','.json','.png','.md','.txt'}
        with zipfile.ZipFile(feedback_zip,'w',zipfile.ZIP_DEFLATED) as z:
            for source,seed in zip(roots,seeds):
                for f in sorted(source.rglob('*')):
                    if f.is_file() and f.suffix in allowed:
                        z.write(f,f'local_runs/seed{seed}/'+f.relative_to(source).as_posix())
            for f in sorted(target.rglob('*')):
                if f.is_file():z.write(f,'compact_results/'+f.relative_to(target).as_posix())
            for f in (ROOT/'water_label_missing_ready/data').rglob('*.json'):
                z.write(f,'water_metadata/'+f.relative_to(ROOT/'water_label_missing_ready/data').as_posix())
            f=ROOT/'results/verification/water_checks_local.json'
            if f.exists():z.write(f,'water_checks_local.json')
        print('Send this diagnostic archive back:',feedback_zip)
    print('Commit compact evidence directory:',target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-roots',nargs='+',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--feedback-zip',type=Path)
    a=p.parse_args();export(a.input_roots,a.output_dir,a.feedback_zip)
