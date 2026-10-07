"""Recompute integrity and diagnostic tables from an exported feedback ZIP."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

METHODS=('maml','meta_sgd','fixed_l2','meta_l2')
SEEDS=(42,43,44)
EXPECTED=dict(episodes=600,eval_episodes=200,meta_batch=4,n_way=5,k_shot=1,q_query=5,
              inner_steps=1,inner_lr=.4,outer_lr=.001,initial_l2=.0001,l2_outer_lr=.01,
              l2_adam_eps=1e-8,hidden_channels=32,val_every=100,val_episodes=50,
              val_fraction=.2,split_seed=2026,first_order=False,meta_sgd_positive=False,
              save_checkpoint=False)


def require(value,message):
    if not value:raise ValueError(message)


def close(a,b,description):
    if not np.allclose(a,b,rtol=1e-6,atol=1e-12,equal_nan=True):
        raise ValueError('Inconsistent '+description)


def review(archive,output):
    rows=[];l2=[];signs=[];validations=[];task_spread=[];all_splits=[];envs=[]
    with zipfile.ZipFile(archive) as z:
        def csv(name):return pd.read_csv(io.BytesIO(z.read(name)))
        def js(name):return json.loads(z.read(name).decode('utf-8'))
        for seed in SEEDS:
            comp=csv(f'local_runs/seed{seed}/comparison.csv').set_index('method')
            require(set(comp.index)==set(METHODS),f'seed{seed} comparison methods')
            for method in METHODS:
                base=f'local_runs/seed{seed}/{method}/'
                summary=js(base+'summary.json');config=summary['configuration']
                require(config['seed']==seed and summary['method']==method,'Wrong seed/method')
                require(summary['protocol_version']=='review_v2_background_class_holdout','Wrong protocol')
                for key,value in EXPECTED.items():require(config[key]==value,f'{seed}/{method}: {key}')
                require(config==js(base+'configuration.json')['configuration'],'Config snapshot mismatch')
                split=js(base+'split.json');all_splits.append(split);envs.append(summary['environment'])
                tr=set(split['train_class_ids']);va=set(split['validation_class_ids'])
                require(not tr&va and len(tr)==771 and len(va)==193 and split['test_class_count']==659,'Class split')
                digest=hashlib.sha256(str(split['train_class_ids']).encode()).hexdigest()
                require(digest==summary['split_signature'],'Split hash')
                log=csv(base+'training_log.csv');val=csv(base+'validation_log.csv');tasks=csv(base+'test_tasks.csv')
                require(log.episode.tolist()==list(range(1,601)),'Missing/duplicated training episodes')
                require(val.episode.tolist()==[100,200,300,400,500,600],'Validation schedule')
                require(tasks.task.tolist()==list(range(200)),'Missing/duplicated test tasks')
                require(np.isfinite(log[['train_query_loss','train_query_accuracy','all_grad_norm_before_clip']]).all().all(),'Non-finite training')
                require(np.isfinite(tasks[['loss','accuracy']]).all().all(),'Non-finite testing')
                require(np.isfinite(val.select_dtypes(include='number')).all().all(),'Non-finite validation')
                require(tasks.accuracy.between(0,1).all() and log.train_query_accuracy.between(0,1).all(),'Invalid accuracy range')
                require((tasks.loss>=0).all() and (log.train_query_loss>=0).all(),'Invalid loss')
                close(tasks.accuracy*25,np.round(tasks.accuracy*25),'25-query accuracy grid')
                sd=tasks.accuracy.std(ddof=1);ci=t.ppf(.975,199)*sd/math.sqrt(200)
                calculated=dict(test_accuracy_mean=tasks.accuracy.mean(),test_accuracy_std=sd,
                                test_accuracy_ci95=ci,test_loss_mean=tasks.loss.mean())
                for key,value in calculated.items():close(value,summary[key],key)
                for key in ['test_accuracy_mean','test_accuracy_ci95','test_loss_mean']:
                    close(calculated[key],comp.loc[method,key],'comparison '+key)
                for name in ['summary.json','configuration.json','split.json']:
                    require(js(base+name)==js(f'compact_results/seed{seed}/{method}/'+name),'Compact JSON mismatch')
                for name in ['learned_hyperparameters.csv','validation_log.csv']:
                    pd.testing.assert_frame_equal(csv(base+name),csv(f'compact_results/seed{seed}/{method}/'+name))
                hp=csv(base+'learned_hyperparameters.csv')
                if len(hp):require(np.isfinite(hp[['mean','std','min','max']]).all().all(),'Non-finite learned hyperparameters')
                for _,v in val.iterrows():validations.append(dict(seed=seed,method=method,episode=int(v.episode),val_accuracy=v.val_accuracy_mean))
                task_spread.append(dict(seed=seed,method=method,test_task_accuracy_std=sd,test_task_loss_mean=tasks.loss.mean()))
                rows.append(dict(seed=seed,method=method,training_updates=len(log),test_tasks=len(tasks),validation_rounds=len(val),
                                 signed_meta_sgd=True,checkpoint_requested=config['save_checkpoint'],device=config['device_used'],
                                 elapsed_seconds=summary['elapsed_seconds'],status='passed'))
                if method=='meta_l2':
                    trajectory=csv(base+'l2_trajectory.csv')
                    require(len(trajectory)==3005 and trajectory.parameter.nunique()==5,'L2 trajectory shape')
                    require(log.raw_l2_grad_norm.gt(0).all(),'Zero/missing L2 gradient norm')
                    gradient_columns=[]
                    for name,group in trajectory.groupby('parameter'):
                        require(group.episode.tolist()==list(range(601)),'L2 episode coverage')
                        initial=group.iloc[0];last=group.iloc[-1];steps=group.iloc[1:]
                        require(np.isfinite(steps.select_dtypes(include='number')).all().all(),'Non-finite L2 trajectory')
                        require(group.coefficient.gt(0).all(),'Nonpositive L2 coefficient')
                        close(initial.coefficient,1e-4,'L2 initial')
                        close(np.logaddexp(0,group.raw_l2),group.coefficient,'softplus transform')
                        close(steps.coefficient,log['l2::'+name],'L2 log trajectory')
                        close(steps.raw_grad_before_clip,log['raw_l2_grad::'+name],'L2 gradient log')
                        close(steps.relative_change,steps.coefficient/1e-4-1,'L2 relative change')
                        close(steps.raw_step,group.raw_l2.diff().iloc[1:],'raw parameter update')
                        close([last.raw_l2,last.coefficient],summary['l2_diagnostics']['final_values'][name],'L2 final summary')
                        close(last.coefficient,hp.loc[hp.parameter==name,'mean'].iloc[0],'L2 final hyperparameters')
                        require(hp.loc[hp.parameter==name,'scope'].iloc[0]=='per_layer','L2 scope')
                        gradient_columns.append('raw_l2_grad::'+name)
                        l2.append(dict(seed=seed,parameter=name,initial_coefficient=initial.coefficient,
                                       final_coefficient=last.coefficient,relative_change_percent=100*last.relative_change,
                                       nonzero_gradient_updates=int(steps.raw_grad_before_clip.ne(0).sum()),
                                       nonzero_raw_updates=int(steps.raw_step.ne(0).sum()),
                                       raw_gradient_abs_median=steps.raw_grad_before_clip.abs().median(),
                                       raw_gradient_abs_max=steps.raw_grad_before_clip.abs().max(),
                                       final_raw_l2=last.raw_l2))
                    close(np.sqrt(log[gradient_columns].pow(2).sum(axis=1)),log.raw_l2_grad_norm,'raw gradient norm')
                    sample=csv(f'compact_results/seed{seed}/{method}/l2_trajectory_sample.csv')
                    expected=trajectory[(trajectory.episode%20==0)|(trajectory.episode==1)|(trajectory.episode==600)].reset_index(drop=True)
                    pd.testing.assert_frame_equal(sample,expected)
                elif method=='meta_sgd':
                    distribution=csv(base+'learning_rate_signs.csv')
                    require((distribution['negative_count']+distribution['zero_count']+distribution['positive_count']).equals(distribution['count']),'Sign counts')
                    close(distribution.negative_fraction,distribution.negative_count/distribution['count'],'Sign proportions')
                    require(hp.scope.eq('per_parameter').all(),'Step scope')
                    close(hp.negative_count.to_numpy(),distribution.negative_count.to_numpy(),'Sign hyperparameter counts')
                    total=int(distribution['count'].sum());negative=int(distribution.negative_count.sum())
                    signs.append(dict(seed=seed,count=total,negative_count=negative,negative_percent=100*negative/total,
                                      minimum_step=hp['min'].min(),maximum_step=hp['max'].max()))
        require(all(x==all_splits[0] for x in all_splits),'Splits differ between runs')
        require(all(x==envs[0] for x in envs),'Environments differ between runs')
        manifest=js('water_metadata/manifest.json')
        require(manifest['schema_version']=='water_review_v2' and manifest['random_seed']==42,'Water metadata protocol')
        for dataset,metadata in manifest['datasets'].items():require(metadata==js('water_metadata/'+dataset+'/metadata.json'),'Water metadata mismatch')
        local_water='water_checks_local.json' in z.namelist()
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'run_integrity.csv',index=False)
    pd.DataFrame(l2).to_csv(output/'l2_layer_diagnostics.csv',index=False)
    pd.DataFrame(signs).to_csv(output/'learning_rate_sign_summary.csv',index=False)
    pd.DataFrame(task_spread).to_csv(output/'test_task_variability.csv',index=False)
    vf=pd.DataFrame(validations);vf.to_csv(output/'validation_by_seed.csv',index=False)
    avg=vf.groupby(['method','episode']).val_accuracy.agg(['mean','std']).reset_index()
    avg.to_csv(output/'validation_across_seeds.csv',index=False)
    last=vf[vf.episode.isin([500,600])].pivot(index=['seed','method'],columns='episode',values='val_accuracy').reset_index()
    last['last_100_delta_pp']=100*(last[600]-last[500]);last.to_csv(output/'validation_final_progress.csv',index=False)
    record=dict(status='passed',archive_name=archive.name,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                formal_runs=len(rows),expected_configuration=EXPECTED,environment=envs[0],
                training_class_count=771,validation_class_count=193,test_class_count=659,
                meta_l2_all_15_layer_seed_gradients_nonzero_in_600_updates=all(x["nonzero_gradient_updates"]==600 for x in l2),
                local_water_check_attached=local_water,
                scope='Numeric consistency and configuration checks on supplied artifacts; not an independent rerun or proof of unchanged local source code.',
                water_scope='Metadata consistency only; the feedback archive does not contain raw/processed CSV data.')
    (output/'integrity_report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--feedback-zip',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();review(a.feedback_zip,a.output_dir)
