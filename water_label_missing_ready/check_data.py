"""Validate label masking, temporal split and train-only fitted preprocessing."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_data import FEATURES_POT, load_potability, load_uci, sha256, POT_HASHES, UCI_SHA


def check(root):
    results=[]
    if sha256(root/'raw/water_potability.csv') not in POT_HASHES or sha256(root/'raw/water_quality_uci_733.zip')!=UCI_SHA:
        raise AssertionError('Unexpected raw source bytes')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name in ['water_quality_uci','water_potability']:
        folder=root/name;complete=pd.read_csv(folder/'complete.csv');meta=manifest['datasets'][name]
        if name=='water_potability':
            expected,_=load_potability(root/'raw/water_potability.csv',manifest['random_seed'])
            features=meta['output_features']
            x=complete.loc[complete.split=='train',features]
            np.testing.assert_allclose(x.mean(),0,atol=1e-12)
            np.testing.assert_allclose(x.std(ddof=0),1,atol=1e-12)
        else:
            expected,_=load_uci(root/'raw/water_quality_uci_733.zip');features=meta['features']
            assert complete.groupby('day_index').split.nunique().max()==1
            assert complete.loc[complete.split=='train','day_index'].max()<complete.loc[complete.split=='validation','day_index'].min()
            assert complete.loc[complete.split=='validation','day_index'].max()<complete.loc[complete.split=='test','day_index'].min()
        pd.testing.assert_frame_equal(complete,expected,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        previous=set()
        for variant in meta['variants']:
            d=pd.read_csv(folder/variant['file'])
            assert 'label_true' not in d.columns
            pd.testing.assert_frame_equal(d[complete.columns.drop('label_true')],complete.drop(columns='label_true'))
            assert d.row_id.is_unique and d.label_observed.isin([0,1]).all()
            hidden=set(d.index[d.label_observed==0])
            assert previous<=hidden;previous=hidden
            assert len(hidden)==round(variant['requested_missing_rate']*(complete.split=='train').sum())
            assert (d.loc[list(hidden),'split']=='train').all()
            assert d.label_partial.isna().equals(d.label_observed.eq(0))
            np.testing.assert_allclose(d.loc[d.label_observed==1,'label_partial'],complete.loc[d.label_observed==1,'label_true'])
            assert np.isfinite(d[features].to_numpy()).all()
        results.append(dict(dataset=name,rows=len(complete),status='passed',split_counts=meta['split_counts']))
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    p.add_argument('--report',type=Path)
    a=p.parse_args();r=check(a.data_dir)
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
