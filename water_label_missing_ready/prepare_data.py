"""Rebuild the two benchmarks from verified source bytes. No data committed."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
UCI_URL = 'https://archive.ics.uci.edu/static/public/733/water%2Bquality%2Bprediction-1.zip'
POT_URL = 'https://raw.githubusercontent.com/prasadposture/Water-Potability-Project/main/water_potability.csv'
POT_PAGE = 'https://www.kaggle.com/datasets/adityakadiwal/water-potability'
UCI_SHA = '06fb3c435ab21c5504b4ea656da51910770d809a8f6cbf670d88d512bea97240'
POT_SHA = 'ce044ba3c1802cd4b72681d81e508630b9c6796632ad362e78dea8441efb0b80'
# Git archives may normalize CRLF to LF; accept exactly these two byte snapshots.
POT_LF_SHA = '9f53c322bb39ebe7cde08a370624afe21748c2d97b5da0e5ef9666879f7b756a'
POT_HASHES = (POT_SHA, POT_LF_SHA)
FEATURES_UCI = ['conductance_max','ph_max','ph_min','conductance_min','conductance_mean',
                'dissolved_oxygen_max','dissolved_oxygen_mean','dissolved_oxygen_min',
                'temperature_mean','temperature_min','temperature_max']
FEATURES_POT = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity',
                'Organic_carbon','Trihalomethanes','Turbidity']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def obtain(local, destination, url, expected):
    expected = (expected,) if isinstance(expected, str) else expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    if local is not None:
        source = Path(local)
        if sha256(source) not in expected:
            raise ValueError(f'SHA256 mismatch: {source.name}. Use the original submitted raw file for this protocol.')
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    elif not destination.exists():
        temporary = destination.with_suffix(destination.suffix+'.part')
        try:
            req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=90) as response, temporary.open('wb') as handle:
                shutil.copyfileobj(response, handle)
            if sha256(temporary) not in expected:
                raise ValueError('Downloaded bytes differ from the reviewed source snapshot; do not silently mix versions.')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(destination) not in expected:
        raise ValueError(f'SHA256 mismatch: {destination.name}')
    return destination


def load_uci(archive):
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('water_dataset.mat') and not n.startswith('__MACOSX')]
        if len(names)!=1: raise ValueError('Expected exactly one water_dataset.mat')
        m=loadmat(io.BytesIO(z.read(names[0])))
    n_train=m['X_tr'].size; n_test=m['X_te'].size
    locations=m['location_ids'].ravel()
    frames=[]
    for key, offset in [('tr',0),('te',n_train)]:
        xs=m['X_'+key].ravel();ys=m['Y_'+key]
        if ys.shape != (len(locations),len(xs)):raise ValueError('Unexpected UCI target shape')
        for i, x in enumerate(xs):
            if x.shape != (len(locations),len(FEATURES_UCI)):raise ValueError('Unexpected UCI feature shape')
            day=offset+i
            f=pd.DataFrame(x,columns=FEATURES_UCI)
            f.insert(0,'split','test' if key=='te' else ('train' if i<int(.8*n_train) else 'validation'))
            f.insert(0,'location_id',locations);f.insert(0,'day_index',day)
            f['label_true']=ys[:,i]
            frames.append(f)
    frame=pd.concat(frames,ignore_index=True);frame.insert(0,'row_id',np.arange(len(frame)))
    if not np.isfinite(frame[FEATURES_UCI+['label_true']].to_numpy()).all():
        raise ValueError('Unexpected missing/non-finite UCI source values')
    metadata=dict(name='Water Quality Prediction',source=UCI_URL,uci_dataset_id=733,
                  task='regression',target='source-provided normalized median field pH target',
                  target_alignment='Preserve X_tr/Y_tr and X_te/Y_te pairing supplied by UCI; no extra label shift.',
                  temporal_caveat='Source describes next-day prediction; exact physical dates/alignment and source normalization fitting scope are not independently reconstructed.',
                  official_day_counts=dict(train=n_train,test=n_test,total=n_train+n_test),
                  train_days=int(.8*n_train),validation_days=n_train-int(.8*n_train),
                  locations_in_mat_file=len(locations),tabular_rows_after_flattening_site_day_pairs=len(frame),
                  features=FEATURES_UCI,source_feature_names=[str(v.ravel()[0]) for v in m['features'].ravel()],
                  preprocessing='Retain official processed numbers; chronological split by day, all sites share the split.',
                  raw_archive_sha256=sha256(archive))
    return frame,metadata


def load_potability(csv, seed):
    raw=pd.read_csv(csv)
    if list(raw.columns) != FEATURES_POT+['Potability'] or len(raw)!=3276:
        raise ValueError('Unexpected Water Potability schema/row count')
    y=raw.Potability
    if y.isna().any() or not set(y.unique())<={0,1}:raise ValueError('Invalid source labels')
    ids=np.arange(len(raw))
    train,rest=train_test_split(ids,test_size=.3,stratify=y,random_state=seed)
    val,test=train_test_split(rest,test_size=.5,stratify=y.iloc[rest],random_state=seed)
    split=np.full(len(raw),'test',dtype=object);split[train]='train';split[val]='validation'
    x=raw[FEATURES_POT].astype(float)
    med=x.iloc[train].median();filled=x.fillna(med)
    mean=filled.iloc[train].mean();scale=filled.iloc[train].std(ddof=0).replace(0,1)
    z=(filled-mean)/scale;z.columns=['z_'+name for name in FEATURES_POT]
    if not np.isfinite(z.to_numpy()).all():raise ValueError('Non-finite processed features')
    frame=z.copy();frame.insert(0,'split',split);frame.insert(0,'row_id',ids);frame['label_true']=y
    metadata=dict(name='Water Potability',source=POT_URL,canonical_page=POT_PAGE,
                  source_type='SHA256-pinned third-party CSV mirror of the Kaggle dataset',
                  task='binary classification',target='Potability (0=not potable, 1=potable)',rows=len(raw),
                  class_counts={str(k):int(v) for k,v in y.value_counts().sort_index().items()},
                  original_missing_feature_counts=x.isna().sum().to_dict(),
                  feature_preprocessing='training-median imputation then training-fitted z-score; ddof=0',
                  training_medians=med.to_dict(),scaler_mean=mean.to_dict(),scaler_scale=scale.to_dict(),
                  output_features=list(z.columns),raw_csv_sha256=sha256(csv),
                  canonical_lf_sha256=POT_LF_SHA,
                  line_endings_note='Original URL uses CRLF; Git ZIP copy uses LF; normalized content is identical.',
                  split_method='sklearn stratified 70/15/15; train_test_split test_size=.3 then .5; seed repeated',
                  temporal_caveat='No timestamps/site identifiers: tabular classification only, not a time-series benchmark.')
    return frame,metadata


def write_variants(frame, metadata, folder, seed):
    folder.mkdir(parents=True,exist_ok=True)
    train=np.flatnonzero(frame.split.to_numpy()=='train')
    order=np.random.default_rng(seed).permutation(train)
    metadata.update(schema_version='water_review_v2',mask_seed=seed,
                    split_counts={k:int(v) for k,v in frame.split.value_counts().items()},
                    missingness_mechanism='MCAR without replacement, training labels only; independent RNG per dataset',
                    nested_masks=True,rounding='int(round(rate * n_train))',
                    label_true_policy='Present only in complete.csv; missing variants physically omit it',variants=[])
    frame.to_csv(folder/'complete.csv',index=False)
    for rate in [.2,.4,.6]:
        d=frame.drop(columns='label_true').copy()
        d['label_partial']=frame.label_true.astype(float)
        d['label_observed']=1
        selected=order[:round(rate*len(train))]
        d.loc[selected,'label_partial']=np.nan;d.loc[selected,'label_observed']=0
        name=f'missing_{round(rate*100)}pct.csv';d.to_csv(folder/name,index=False)
        metadata['variants'].append(dict(file=name,requested_missing_rate=rate,
                                         actual_train_missing_rate=len(selected)/len(train),
                                         masked_train_labels=len(selected),train_rows=len(train)))
    write_json(folder/'metadata.json',metadata)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uci-zip',type=Path);p.add_argument('--potability-csv',type=Path)
    p.add_argument('--output-dir',type=Path,default=HERE/'data');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();out=a.output_dir
    uci=obtain(a.uci_zip,out/'raw/water_quality_uci_733.zip',UCI_URL,UCI_SHA)
    pot=obtain(a.potability_csv,out/'raw/water_potability.csv',POT_URL,POT_HASHES)
    datasets={}
    for name,(frame,metadata) in [('water_quality_uci',load_uci(uci)),('water_potability',load_potability(pot,a.seed))]:
        datasets[name]=write_variants(frame,metadata,out/name,a.seed)
        print(name,metadata['split_counts'])
    write_json(out/'manifest.json',dict(schema_version='water_review_v2',random_seed=a.seed,
              missing_rates=[.2,.4,.6],nested_masks=True,datasets=datasets))
    print('Prepared and saved provenance. Run check_data.py to validate.')


if __name__=='__main__':main()
