"""Safe arrays for the downstream supervised / semi-supervised module."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_split(data_dir, dataset, missing_percent=40, split='train'):
    if dataset not in ('water_quality_uci','water_potability'):raise ValueError('Unknown dataset')
    if missing_percent not in (20,40,60):raise ValueError('Use 20, 40 or 60')
    if split not in ('train','validation','test'):raise ValueError('Unknown split')
    folder=Path(data_dir)/dataset
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    features=metadata.get('output_features',metadata.get('features'))
    d=pd.read_csv(folder/f'missing_{missing_percent}pct.csv')
    d=d.loc[d.split==split].copy()
    # Explicit whitelist excludes identifiers, split, label_true and label_observed from X.
    return dict(X=d[features].to_numpy(dtype=np.float32),
                y=d.label_partial.to_numpy(dtype=np.float32),
                observed=d.label_observed.to_numpy(dtype=bool),
                row_id=d.row_id.to_numpy(),feature_names=features)
