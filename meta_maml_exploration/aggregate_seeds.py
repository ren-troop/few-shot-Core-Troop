"""Seed-level uncertainty. CI overlap is descriptive, not a significance test."""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

COMPARABLE = ('episodes','eval_episodes','meta_batch','n_way','k_shot','q_query','inner_steps',
              'inner_lr','outer_lr','initial_l2','hidden_channels','first_order','val_fraction',
              'split_seed','val_every','val_episodes','meta_sgd_positive','l2_outer_lr','l2_adam_eps')


def stats(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('At least two finite seed results required')
    sd = a.std(ddof=1)
    half = t.ppf(.975, len(a)-1)*sd/math.sqrt(len(a))
    return dict(n_seeds=len(a), mean=float(a.mean()), std=float(sd), ci95_half=float(half),
                ci95_low=float(a.mean()-half), ci95_high=float(a.mean()+half))


def aggregate(roots, output_dir):
    records, seen, reference = [], set(), None
    method_sets = []
    for root in roots:
        summaries = sorted(root.glob('*/summary.json'))
        if not summaries:
            raise ValueError(f'No method summaries under {root}')
        current_methods = set()
        root_seeds = set()
        for file in summaries:
            s = json.loads(file.read_text(encoding='utf-8')); c = s['configuration']
            method, seed = s['method'], int(c['seed'])
            if (method, seed) in seen:
                raise ValueError(f'Duplicate method/seed: {method}/{seed}')
            seen.add((method, seed)); current_methods.add(method); root_seeds.add(seed)
            signature = (s.get('protocol_version', 'legacy_v1_no_validation'), s.get('split_signature'),
                         tuple((k, c.get(k)) for k in COMPARABLE))
            if reference is None:
                reference = signature
            if signature != reference:
                raise ValueError(f'Incomparable configurations/protocols: {file}')
            records.append(dict(seed=seed, method=method, test_accuracy=s['test_accuracy_mean'],
                                test_loss=s['test_loss_mean'], task_ci95=s['test_accuracy_ci95']))
        if len(root_seeds) != 1:
            raise ValueError(f'Mixed seeds in {root}')
        method_sets.append(current_methods)
    if any(x != method_sets[0] for x in method_sets):
        raise ValueError('All seeds must contain the same set of methods')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values(['seed','method'])
    frame.to_csv(output_dir/'per_seed.csv', index=False)
    rows = []
    for method, part in frame.groupby('method'):
        rows.append(dict(method=method, **stats(part.test_accuracy), test_loss_mean=part.test_loss.mean()))
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir/'aggregate.csv', index=False)
    overlap, paired = [], []
    for a, b in itertools.combinations(sorted(frame.method.unique()), 2):
        sa = summary[summary.method==a].iloc[0]; sb = summary[summary.method==b].iloc[0]
        overlap.append(dict(method_a=a, method_b=b,
                            ci95_overlap=bool(max(sa.ci95_low,sb.ci95_low)<=min(sa.ci95_high,sb.ci95_high)),
                            interpretation='descriptive_only_not_a_significance_test'))
        pivot = frame.pivot(index='seed', columns='method', values='test_accuracy')
        delta = stats(pivot[b]-pivot[a])
        paired.append(dict(method_a=a, method_b=b, difference='b_minus_a', **delta,
                           ci_contains_zero=bool(delta['ci95_low'] <= 0 <= delta['ci95_high'])))
    pd.DataFrame(overlap).to_csv(output_dir/'ci_overlap.csv', index=False)
    pd.DataFrame(paired).to_csv(output_dir/'paired_differences.csv', index=False)
    (output_dir/'aggregation.json').write_text(json.dumps(dict(
        protocol=reference[0], seeds=sorted(frame.seed.unique().tolist()),
        definition='Student t interval across independent training-seed means; sd uses ddof=1',
        caveat='Only three seeds: uncertainty is large. CI overlap is not a significance test.',
        paired_definition='Same-seed method differences, two-sided Student t CI; exploratory, no multiplicity correction'
    ), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(summary.to_string(index=False))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-roots', nargs='+', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    aggregate(a.input_roots, a.output_dir)


if __name__ == '__main__':
    main()
