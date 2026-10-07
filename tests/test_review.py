"""Regression tests for the technical points raised in review (no downloads)."""
import copy
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'meta_maml_exploration'))
from meta_experiment.learner import MetaLearner,_inverse_softplus
from meta_experiment.data import Episode,OmniglotEpisodeSampler
from train import build_optimizer,build_parser,evaluate
from run_all import build_parser as run_parser,build_command
from aggregate_seeds import stats,aggregate


class ReviewTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(1)
        self.episode=Episode(torch.randn(8,3,dtype=torch.float64),torch.tensor([0,1]*4),
                             torch.randn(8,3,dtype=torch.float64),torch.tensor([1,0]*4))

    def learner(self,method='meta_l2',**kw):
        return MetaLearner(nn.Linear(3,2).double(),method,.4,1,kw.pop('initial_l2',.03),False,**kw)

    def test_meta_l2_hypergradient_matches_finite_difference(self):
        m=self.learner();loss,_=m.episode_metrics(self.episode,True);loss.backward()
        raw=next(iter(m.raw_l2.values()));automatic=raw.grad.item();original=raw.item();eps=1e-4
        values=[]
        for offset in [eps,-eps]:
            with torch.no_grad():raw.fill_(original+offset)
            v,_=m.episode_metrics(self.episode,False);values.append(v.item())
        finite=(values[0]-values[1])/(2*eps)
        self.assertGreater(abs(automatic),1e-12)
        self.assertTrue(math.isclose(automatic,finite,rel_tol=1e-5,abs_tol=1e-10),(automatic,finite))

    def test_disjoint_optimizer_groups_and_raw_update(self):
        m=self.learner();args=build_parser().parse_args(['--method','meta_l2','--output-dir','unused'])
        opt=build_optimizer(m,args)
        ids=[id(p) for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in m.parameters()})
        self.assertEqual(opt.param_groups[1]['lr'],.01)
        before=next(iter(m.raw_l2.values())).item();m.episode_metrics(self.episode,True)[0].backward();opt.step()
        self.assertNotEqual(before,next(iter(m.raw_l2.values())).item())

    def test_softplus_initialization_and_zero_rejected(self):
        for value in [1e-12,1e-4,.5,100.]:
            got=torch.nn.functional.softplus(torch.tensor(_inverse_softplus(value),dtype=torch.float64)).item()
            self.assertTrue(math.isclose(value,got,rel_tol=1e-10))
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):_inverse_softplus(value)

    def test_positive_steps_and_scalar_scope(self):
        for method in ['meta_sgd','meta_l2']:
            m=self.learner(method,meta_sgd_positive=True)
            for row in m.learned_hyperparameter_rows():
                self.assertGreater(row['min'],0)
                self.assertEqual(row['scope'],'per_parameter' if method=='meta_sgd' else 'per_layer')
                if method=='meta_l2':self.assertEqual(row['count'],1)
        with self.assertRaises(ValueError):MetaLearner(nn.Linear(3,2),'meta_l2',.4,1,.001,True)

    def test_all_run_arguments_forwarded(self):
        a=run_parser().parse_args(['--inner-lr','.2','--outer-lr','.002','--l2-outer-lr','.03',
                '--initial-l2','.003','--hidden-channels','8','--first-order','--methods','maml',
                '--save-checkpoint','--meta-sgd-positive','--val-every','8','--num-threads','2'])
        command=build_command(a,'maml');child=build_parser().parse_args(command[2:])
        for key,value in vars(a).items():
            if key not in ('output_root','methods','plots'):self.assertEqual(getattr(child,key),value,key)

    def test_validation_restores_mode_and_task_stream(self):
        ep=self.episode
        class Sampler:
            rng=random.Random(18)
            def sample(self):self.rng.random();return ep
        sampler=Sampler();m=self.learner('maml');m.train();rng=sampler.rng.getstate()
        first,_=evaluate(m,sampler,torch.device('cpu'),3,'val')
        second,_=evaluate(m,sampler,torch.device('cpu'),3,'val')
        self.assertEqual(first,second);self.assertEqual(rng,sampler.rng.getstate());self.assertTrue(m.training)

    def test_class_partition_no_leakage(self):
        class Fake:
            _flat_character_images=[(str(i),c) for c in range(20) for i in range(20)]
            def __init__(self,**kw):pass
        common=dict(root='unused',background=True,n_way=3,k_shot=1,q_query=5,download=False,seed=42)
        with patch('meta_experiment.data.datasets.Omniglot',Fake):
            train=OmniglotEpisodeSampler(**common,partition='train')
            val=OmniglotEpisodeSampler(**common,partition='validation')
        self.assertFalse(set(train.classes)&set(val.classes));self.assertEqual(len(train.classes)+len(val.classes),20)

    def test_seed_statistics_and_duplicate_rejection(self):
        result=stats([.5,.6,.7]);self.assertAlmostEqual(result['mean'],.6)
        self.assertAlmostEqual(result['std'],.1)
        self.assertAlmostEqual(result['ci95_half'],4.3026527297*.1/math.sqrt(3))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'seed42';(root/'maml').mkdir(parents=True)
            (root/'maml/summary.json').write_text(json.dumps(dict(method='maml',configuration=dict(seed=42),
                test_accuracy_mean=.6,test_loss_mean=1.,test_accuracy_ci95=.02)))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                aggregate([root,root],Path(tmp)/'aggregate')


if __name__=='__main__':unittest.main(verbosity=2)
