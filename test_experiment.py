"""No-network tests of the ten-class pipeline, resume, metrics and paired design."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

import experiment as exp
from experiment_data import assign_colors,validate_colors,palette,split_sources
from experiment_io import load_checkpoint,read_json,model_hash
from experiment_metrics import calculate,recovery,self_check


def fixture(name,root,download):
    rng=np.random.default_rng(411 if name=='mnist' else 412)
    return {split:(rng.integers(0,256,(n,28,28),dtype=np.uint8),np.arange(n,dtype=np.int64)%10)
            for split,n in [('train',1000),('test',100)]}


class MetricsTests(unittest.TestCase):
    def test_known_values(self):
        self_check()
        pi=np.roll(np.arange(10),1)
        p=np.zeros((1,10),dtype=int);p[0,pi[3]]=3
        r=calculate(p,np.array([3]),pi)
        self.assertEqual(r['aligned_accuracy'],1)
        self.assertEqual(r['conflict_accuracy'],0)
        self.assertAlmostEqual(r['mean_color_accuracy'],.1)

    def test_pairwise_identity_and_q(self):
        rng=np.random.default_rng(7)
        pred=rng.integers(0,10,(50,10));y=rng.integers(0,10,50)
        r=calculate(pred,y,np.arange(10))
        direct=np.mean([pred[:,a]!=pred[:,b] for a in range(10) for b in range(a+1,10)])
        self.assertAlmostEqual(r['flip_rate'],direct)
        self.assertEqual(r['all_colors_accuracy'],np.all(pred==y[:,None],axis=1).mean())
        self.assertGreater(recovery(.95,.2,.9)[0],100)
        self.assertLess(recovery(.1,.2,.9)[0],0)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):calculate(np.full((1,10),10),np.array([0]),np.arange(10))
        with self.assertRaises(ValueError):calculate(np.zeros((1,10)),np.array([0]),np.arange(10))


class DesignTests(unittest.TestCase):
    def test_balancing_and_splits(self):
        labels=np.repeat(np.arange(10),113)
        ids=np.arange(len(labels))
        splits=split_sources(ids,labels,42)
        self.assertEqual([len(splits[k]) for k in ('exposure','recovery','validation')],[452,452,226])
        self.assertEqual(len(np.unique(np.concatenate(list(splits.values())))),len(ids))
        pi=np.random.default_rng(2).permutation(10)
        for p in [.1,.7,.8,.9,.99]:
            colors=assign_colors(labels,p,pi,3)
            validate_colors(labels,colors,p,pi)
            np.testing.assert_array_equal(colors,assign_colors(labels,p,pi,3))

    def test_hsv_and_times(self):
        import colorsys
        hue=np.array([-5,0,5],dtype=np.float32)
        pal=palette(hue).numpy()
        for i,h in enumerate(hue):
            for c in range(10):
                np.testing.assert_allclose(pal[i,c],colorsys.hsv_to_rgb(((36*c+h)%360)/360,1,1),atol=1e-6)
        self.assertEqual(exp.evaluation_times(2,375),{94:[.25],188:[.5],282:[.75],375:[1.]})
        self.assertEqual(exp.evaluation_times(1,375),{375:[0.]})
        self.assertEqual(exp.evaluation_times(3,375),{375:[2.]})

    def test_eval_noninterference(self):
        torch.set_num_threads(1)
        model=exp.initial_model(42);model.train()
        before=model_hash(model);rng=torch.get_rng_state().clone()
        x=torch.zeros(10,28,28,dtype=torch.uint8)
        exp.evaluate(model,x,np.arange(10),np.arange(10),palette(np.zeros(10)),np.arange(10),'cpu',3)
        self.assertEqual(before,model_hash(model))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


class PipelineTests(unittest.TestCase):
    @patch('experiment.load_source',side_effect=fixture)
    def test_full_smoke_and_exact_resume(self,_):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            common=['--smoke','--device','cpu','--num-threads','1','--eval-batch-size','100','--no-keep-awake']
            a=root/'continuous';b=root/'resumed'
            self.assertEqual(exp.main(common+['--output',str(a)]),0)
            self.assertEqual(exp.main(common+['--output',str(b),'--stop-after-steps','19']),130)
            self.assertEqual(exp.main(common+['--output',str(b)]),0)
            status=read_json(b/'status.json')
            self.assertEqual(status['validation_observations'],24)
            self.assertEqual(status['test_observations'],8)
            self.assertEqual(status['training_paths'],4)
            for pointer in a.glob('runs/*/seed*/p*/latest.json'):
                rel=pointer.parent.relative_to(a)
                sa=load_checkpoint(pointer.parent/read_json(pointer)['file'])
                other=b/rel
                sb=load_checkpoint(other/read_json(other/'latest.json')['file'])
                self.assertEqual(sa['global_step'],28)
                for k,v in sa['model'].items():self.assertTrue(torch.equal(v,sb['model'][k]),k)
                for ra,rb in zip(sa['observations'],sb['observations']):
                    for key in exp.calculate(np.zeros((1,10),dtype=int),np.array([0]),np.arange(10)):
                        self.assertEqual(ra[key],rb[key])
                self.assertEqual(sa['initial_model_sha256'],sb['initial_model_sha256'])
            for dataset in ('mnist','emnist_digits'):
                configs=[read_json(p) for p in (b/'runs'/dataset).glob('seed*/p*/config.json')]
                self.assertEqual(len({c['initial_model_sha256'] for c in configs}),1)
                self.assertEqual(len({c['design_hash'] for c in configs}),1)
            # Completed reruns must not train again.
            with patch('torch.optim.Adam.step',side_effect=AssertionError('unexpected training')):
                self.assertEqual(exp.main(common+['--output',str(b)]),0)
            self.assertEqual(exp.main(['--analyze-only','--output',str(b)]),0)
            from experiment_io import verify_manifest
            verify_manifest(b)
            # Incompatible resume must fail instead of silently mixing results.
            with self.assertRaises(ValueError):exp.main(common+['--output',str(b),'--seeds','43'])

    @patch('experiment.load_source',side_effect=fixture)
    def test_duplicate_audit(self,_):
        from experiment_data import prepare_source
        src=fixture('mnist',None,False)
        src['train'][0][1]=src['test'][0][0]
        src['train'][0][2]=src['train'][0][3]
        with tempfile.TemporaryDirectory() as td:
            result=prepare_source(Path(td),'mnist',src)
            self.assertNotIn(1,result['ids'])
            self.assertNotIn(3,result['ids'])
            self.assertEqual(result['audit']['train_excluded_exact_duplicates'],2)


class PublicationTests(unittest.TestCase):
    def test_publish_to_local_remote_and_staging_guard(self):
        from experiment_io import git,publish,write_json,write_manifest
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);repo=base/'repo';remote=base/'remote.git'
            repo.mkdir()
            subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
            git(repo,'init','-b','main')
            git(repo,'config','user.name','Pipeline Test')
            git(repo,'config','user.email','test@example.invalid')
            git(repo,'remote','add','origin',str(remote))
            (repo/'.gitignore').write_text('*.pt\n!experiment_results/\n!experiment_results/**\n')
            git(repo,'add','.gitignore');git(repo,'commit','-m','Initial test repository')
            root=repo/'experiment_results'/'full';root.mkdir(parents=True)
            (root/'weights.pt').write_bytes(b'publication fixture, not a model')
            write_json(root/'status.json',{'status':'complete'})
            write_manifest(root)
            (repo/'unrelated.txt').write_text('must remain outside result commit')
            git(repo,'add','unrelated.txt')
            with self.assertRaisesRegex(ValueError,'Unrelated staged'):publish(root)
            git(repo,'reset','--','unrelated.txt')
            sha=publish(root)
            self.assertEqual(git(remote,'rev-parse','refs/heads/main'),sha)
            self.assertNotIn('unrelated.txt',git(repo,'ls-files'))
            self.assertIn('experiment_results/full/weights.pt',git(repo,'ls-files'))
            (root/'weights.pt').write_bytes(b'corruption')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):publish(root)


if __name__=='__main__':unittest.main()
