"""One-command paired 10-class recovery study.

python experiment.py               # both datasets, five p values, three seeds
python experiment.py --smoke       # small real-data integration run (NOT paper results)
python experiment.py --publish     # same pipeline, then explicitly commit/push artifacts

Repeating the identical command resumes the same output directory automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision
from filelock import FileLock, Timeout

from model import SmallCNN
from experiment_data import load_source, prepare_source, shared_design
from experiment_io import (atomic, digest, git, keep_awake, load_checkpoint, model_hash,
    publish, read_json, restore_rng, rng_state, save_checkpoint, save_npz, stable_seed,
    verify_manifest, write_csv, write_json, write_manifest, write_text)
from experiment_metrics import calculate, self_check
from experiment_report import build_report

HERE=Path(__file__).resolve().parent
CODE_FILES=['experiment.py','experiment_data.py','experiment_io.py','experiment_metrics.py','experiment_report.py','model.py']


class PlannedStop(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def configure(device,threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False
    if hasattr(torch.backends,'cuda'):
        torch.backends.cuda.matmul.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment. Install a CUDA build or use --device cpu.')


def colored_batch(images, ids, palette, colors, device):
    ids_t=torch.as_tensor(ids,dtype=torch.long)
    gray=images[ids_t].to(device=device,dtype=torch.float32)/255
    rgb=palette[ids_t,torch.as_tensor(colors,dtype=torch.long)].to(device)
    return gray[:,None,:,:]*rgb[:,:,None,None]


def evaluate(model,images,labels,ids,palette,permutation,device,batch_size):
    saved_rng=rng_state()
    was_training=model.training
    predictions=np.empty((len(ids),10),dtype=np.uint8)
    source_batch=max(1,batch_size//10)
    started=time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(ids),source_batch):
                ix=ids[start:start+source_batch]
                gray=images[torch.as_tensor(ix)].to(device=device,dtype=torch.float32)/255
                rgb=palette[torch.as_tensor(ix)].to(device)
                x=(gray[:,None,None,:,:]*rgb[:,:,:,None,None]).flatten(0,1)
                # Even eval_batch_size < 10 remains a hard colored-input memory bound.
                out=torch.cat([model(chunk).argmax(1).cpu() for chunk in x.split(batch_size)])
                predictions[start:start+len(ix)]=out.reshape(-1,10).numpy().astype(np.uint8)
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
    elapsed=time.perf_counter()-started
    values=calculate(predictions,labels[ids],permutation)
    for y in range(10):
        mask=labels[ids]==y
        values[f'class_q_{y}']=float((predictions[mask]==y).all(axis=1).mean())
    return predictions,values,elapsed


def evaluation_times(epoch_index,batches):
    if epoch_index==1:return {batches:[0.]}
    if epoch_index<2:return {}
    epoch=epoch_index-1
    if epoch!=1:return {batches:[float(epoch)]}
    result={}
    for fraction in (.25,.5,.75,1.):
        result.setdefault(math.ceil(fraction*batches),[]).append(fraction)
    return result


def initial_model(seed):
    torch.manual_seed(stable_seed(seed,'model'))
    return SmallCNN()


def run_path(root,protocol,name,seed,p,source,design,args):
    run_id=f'{name}_s{seed}_p{round(p*100):03d}'
    folder=root/'runs'/name/f'seed{seed}'/f'p{round(p*100):03d}'
    folder.mkdir(parents=True,exist_ok=True)
    model=initial_model(seed).to(args.device)
    initial_hash=model_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    latest=folder/'latest.json'
    checkpoint_base={'protocol_hash':protocol['protocol_hash'],'design_hash':design['hash'],
                     'initial_model_sha256':initial_hash,'run_id':run_id}
    if latest.exists():
        pointer=read_json(latest)
        path=folder/pointer['file']
        if digest(path)!=pointer['sha256']:raise ValueError('Resume checkpoint checksum mismatch.')
        state=load_checkpoint(path)
        if any(state[k]!=v for k,v in checkpoint_base.items()):raise ValueError('Resume protocol/design/init mismatch.')
        model.load_state_dict(state.pop('model'))
        optimizer.load_state_dict(state.pop('optimizer'))
        restore_rng(state.pop('rng'))
    else:
        state={**checkpoint_base,'epoch_index':0,'next_batch':0,'global_step':0,
               'loss_sum':0.,'correct':0,'samples':0,'observations':[],'train_log':[],
               'elapsed_seconds':0.}
    config={**checkpoint_base,'dataset':name,'seed':seed,'p':p,
            'num_exposure':len(design['exposure']),'num_recovery':len(design['recovery']),
            'num_validation':len(design['validation']),'num_test':len(source['test_ids']),
            'permutation':design['permutation'].tolist()}
    write_json(folder/'config.json',config)
    initial_path=root/'shared'/name/f'seed{seed}'/'initial_model.pt'
    if not initial_path.exists():save_checkpoint(initial_path,initial_model(seed).state_dict())
    # Creating provenance must not reset training RNG when resuming.
    if latest.exists():
        saved=load_checkpoint(folder/read_json(latest)['file'])
        restore_rng(saved['rng'])
    write_json(folder/'metrics.json',state['observations'])
    base_elapsed=state['elapsed_seconds']
    started=time.perf_counter()

    def checkpoint():
        state['elapsed_seconds']=base_elapsed+time.perf_counter()-started
        path=folder/'checkpoints'/f"step_{state['global_step']:07d}.pt"
        save_checkpoint(path,{**state,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state()})
        write_json(latest,{'file':path.relative_to(folder).as_posix(),'sha256':digest(path)})
        write_json(folder/'metrics.json',state['observations'])
        write_csv(folder/'train_log.csv',state['train_log'])
        write_json(folder/'status.json',{'run_id':run_id,'training_complete':state['epoch_index']==2+protocol['recovery_epochs'],
                   'global_step':state['global_step'],'next_epoch_index':state['epoch_index'],
                   'next_batch':state['next_batch'],'updated_at':now()})

    print(f'[{run_id}] start/resume: epoch_index={state["epoch_index"]}, batch={state["next_batch"]}',flush=True)
    while state['epoch_index']<2+protocol['recovery_epochs']:
        eidx=state['epoch_index']
        phase='exposure' if eidx<2 else 'recovery'
        epoch=eidx+1 if eidx<2 else eidx-1
        ids=design[phase]
        order=design[f'{phase}_order_{epoch}']
        colors=design[f'exposure_p{round(p*100):03d}'] if eidx<2 else design[f'recovery_colors_{epoch}']
        batches=math.ceil(len(ids)/64)
        events=evaluation_times(eidx,batches)
        for bi in range(state['next_batch'],batches):
            positions=order[bi*64:(bi+1)*64]
            ix=ids[positions]
            model.train()
            x=colored_batch(source['images'],ix,design['palette'],colors[positions],args.device)
            targets=torch.as_tensor(source['labels'][ix],device=args.device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits,targets)
            if not torch.isfinite(loss):raise ValueError(f'{run_id}: non-finite training loss')
            loss.backward()
            optimizer.step()
            state['global_step']+=1
            state['next_batch']=bi+1
            state['loss_sum']+=float(loss.detach())*len(ix)
            state['correct']+=int((logits.argmax(1)==targets).sum())
            state['samples']+=len(ix)
            if bi+1 in events:
                predictions,values,seconds=evaluate(model,source['images'],source['labels'],design['validation'],
                    design['palette'],design['permutation'],args.device,args.eval_batch_size)
                for t in events[bi+1]:
                    predfile=folder/'raw'/'predictions'/f'validation_t{t:g}.npz'
                    save_npz(predfile,predictions=predictions,targets=source['labels'][design['validation']],
                             source_ids=design['validation'],permutation=design['permutation'],color_ids=np.arange(10))
                    actual=0. if eidx<2 else epoch-1+min((bi+1)*64,len(ids))/len(ids)
                    row={'run_id':run_id,'dataset':name,'seed':seed,'p':p,'split':'validation',
                         'recovery_epoch':t,'actual_recovery_epoch':actual,'global_step':state['global_step'],
                         'recovery_step':0 if eidx<2 else (epoch-1)*batches+bi+1,
                         'recovery_sources_seen':0 if eidx<2 else (epoch-1)*len(ids)+min((bi+1)*64,len(ids)),
                         'num_sources':len(design['validation']),'num_colors':10,
                         'num_exposure':len(design['exposure']),'num_recovery':len(design['recovery']),
                         'initial_model_sha256':initial_hash,'design_sha256':design['hash'],
                         'evaluation_seconds':seconds,'elapsed_seconds':base_elapsed+time.perf_counter()-started,
                         'predictions_file':predfile.relative_to(root).as_posix(),'predictions_sha256':digest(predfile),
                         'checkpoint_file':(folder/'checkpoints'/f"step_{state['global_step']:07d}.pt").relative_to(root).as_posix(),
                         **values}
                    state['observations']=[r for r in state['observations'] if r['recovery_epoch']!=t]+[row]
                    print(f'[{run_id}] t={t:g} Q={values["all_colors_accuracy"]:.4f} F={values["flip_rate"]:.4f} eval={seconds:.1f}s',flush=True)
            end=bi+1==batches
            if end:
                state['train_log'].append({'phase':phase,'epoch':epoch,'loss':state['loss_sum']/state['samples'],
                    'accuracy':state['correct']/state['samples'],'num_sources':state['samples'],'global_step':state['global_step']})
                state.update(epoch_index=eidx+1,next_batch=0,loss_sum=0.,correct=0,samples=0)
                if eidx==1:
                    before=model_hash(model)
                    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                    assert not optimizer.state and model_hash(model)==before
            stopping=args.stop_after_steps is not None and state['global_step']>=args.stop_after_steps
            if end or bi+1 in events or stopping:
                checkpoint()
            if stopping:raise PlannedStop(f'Saved {run_id} at step {state["global_step"]}. Rerun without --stop-after-steps.')
    return folder


def run_test(root,protocol,folder,source,design,args):
    config=read_json(folder/'config.json')
    existing=folder/'test_metrics.json'
    rows=read_json(existing) if existing.exists() else []
    validation=read_json(folder/'metrics.json')
    for t in [0.,float(protocol['recovery_epochs'])]:
        if any(r['recovery_epoch']==t for r in rows):continue
        v=next(r for r in validation if r['recovery_epoch']==t)
        saved=load_checkpoint(root/v['checkpoint_file'])
        model=SmallCNN().to(args.device)
        model.load_state_dict(saved['model'])
        pred,values,seconds=evaluate(model,source['test_images'],source['test_labels'],source['test_ids'],
                     design['test_palette'],design['permutation'],args.device,args.eval_batch_size)
        path=folder/'raw'/'predictions'/f'test_t{t:g}.npz'
        save_npz(path,predictions=pred,targets=source['test_labels'][source['test_ids']],
                 source_ids=source['test_ids'],permutation=design['permutation'],color_ids=np.arange(10))
        row={**v,'split':'test','num_sources':len(source['test_ids']),
             'evaluation_seconds':seconds,'elapsed_seconds':None,'predictions_file':path.relative_to(root).as_posix(),
             'predictions_sha256':digest(path),**values}
        rows.append(row)
        write_json(existing,rows)
        print(f'[{config["run_id"]}] official test t={t:g} Q={values["all_colors_accuracy"]:.4f}',flush=True)


def source_preview(root,name,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,10,figsize=(12,3))
    for y in range(10):
        ids=source['ids'][source['labels'][source['ids']]==y][:2]
        for row,idx in enumerate(ids):
            axes[row,y].imshow(source['images'][int(idx)].numpy(),cmap='gray',vmin=0,vmax=255)
            axes[row,y].set_title(str(y));axes[row,y].axis('off')
    fig.suptitle(f'{name}: canonical orientation (inspect digits before interpreting results)')
    fig.tight_layout()
    path=root/'figures'/f'{name}_source_preview.png';path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=140);plt.close(fig)


def protocol_for(args):
    code={name:digest(HERE/name) for name in CODE_FILES}
    profile='smoke' if args.smoke else 'full'
    if not args.smoke and (args.datasets!=['mnist','emnist_digits'] or args.seeds!=[42,43,44] or
                          args.p_values!=[.1,.7,.8,.9,.99] or args.recovery_epochs!=10):profile='custom'
    p={'version':1,'profile':profile,'datasets':args.datasets,'seeds':args.seeds,'p_values':args.p_values,
       'exposure_epochs':2,'recovery_epochs':args.recovery_epochs,'batch_size':64,
       'eval_batch_size':args.eval_batch_size,'model':'SmallCNN','learning_rate':.001,'optimizer':'Adam',
       'optimizer_reset_after_exposure':True,'split_ratios':[.4,.4,.2],'hue_jitter_degrees':5,
       'smoke_source_limit':1000 if args.smoke else None,'device':args.device,'num_threads':args.num_threads,
       'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,
       'numpy_version':np.__version__,'python_version':platform.python_version(),'platform':platform.system(),
       'cuda_version':torch.version.cuda,'device_name':torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',
       'precision':'float32; TF32 disabled','deterministic_algorithms':True,
       'q_reference':'same dataset/seed/evaluation split neutral control at final recovery epoch',
       'min_initial_q_gap':.001,'flip_rate':'mean disagreement over 45 unordered color pairs',
       'recovery_score':'M*(1-abs(aligned-conflict))*(1-flip_rate)',
       'code_sha256':code}
    p['protocol_hash']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    return p


def setup(root,protocol):
    root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists():
        old=read_json(path)
        if old!=protocol:raise ValueError('Protocol/code/environment differs from saved run. Restore it or use a new --output directory.')
    else:
        write_json(path,protocol)
        for name in CODE_FILES:
            atomic(root/'provenance'/'code'/name,lambda stream,n=name:stream.write((HERE/n).read_bytes()))
        try:
            provenance={'commit':git(HERE,'rev-parse','HEAD'),'worktree_changes':git(HERE,'status','--short','--',*CODE_FILES)}
        except Exception as exc:
            provenance={'git':'unavailable','note':type(exc).__name__}
        write_json(root/'provenance'/'git.json',provenance)


def schedule(args):
    ordered=[]
    for seeds,ps in [(args.seeds[:1],[.1,.99]),(args.seeds[1:],[.1,.99]),
                     (args.seeds,[.9]),(args.seeds,[.7,.8])]:
        for seed in seeds:
            for name in args.datasets:
                for p in ps:
                    if p in args.p_values and (name,seed,p) not in ordered:ordered.append((name,seed,p))
    for name in args.datasets:
        for seed in args.seeds:
            for p in args.p_values:
                if (name,seed,p) not in ordered:ordered.append((name,seed,p))
    return ordered


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke',action='store_true',help='1000-source real-data check, 1 seed, p=.1/.99, recovery2; not formal results')
    parser.add_argument('--datasets',nargs='+',choices=['mnist','emnist_digits'],default=['mnist','emnist_digits'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--p-values',nargs='+',type=float)
    parser.add_argument('--recovery-epochs',type=int)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--num-threads',type=int,default=min(8,os.cpu_count() or 1))
    parser.add_argument('--eval-batch-size',type=int,default=512,help='Maximum colored inputs per forward pass')
    parser.add_argument('--data-root',type=Path,default=HERE/'data')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--offline',action='store_true',help='Use cached source data without downloads')
    parser.add_argument('--analyze-only',action='store_true',help='Verify saved predictions and regenerate summaries/plots without training')
    parser.add_argument('--publish',action='store_true',help='After verification, commit only this run and push current branch to origin')
    parser.add_argument('--no-keep-awake',action='store_true')
    parser.add_argument('--stop-after-steps',type=int,help='Checkpoint and exit after N steps in a path (resume test)')
    args=parser.parse_args(argv)
    args.seeds=sorted(args.seeds if args.seeds is not None else ([42] if args.smoke else [42,43,44]))
    args.p_values=sorted(args.p_values if args.p_values is not None else ([.1,.99] if args.smoke else [.1,.7,.8,.9,.99]))
    if args.recovery_epochs is None:args.recovery_epochs=2 if args.smoke else 10
    if not (args.recovery_epochs>=1 and args.num_threads>=1 and args.eval_batch_size>=1):parser.error('epochs, threads, batch must be positive')
    if len(set(args.seeds))!=len(args.seeds) or any(not 0<=s<2**32 for s in args.seeds):parser.error('seeds must be unique uint32 integers')
    if len(set(args.datasets))!=len(args.datasets):parser.error('duplicate datasets')
    if len(set(args.p_values))!=len(args.p_values) or .1 not in args.p_values or any(not math.isfinite(p) or p<.1 or p>1 or round(p,2)!=p for p in args.p_values):parser.error('p values must be unique 2-decimal probabilities in [.1,1], including neutral .1')
    if args.stop_after_steps is not None and args.stop_after_steps<1:parser.error('stop-after-steps must be positive')
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    args.output=(args.output or HERE/'experiment_results'/('smoke' if args.smoke else 'full')).resolve()
    return args


def main(argv=None):
    args=parse_args(argv)
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(root/'.experiment.lock'),timeout=0),keep_awake(not args.no_keep_awake):
            if args.analyze_only:
                protocol=read_json(root/'protocol.json')
                completed=read_json(root/'status.json').get('status')=='complete'
                if completed:verify_manifest(root)
                build_report(root,protocol,complete=completed)
                if completed:write_manifest(root)
                if args.publish:print('Published:',publish(root))
                return 0
            configure(args.device,args.num_threads)
            protocol=protocol_for(args)
            setup(root,protocol)
            self_check()
            write_json(root/'status.json',{'status':'running','started_or_resumed_at':now(),'profile':protocol['profile']})
            print(f"Profile={protocol['profile']}; device={args.device}; paths={len(schedule(args))}; output={root}",flush=True)
            sources={}
            designs={}
            paths=[]
            try:
                for name in args.datasets:
                    print(f'Preparing {name}: download/cache, source archive, duplicate audit...',flush=True)
                    sources[name]=prepare_source(root,name,load_source(name,args.data_root,not args.offline),1000 if args.smoke else None)
                    source_preview(root,name,sources[name])
                    for seed in args.seeds:
                        designs[name,seed]=shared_design(root,name,sources[name],seed,args.recovery_epochs,args.p_values)
                for name,seed,p in schedule(args):
                    folder=run_path(root,protocol,name,seed,p,sources[name],designs[name,seed],args)
                    paths.append((name,seed,folder))
                    # Save analysis-ready interim measurements; plots follow once at completion.
                    from experiment_report import collect
                    write_csv(root/'raw'/'observations.csv',collect(root))
                    write_json(root/'progress.json',{'completed_training_paths':len(paths),'expected_paths':len(schedule(args)),
                                                   'last_run':f'{name}/{seed}/{p}','updated_at':now()})
                # Test is evaluated only after the entire selected training protocol finishes.
                for name,seed,folder in paths:
                    run_test(root,protocol,folder,sources[name],designs[name,seed],args)
                rows,derived=build_report(root,protocol,complete=True)
                write_manifest(root)
                write_json(root/'status.json',{'status':'complete','finished_at':now(),'profile':protocol['profile'],
                    'training_paths':len(paths),'validation_observations':sum(r['split']=='validation' for r in rows),
                    'test_observations':sum(r['split']=='test' for r in rows),'protocol_hash':protocol['protocol_hash']})
            except BaseException as exc:
                write_json(root/'status.json',{'status':'interrupted' if isinstance(exc,(KeyboardInterrupt,PlannedStop)) else 'failed',
                           'error':str(exc),'updated_at':now(),'profile':protocol['profile']})
                if isinstance(exc,PlannedStop):print(str(exc),flush=True);return 130
                raise
            print(f'Complete. Open {root / "analysis_brief.md"}',flush=True)
            if args.publish:print('Published:',publish(root))
            return 0
    except Timeout:
        print('Another process is already using this output directory.',file=sys.stderr)
        return 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Rerun the same command to resume the last committed checkpoint.',file=sys.stderr)
        raise SystemExit(130)
