"""Recompute measurements, aggregate paired seeds and produce analysis-ready files."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import statistics
import numpy as np

from experiment_io import digest, read_json, write_csv, write_json, write_text
from experiment_metrics import METRICS, calculate, recovery


def collect(root):
    rows=[]
    for path in sorted((root/'runs').glob('*/seed*/p*/metrics.json')):
        rows.extend(read_json(path))
        test=path.with_name('test_metrics.json')
        if test.exists():
            rows.extend(read_json(test))
    return rows


def summarize(rows, metrics):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['p'],row['split'],row['recovery_epoch'])].append(row)
    result=[]
    for (dataset,p,split,t), group in sorted(groups.items()):
        if len({r['seed'] for r in group})!=len(group):
            raise ValueError('Duplicate seeds in summary group.')
        out={'dataset':dataset,'p':p,'split':split,'recovery_epoch':t,'num_seeds':len(group)}
        for metric in metrics:
            values=[r[metric] for r in group if r.get(metric) is not None]
            out[metric+'_mean']=statistics.mean(values) if values else None
            out[metric+'_std']=statistics.stdev(values) if len(values)>1 else None
            out[metric+'_n']=len(values)
        result.append(out)
    return result


def verify(root, rows, protocol, complete):
    seen=set()
    designs={}
    initial_hashes={}
    for r in rows:
        key=(r['dataset'],r['seed'],r['p'],r['split'],r['recovery_epoch'])
        if key in seen:
            raise ValueError(f'Duplicate observation: {key}')
        seen.add(key)
        design_key=(r['dataset'],r['seed'])
        design_path=root/'shared'/r['dataset']/f"seed{r['seed']}"/'design.npz'
        if design_key not in designs:
            with np.load(design_path) as saved:
                designs[design_key]={k:saved[k] for k in ('validation','test_ids','permutation')}
            designs[design_key]['sha256']=digest(design_path)
        design=designs[design_key]
        if r['design_sha256']!=design['sha256']:
            raise ValueError('Shared design checksum mismatch.')
        initial_hashes.setdefault(design_key,r['initial_model_sha256'])
        if initial_hashes[design_key]!=r['initial_model_sha256']:
            raise ValueError('Conditions did not share initial model weights.')
        path=root/r['predictions_file']
        if digest(path)!=r['predictions_sha256']:
            raise ValueError(f'Prediction checksum mismatch: {path}')
        with np.load(path) as data:
            if data['predictions'].shape != (r['num_sources'],10):
                raise ValueError('Prediction/source counts disagree.')
            values=calculate(data['predictions'],data['targets'],data['permutation'])
            if len(np.unique(data['source_ids']))!=len(data['source_ids']):
                raise ValueError('Duplicate source IDs in evaluation.')
            expected_ids=design['test_ids' if r['split']=='test' else 'validation']
            if not (np.array_equal(data['source_ids'],expected_ids) and
                    np.array_equal(data['permutation'],design['permutation']) and
                    np.array_equal(data['color_ids'],np.arange(10))):
                raise ValueError('Evaluation source IDs/color mapping differ from shared design.')
            for y in range(10):
                class_q=float((data['predictions'][data['targets']==y]==y).all(axis=1).mean())
                if not math.isclose(class_q,r[f'class_q_{y}'],abs_tol=1e-12,rel_tol=0):
                    raise ValueError('Per-digit Q differs from saved predictions.')
        if any(not math.isclose(values[k],r[k],abs_tol=1e-12,rel_tol=0) for k in METRICS):
            raise ValueError(f'Saved metrics differ from predictions: {key}')
    times=[0,.25,.5,.75,*range(1,protocol['recovery_epochs']+1)]
    expected={(d,s,p,split,t) for d in protocol['datasets'] for s in protocol['seeds']
              for p in protocol['p_values'] for split,ts in [('validation',times),('test',[0,protocol['recovery_epochs']])]
              for t in ts}
    if not seen <= expected or (complete and seen!=expected):
        raise ValueError(f'Observation grid mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}')
    # Source archives also remain independently verifiable after data cache removal.
    for name in protocol['datasets']:
        manifest=read_json(root/'raw_sources'/name/'manifest.json')
        for entry in manifest['files']:
            if digest(root/entry['file'])!=entry['sha256']:
                raise ValueError('Source archive checksum mismatch.')
    return {'prediction_metrics_recomputed':len(rows),'expected_observations':len(expected),
            'missing_observations':len(expected-seen),'complete':complete,
            'raw_source_checksums_verified':True,'paired_design_verified':True,'tolerance':1e-12}


def build_report(root, protocol, complete=False, check=True):
    root=Path(root)
    rows=collect(root)
    if check:
        write_json(root/'processed'/'verification.json',verify(root,rows,protocol,complete))
    index={(r['dataset'],r['seed'],r['p'],r['split'],r['recovery_epoch']):r for r in rows}
    derived=[]
    for row in rows:
        r=dict(row)
        d,s,p,split,t=(r[k] for k in ('dataset','seed','p','split','recovery_epoch'))
        initial=index.get((d,s,p,split,0))
        reference=index.get((d,s,.1,split,protocol['recovery_epochs']))
        concurrent=index.get((d,s,.1,split,t))
        q=r['all_colors_accuracy']
        r.update(q_initial=initial['all_colors_accuracy'] if initial else None,
                 q_reference=reference['all_colors_accuracy'] if reference else None,
                 reference_run_id=reference['run_id'] if reference else None,
                 q_recovery_percent=None,recovery_status='missing_reference',
                 q_gap_same_epoch_pp=100*(concurrent['all_colors_accuracy']-q) if concurrent else None,
                 q_gap_final_reference_pp=100*(reference['all_colors_accuracy']-q) if reference else None)
        if p==.1:
            r['recovery_status']='neutral_control'
        elif initial and reference:
            r['q_recovery_percent'],r['recovery_status']=recovery(q,r['q_initial'],r['q_reference'])
        derived.append(r)
    derived.sort(key=lambda r:(r['dataset'],r['p'],r['seed'],r['split'],r['recovery_epoch']))
    metrics=[*METRICS,'q_recovery_percent','q_gap_same_epoch_pp','q_gap_final_reference_pp']
    summary=summarize(derived,metrics)
    write_csv(root/'raw'/'observations.csv',rows)
    write_csv(root/'processed'/'derived_results.csv',derived)
    for split in ('validation','test'):
        write_csv(root/'processed'/f'{split}_summary.csv',[r for r in summary if r['split']==split])
    write_csv(root/'processed'/'key_results.csv',[r for r in summary if r['recovery_epoch'] in (0,1,2,protocol['recovery_epochs'])])
    per_digit=[{**{k:r[k] for k in ('dataset','seed','p','split','recovery_epoch')},
                'digit':y,'all_colors_accuracy':r[f'class_q_{y}']} for r in rows for y in range(10)]
    write_csv(root/'processed'/'per_digit_results.csv',per_digit)
    crossings=[]
    for d in protocol['datasets']:
        for seed in protocol['seeds']:
            for p in protocol['p_values']:
                if p==.1: continue
                group=sorted([r for r in derived if (r['dataset'],r['seed'],r['p'],r['split'])==(d,seed,p,'validation')],key=lambda r:r['recovery_epoch'])
                hit=next((i for i,r in enumerate(group) if r['q_recovery_percent'] is not None and r['q_recovery_percent']>=90),None)
                crossings.append({'dataset':d,'seed':seed,'p':p,
                    'previous_observed_epoch':group[hit-1]['recovery_epoch'] if hit is not None and hit>0 else None,
                    'first_observed_epoch_rq90':group[hit]['recovery_epoch'] if hit is not None else None,
                    'actual_epoch_at_first_observation':group[hit]['actual_recovery_epoch'] if hit is not None else None,
                    'dropped_below_later':any(r['q_recovery_percent'] is not None and r['q_recovery_percent']<90 for r in group[hit+1:]) if hit is not None else None,
                    'status':'observed' if hit is not None else 'not_observed_or_undefined'})
    write_csv(root/'processed'/'recovery_times.csv',crossings)
    plots(root,summary,protocol)
    lines=['# 실험 결과 분석 자료','',f"실행 유형: **{protocol['profile']}**. 상태: **{'완료' if complete else '부분 결과'}**.",
           '','이 문서는 측정 결과를 자동 정리한 분석 자료입니다. 심사용 최종 초록은 아닙니다.',
           'smoke/custom 결과를 기본 30경로 정식 실험 결과로 표현하지 마세요.','',
           f"조건: {len(protocol['datasets'])} datasets × {len(protocol['p_values'])} p × {len(protocol['seeds'])} seeds. Exposure 2, recovery {protocol['recovery_epochs']} epochs.",
           '','R_Q는 같은 dataset·seed·평가 split의 중립 모델 최종 Q를 reference로 사용합니다. 비율을 seed별 계산한 후 평균합니다.',
           '표의 ±는 seed 간 표본 표준편차이며, n=1에서는 미정의입니다. 정확도는 %, gap은 %p, R_Q는 %, S는 0~1입니다.','',
           '| Dataset | Split | p | t | Q (%) | R_Q (%) | 같은 시점 Q gap (%p) | n |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(r,k,factor=1):
        m,sd=r.get(k+'_mean'),r.get(k+'_std')
        if m is None:return 'N/A'
        return f'{m*factor:.3f}' + (f' ± {sd*factor:.3f}' if sd is not None else ' (SD 미정의)')
    for r in summary:
        if r['recovery_epoch'] not in (0,1,protocol['recovery_epochs']):continue
        lines.append(f"| {r['dataset']} | {r['split']} | {r['p']:.2f} | {r['recovery_epoch']:g} | {fmt(r,'all_colors_accuracy',100)} | {fmt(r,'q_recovery_percent')} | {fmt(r,'q_gap_same_epoch_pp')} | {r['num_seeds']} |")
    lines += ['', '## 해석 전 확인', '',
              '- Q·R_Q·같은 시점 대조군 격차를 함께 해석하세요. R_Q 100%는 정확도 100%가 아닙니다.',
              '- R_Q 분모가 0.1%p 이하인 경우 미정의이며, 유효 seed 수는 요약 CSV의 *_n 열에 있습니다.',
              '- validation 곡선과 공식 test의 시작·종료 결과를 구분하세요.',
              '- MNIST와 EMNIST의 차이는 데이터 수만의 효과가 아닙니다. 2 epoch의 업데이트 수도 다릅니다.',
              '- seed·색상 변형·epoch를 독립 표본처럼 합산하지 마세요.',
              '- raw_sources의 audit.json에 중복 제외 수와 실제 표본 수가 있습니다.',
              '- processed/verification.json과 status.json을 확인하고 완료되지 않은 조건을 완료로 서술하지 마세요.',
              '', '## 초록 작성에 사용할 파일', '',
              '- processed/key_results.csv: 초기·초기 회복·최종 결과 및 표본 SD.',
              '- processed/derived_results.csv: seed별 지표, reference, 회복률 및 미정의 이유.',
              '- processed/recovery_times.csv: 90% 회복의 첫 관측 구간. 관측 사이 정확한 시점은 알 수 없습니다.',
              '- figures/: PNG와 SVG 회복 곡선. raw/predictions는 개별 run 폴더에 저장됩니다.',
              '- raw_sources/: 원본 이미지·라벨·원본 ID. shared/: 색상 배정·분할·난수 설계.',
              '', '제출 형식: 제목·Abstract·본문·참고문헌, 참고문헌 포함 2페이지 이내. 연구자 식별 정보를 제출 파일에서 제거하세요.']
    write_text(root/'analysis_brief.md','\n'.join(lines)+'\n')
    return rows,derived


def plots(root,summary,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'
    folder.mkdir(exist_ok=True)
    colors={.1:'#666666',.7:'#009E73',.8:'#CC79A7',.9:'#0072B2',.99:'#D55E00'}
    panels=[('all_colors_accuracy',100,'All-colors accuracy Q (%)'),
            ('q_recovery_percent',1,'Recovery R_Q (%)'),
            ('q_gap_same_epoch_pp',1,'Gap to concurrent control (pp)'),
            ('flip_rate',100,'Pairwise flip rate (%)'),
            ('recovery_score',1,'Composite Recovery Score'),
            ('mean_color_accuracy',100,'Mean-color accuracy (%)')]
    for dataset in protocol['datasets']:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,(metric,factor,title) in zip(axes.flat,panels):
            for p in protocol['p_values']:
                rs=sorted([r for r in summary if r['dataset']==dataset and r['split']=='validation' and r['p']==p and r[metric+'_mean'] is not None],key=lambda r:r['recovery_epoch'])
                if not rs:continue
                x=[r['recovery_epoch'] for r in rs]
                y=[r[metric+'_mean']*factor for r in rs]
                sd=[r[metric+'_std'] for r in rs]
                if all(s is not None for s in sd):
                    ax.errorbar(x,y,yerr=[s*factor for s in sd],marker='o',ms=3,capsize=2,label=f'p={p:g}',color=colors.get(p))
                else:ax.plot(x,y,marker='o',ms=3,label=f'p={p:g}',color=colors.get(p))
            ax.set(title=title,xlabel='Recovery epoch (0 = after exposure)')
            ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=8)
        fig.suptitle(f"{dataset} | {protocol['profile']} | validation | mean +/- sample SD",fontsize=14)
        fig.tight_layout(rect=(0,0,1,.95))
        for ext in ('png','svg'):fig.savefig(folder/f'{dataset}_recovery.{ext}',dpi=160)
        for ax in axes.flat:ax.set_xlim(0,1)
        for ext in ('png','svg'):fig.savefig(folder/f'{dataset}_first_epoch.{ext}',dpi=160)
        plt.close(fig)
