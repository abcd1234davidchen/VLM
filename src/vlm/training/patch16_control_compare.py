"""Aligned same-epoch Native vs CE/retention comparison; never reads test."""
import csv
import json
from pathlib import Path

import torch

from vlm.patch16_skeleton_contracts import promising, view_probabilities
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file


def align_predictions(payload, ids, labels):
    saved = payload['sample_ids']
    if len(saved) != len(ids) or len(set(saved)) != len(saved) or set(saved) != set(ids) or payload.get('test_evaluated') is not False:
        raise ValueError('Prediction IDs/coverage differ or test provenance missing')
    lookup = {sample:i for i,sample in enumerate(saved)}
    order = torch.tensor([lookup[sample] for sample in ids])
    if not torch.equal(payload['labels'][order], labels):raise ValueError('Prediction labels differ')
    logits = {view:values[order] for view,values in payload['logits'].items()}
    view_probabilities(logits)
    return logits


def paired_counts(native, control, labels):
    if native.shape != control.shape or native.shape != labels.shape:raise ValueError('Aligned prediction shapes differ')
    a,b = native.eq(labels), control.eq(labels)
    masks = {'native_only_correct':a & ~b,'control_only_correct':~a & b,'both_correct':a & b,'both_wrong':~a & ~b}
    counts = {key:int(mask.sum()) for key,mask in masks.items()}
    if sum(counts.values()) != len(labels) or counts['native_only_correct']-counts['control_only_correct'] != int(a.sum()-b.sum()):
        raise RuntimeError('Paired counts do not match accuracy difference')
    return counts,masks


def compare_runs(native_root, control_root, output):
    from vlm.training.patch16_skeleton import reports, resume_selection
    native_root,control_root,output = Path(native_root),Path(control_root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    paired=json.loads(Path('artifacts/skeleton_data/paired_manifest.json').read_text());paired_ids={r['sample_id'] for r in paired['val']}
    baseline_payload=torch.load('artifacts/patch16_resolution/whole_zoom_validation/view_predictions.pt',map_location='cpu',weights_only=True)
    ids,labels=baseline_payload['sample_ids'],baseline_payload['labels']
    if len(ids)!=12960 or len(set(ids))!=12960:raise ValueError('Complete12960 validation required')
    baseline_logits=align_predictions(baseline_payload,ids,labels);baseline=reports(baseline_logits,labels,ids,paired_ids)
    metrics,epochs={'native':[],'control':[]},{}
    for epoch in (1,2):
        logits={arm:align_predictions(torch.load(root/f'epoch{epoch:02d}/view_predictions.pt',map_location='cpu',weights_only=True),ids,labels)
                for arm,root in [('native',native_root),('control',control_root)]}
        current={arm:reports(value,labels,ids,paired_ids,baseline_logits) for arm,value in logits.items()}
        for arm in metrics:metrics[arm].append(current[arm])
        predictions={arm:{view:p.argmax(-1) for view,p in view_probabilities(value).items()} for arm,value in logits.items()}
        views={}
        for view in predictions['native']:
            counts,masks=paired_counts(predictions['native'][view],predictions['control'][view],labels)
            a,b=current['native'][view],current['control'][view]
            incremental=promising(a,b) and promising(a,baseline[view]) if view in ('whole','ensemble') else False
            views[view]={'native_accuracy':a['raw_accuracy'],'control_accuracy':b['raw_accuracy'],
                         'control_minus_baseline_pp':100*(b['raw_accuracy']-baseline[view]['raw_accuracy']),
                         'native_minus_control_pp':100*(a['raw_accuracy']-b['raw_accuracy']),
                         'net_correct':a['correct']-b['correct'],'paired_counts':counts,
                         'incremental_promising':incremental,
                         'guardrails':{'net_correct_at_least26':a['correct']-b['correct']>=26,
                                       'macro_f1_not_lower':a['raw_macro_f1']>=b['raw_macro_f1'],
                                       'unpaired_not_down_over002':a['groups']['unpaired']['accuracy']>=b['groups']['unpaired']['accuracy']-.002-1e-12,
                                       'A060_A120_not_down_over002':a['groups']['A060_A120']['accuracy']>=b['groups']['A060_A120']['accuracy']-.002-1e-12,
                                       'attention_correct_not_lower':a['attention_correct']>=b['attention_correct'],
                                       'native_passes_original_baseline':promising(a,baseline[view])},
                         'case_ids':{key:[ids[i] for i in mask.nonzero().flatten().tolist()] for key,mask in masks.items()},
                         'per_action':{f'A{label+1:03d}':{key:int((mask & labels.eq(label)).sum()) for key,mask in masks.items()} for label in range(120)}}
            with (output/f'epoch{epoch:02d}_{view}_samples.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['sample_id','class_id','native_prediction','control_prediction','case'])
                for i,sample in enumerate(ids):
                    case=next(key for key,mask in masks.items() if mask[i])
                    writer.writerow([sample,int(labels[i])+1,int(predictions['native'][view][i])+1,int(predictions['control'][view][i])+1,case])
        epochs[str(epoch)]={'views':views,'metrics':current}
    primary=epochs['2']['views']
    status='incremental_promising' if any(primary[v]['incremental_promising'] for v in ('whole','ensemble')) else 'no_native_incremental_gain' if all(primary[v]['net_correct']<=0 for v in ('whole','ensemble')) else 'inconclusive'
    selected={arm:{'best':best,'best_promising':qualified} for arm,history in metrics.items() for best,qualified in [resume_selection(history,baseline,'pilot')]}
    recipe=json.loads((control_root/'matched_recipe.json').read_text())
    compute={arm:{str(epoch):json.loads((root/f'epoch{epoch:02d}/compute.json').read_text()) for epoch in (1,2)} for arm,root in [('native',native_root),('control',control_root)]}
    scaler={arm:torch.load(root/'last_checkpoint.pt',map_location='cpu',weights_only=False)['grad_scaler'] for arm,root in [('native',native_root),('control',control_root)]}
    report={'status':status,'primary_epoch':2,'secondary_epochs':[1],'baseline':baseline,'epochs':epochs,
            'validation_selected_secondary':selected,'matched_recipe':recipe,'compute':compute,'final_grad_scaler':scaler,
            'native_explicit_skipped_update_log_available':False,
            'native_no_skip_evidence': {'scale_matches_init256':scaler['native']['scale']==256, 'growth_interval1000000':scaler['native']['growth_interval']==1000000, 'successful_update_growth_tracker':scaler['native']['_growth_tracker']},'control_optimizer_skipped_updates':sum(c.get('optimizer_skipped_updates',0) for c in compute['control'].values()),
            'recipe_source_hashes':{arm:json.loads((root/'experiment_lock.json').read_text())['source_sha256'] for arm,root in [('native',native_root),('control',control_root)]},
            'comparison_source_sha256':sha256_file(Path(__file__)),'test_evaluated':False,'raw_accuracy_gain':None,
            'next_action':'Stop at this review; any extension or seed43 pair requires a separate request.'}
    write_json(output/'run_report.json',report)
    return report
