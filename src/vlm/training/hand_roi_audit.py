"""Bounded hand/context crop feasibility audit with privileged pose localization."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import XCLIPModel, XCLIPProcessor
from vlm.experiments.video_sampling import read_video_pyav, uniform_frame_indices
from vlm.models.lora import load_lora_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionLoRAClassifier
from vlm.phase4_alignment import color_joints
from vlm.reporting import write_json
from vlm.skeleton_teacher_cache import sha256_file
from vlm.training.rgb_input_audit import PRODUCT_EXCLUDED_IDS, action_names
from vlm.training.ski_semantic import MODEL, MODEL_REVISION

CLASSES = (2, 3, 13, 19, 33, 37, 39, 41, 45, 47, 49)
TARGETS = (2, 41, 49)
HANDS = (6, 7, 10, 11, 21, 22, 23, 24)


def boxes(path, indices, width, height):
    count, frames = color_joints(path, set(indices))
    if max(indices) >= count:
        raise ValueError('RGB and skeleton frame indices differ')
    votes = Counter()
    for bodies in frames.values():
        for body in bodies:
            votes[body['body_id']] += sum(j['state'] == 2 for j in body['joints'])
    if not votes:
        raise ValueError('No tracked body')
    body_id = sorted(votes, key=lambda b: (-votes[b], b))[0]
    hand, context, scales = [], [], []
    valid_frames = 0
    for idx in indices:
        body = next((b for b in frames[idx] if b['body_id'] == body_id), None)
        if body is None:
            continue
        usable = {j['joint']: np.array([j['x'], j['y']]) for j in body['joints']
                  if j['state'] == 2 and math.isfinite(j['x']) and math.isfinite(j['y'])
                  and 0 <= j['x'] < width and 0 <= j['y'] < height}
        pts = [usable[j] for j in HANDS if j in usable]
        if not pts:
            continue
        valid_frames += 1
        hand.extend(pts)
        context.extend(pts + [usable[j] for j in (2, 3, 20) if j in usable])
        if 0 in usable and 20 in usable:
            scales.append(float(np.linalg.norm(usable[20] - usable[0])))
    if valid_frames < 4 or not scales:
        raise ValueError('Insufficient tracked hand/torso geometry')
    result = square_boxes(hand, context, scales, width, height)
    return result, {'body_id': body_id, 'valid_hand_frames': valid_frames, 'torso_pixels': float(np.median(scales))}


def square_boxes(hand, context, scales, width, height):
    torso = float(np.median(scales))
    result = {}
    for name, pts in [('hands', hand), ('interaction', context)]:
        pts = np.asarray(pts)
        low, high = pts.min(0), pts.max(0)
        center = (low + high) / 2
        # Fixed clip box preserves relative hand movement; margin includes nearby objects.
        side = min(min(width, height), max(float((high-low).max()) + .6*torso, 1.2*torso, 128))
        side = int(math.ceil(side))
        left = max(0, min(width-side, int(round(center[0]-side/2))))
        top = max(0, min(height-side, int(round(center[1]-side/2))))
        result[name] = [left, top, left+side, top+side]
    return result


def run(args):
    root = args.artifacts_dir
    if (root/'experiment_lock.json').exists():
        raise ValueError('Use a fresh artifacts directory')
    torch.set_num_threads(2)
    split_path = Path('artifacts/xclip/selected_split.json')
    paired_path = Path('artifacts/skeleton_data/paired_manifest.json')
    ref_path = Path('artifacts/whole_zoom_validation_v1/view_predictions.pt')
    adapter_path = Path('artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt')
    names_path = Path('docs/xclip-vision-lora-validation-action-accuracy.md')
    split = json.loads(split_path.read_text())
    paired = json.loads(paired_path.read_text())
    canonical = {Path(r['video_path']).stem.removesuffix('_rgb'):r for r in split['val']}
    reference = torch.load(ref_path, map_location='cpu', weights_only=True)
    ref_index = {sid:i for i,sid in enumerate(reference['sample_ids'])}
    chosen = []
    for cls in CLASSES:
        rows = [r for r in paired['val'] if r['class_id'] == f'A{cls:03d}']
        rows.sort(key=lambda r:hashlib.sha256(f'hand-roi42:{r["sample_id"]}'.encode()).hexdigest())
        if len(rows) < args.per_class:
            raise ValueError('Too few paired validation samples')
        chosen.extend(rows[:args.per_class])
    rgb_crops = None
    if args.rgb_crop_manifest is not None:
        rgb_payload = json.loads(args.rgb_crop_manifest.read_text())
        rgb_crops = {row['sample_id']:row for row in rgb_payload['samples']}
        if set(rgb_crops) != {row['sample_id'] for row in chosen}:
            raise ValueError('RGB crop selection differs from diagnostic')
        if rgb_payload['localization'] != 'RGB-only MediaPipe PoseLandmarker lite':
            raise ValueError('Unexpected RGB crop provenance')
    source = torch.load(adapter_path,map_location='cpu',weights_only=False)
    if source['identity']['split_sha256'] != sha256_file(split_path) or source['identity']['model_revision'] != MODEL_REVISION:
        raise ValueError('Checkpoint provenance mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    root.mkdir(parents=True,exist_ok=True)
    lock = {'training':False,'test_evaluated':False,'scope':('RGB-only crop inference feasibility' if rgb_crops is not None else 'Privileged skeleton localization feasibility only, not RGB-only deployment'),
            'classes':CLASSES,'targets':TARGETS,'per_class':args.per_class,'sample_ids':[r['sample_id'] for r in chosen],
            'selection':'Seed42 sample-ID hash within paired canonical validation, independent of predictions',
            'policy':'Same hands and hands+head clip-static square boxes for all labels; full-video uniform8; no per-example oracle',
            'margin':'0.6 torso pixels added to span; minimum side max(1.2 torso,128); image bounds clamped',
            'fallback':'Both local views use original whole frames if tracking invalid; report fallback count',
            'ensembles':'Fixed equal probability means whole+zoom+one local view; no weight search',
            'excluded_source_ids':PRODUCT_EXCLUDED_IDS,
            'source_sha256':{k:sha256_file(p) for k,p in {'runner':Path(__file__),'split':split_path,'paired':paired_path,'reference':ref_path,'adapter':adapter_path,'names':names_path}.items()}}
    if rgb_crops is not None:
        lock['localization'] = rgb_payload['localization']
        lock['source_sha256']['rgb_crop_manifest'] = sha256_file(args.rgb_crop_manifest)
    write_json(root/'experiment_lock.json',lock)
    (root/'runner_snapshot.py').write_text(Path(__file__).read_text())
    processor = XCLIPProcessor.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True)
    model = XCLIPVisionLoRAClassifier(XCLIPModel.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True),num_classes=120,
            head_dropout=.25,feature_noise_std=.01,upper_blocks=2,lora_rank=16,lora_alpha=32,lora_dropout=.05).cuda().requires_grad_(False).eval()
    if model.target_modules != source['model']['target_modules']:
        raise ValueError('Adapter modules mismatch')
    load_lora_state_dict(model,source['model']['lora_state_dict'])
    model.classifier.load_state_dict(source['model']['classifier_state_dict'],strict=True)
    out = {k:[] for k in ('whole','hands','interaction')}; metadata=[];labels=[];selected=[]
    start=time.monotonic()
    with torch.inference_mode():
        for position,row in enumerate(chosen):
            sid=row['sample_id'];cls=int(row['class_id'][1:]);idx=ref_index[sid]
            if sid not in canonical or int(canonical[sid]['class_id']) != cls or int(reference['labels'][idx]) != cls-1:
                raise ValueError('Sample provenance mismatch')
            with av.open(row['rgb_path']) as container:
                count=container.streams.video[0].frames
                indices=uniform_frame_indices(8,count)
                frames=read_video_pyav(container,indices)
            height,width=frames.shape[1:3]
            failure=None
            try:
                if rgb_crops is not None:
                    crop_row = rgb_crops[sid]
                    if crop_row['frame_indices'] != indices.tolist() or crop_row['rgb_sha256'] != sha256_file(Path(row['rgb_path'])):
                        raise RuntimeError('RGB crop source mismatch')
                    if crop_row['fallback'] is not None:
                        raise ValueError(crop_row['fallback'])
                    roi,quality=crop_row['boxes'],crop_row['quality']
                    for left,top,right,bottom in roi.values():
                        if not (0 <= left < right <= width and 0 <= top < bottom <= height and right-left == bottom-top):
                            raise ValueError('Invalid RGB crop bounds')
                else:
                    roi,quality=boxes(Path(row['skeleton_path']),list(map(int,indices)),width,height)
            except ValueError as error:
                roi={k:[0,0,width,height] for k in ('hands','interaction')};quality={};failure=str(error)
            inputs={'whole':frames}
            for name, (left,top,right,bottom) in roi.items():inputs[name]=frames[:,top:bottom,left:right]
            for name,images in inputs.items():
                pixels=processor(images=list(images),return_tensors='pt')['pixel_values'].squeeze(0).unsqueeze(0)
                logits=model(pixels.cuda()).cpu()
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite logits')
                out[name].append(logits)
                # One preselected strip per class, independent of inference outcome.
                if position % args.per_class == 0:
                    values=(pixels[0]*torch.tensor(processor.image_processor.image_std)[None,:,None,None]+torch.tensor(processor.image_processor.image_mean)[None,:,None,None])
                    tiles=(values.permute(0,2,3,1).clamp(0,1).numpy()*255).astype('uint8')
                    canvas=Image.new('RGB',(224*4,244*2))
                    draw=ImageDraw.Draw(canvas)
                    for t,tile in enumerate(tiles):
                        x=(t%4)*224;y0=(t//4)*244;canvas.paste(Image.fromarray(tile),(x,y0));draw.text((x+4,y0+224),f'f{indices[t]}',fill='white')
                    canvas.save(root/f'{sid}_{name}.png')
            delta=float((out['whole'][-1][0]-reference['logits']['whole'][idx]).abs().max())
            if delta>1e-3 or out['whole'][-1].argmax().item()!=reference['logits']['whole'][idx].argmax().item():
                raise RuntimeError(f'Whole parity failed: {sid} {delta}')
            metadata.append({'sample_id':sid,'class_id':cls,'frame_indices':indices.tolist(),'boxes':roi,'quality':quality,'fallback':failure,'whole_parity_max_error':delta,
                             'rgb_sha256':sha256_file(Path(row['rgb_path'])),'skeleton_sha256':None if rgb_crops is not None else sha256_file(Path(row['skeleton_path']))})
            labels.append(cls-1);selected.append(idx)
            if (position+1)%8==0 or position+1==len(chosen):
                print(json.dumps({'state':'inference','samples':position+1,'total':len(chosen),'seconds':round(time.monotonic()-start,1)}),flush=True)
    logits={k:torch.cat(v) for k,v in out.items()}; selected=torch.tensor(selected);labels=torch.tensor(labels)
    logits['center_zoom70']=reference['logits']['center_zoom70'][selected]
    probs={k:v.softmax(-1) for k,v in logits.items()}
    probs['whole_zoom_mean']=(probs['whole']+probs['center_zoom70'])/2
    for view in ('hands','interaction'):
        probs[f'whole_zoom_{view}_mean']=(probs['whole']+probs['center_zoom70']+probs[view])/3
    baseline=probs['whole_zoom_mean'].argmax(-1).eq(labels)
    def metrics(mask):
        result={}
        for view,p in probs.items():
            pred=p.argmax(-1);ok=pred.eq(labels)
            result[view]={'samples':int(mask.sum()),'correct':int(ok[mask].sum()),'accuracy':float(ok[mask].double().mean()),
                          'fixed':int((ok & ~baseline & mask).sum()),'regressed':int((~ok & baseline & mask).sum())}
        return result
    all_mask=torch.ones(len(labels),dtype=torch.bool)
    retained=~torch.isin(labels,torch.tensor(PRODUCT_EXCLUDED_IDS)-1)
    targets=torch.isin(labels,torch.tensor(TARGETS)-1)
    names=action_names(names_path)
    report={'training':False,'test_evaluated':False,'fallbacks':sum(m['fallback'] is not None for m in metadata),
            'seconds':time.monotonic()-start,'all_diagnostic':metrics(all_mask),'retained_diagnostic':metrics(retained),
            'targets':metrics(targets),'confusion_negatives':metrics(~targets),
            'per_class':{f'A{cls:03d}':{'name':names[cls],'metrics':metrics(labels.eq(cls-1))} for cls in CLASSES},
            'limitations':['RGB-only CPU localization tested offline; not yet integrated into streaming runtime' if rgb_crops is not None else 'Privileged pose localization; does not establish RGB-only deployability','Small paired validation diagnostic; no full86 accuracy claim','No optimizer, attention loss or crop finetuning; negative result does not rule out a trained local branch']}
    write_json(root/'crop_manifest.json',{'samples':metadata})
    torch.save({'logits':logits,'probabilities':probs,'labels':labels,'sample_ids':[r['sample_id'] for r in chosen],'training':False,'test_evaluated':False},root/'view_predictions.pt')
    write_json(root/'report.json',report)
    print(json.dumps({'state':'complete','targets':report['targets'],'negatives':report['confusion_negatives']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir',type=Path,required=True)
    parser.add_argument('--rgb-crop-manifest',type=Path)
    parser.add_argument('--per-class',type=int,default=16)
    args=parser.parse_args()
    if args.per_class<1:parser.error('per-class must be positive')
    run(args)
