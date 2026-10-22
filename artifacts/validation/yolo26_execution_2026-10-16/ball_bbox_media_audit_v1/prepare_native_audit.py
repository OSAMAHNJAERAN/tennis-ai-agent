"""Freeze32 label checks, then render native frames/patches without a model."""
from pathlib import Path
from fractions import Fraction
import argparse
import datetime
import hashlib
import json
import re
import subprocess

from acquire_two_sources import ROOT,BASE,DEST,resources,sha
LABELS=ROOT/'artifacts/validation/yolo26_execution_2026-10-15/ball_bbox_annotation_inventory_v1/complete_annotation_inventory.jsonl'
EXPECTED_LABELS='cf94291ae84aee7f4f2e79b3dfa6ec7da4c050eaa2e1b0edcb0abf8092b341f8'
PLAN=BASE/'frozen_native_review_plan.json'
OUTPUT=ROOT/'outputs/ball_yolo_author_native_audit32_v1'
ALIASES=('America_Best_10_1','France_Review_Red_Court_1')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def freeze():
    if PLAN.exists():raise ValueError('Frozen plan already exists')
    if sha(LABELS)!=EXPECTED_LABELS:raise ValueError('Changed annotation inventory')
    acquisition=read(BASE/'acquisition_receipt.json')
    if acquisition['status']!='COMPLETE_TWO_AUTHOR_ORIGINAL_MEDIA_PENDING_NATIVE_PTS_AND_VISUAL_BOX_REVIEW':raise ValueError('Incomplete exact acquisition')
    all_labels=[json.loads(line) for line in LABELS.read_text(encoding='utf-8').splitlines()]
    samples=[];bindings={str(LABELS.relative_to(ROOT)):EXPECTED_LABELS,
        str((BASE/'acquisition_receipt.json').relative_to(ROOT)):sha(BASE/'acquisition_receipt.json')}
    timelines={}
    for alias in ALIASES:
        media=DEST/(alias+'.mp4')
        record=next(row for row in acquisition['files'] if Path(row['path']).name==media.name)
        if sha(media)!=record['sha256']:raise ValueError('Changed acquired source')
        probe_path=BASE/'native_probe'/(alias+'.json');probe=read(probe_path)
        stream=probe['streams'][0];frames=probe['frames'];base=Fraction(stream['time_base'])
        pts=[int(row.get('pts',row['best_effort_timestamp'])) for row in frames]
        if not pts or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Missing/nonmonotone native source PTS')
        labels=sorted((r for r in all_labels if r['source_path'].split('/')[0]==alias),key=lambda r:r['frame_index'])
        if len({r['frame_index'] for r in labels})!=len(labels) or any(not 0<=r['frame_index']<len(frames) for r in labels):raise ValueError('Label frame ordinal outside native video')
        # Twelve deterministic temporal strata plus small/large/multi-box cases.
        picked={labels[(i*(len(labels)-1))//11]['frame_index']:labels[(i*(len(labels)-1))//11] for i in range(12)}
        extras=sorted(labels,key=lambda r:(min(b['xywh_normalized'][2]*b['xywh_normalized'][3] for b in r['boxes']),r['frame_index']))
        for row in (extras[0],extras[-1],*[r for r in labels if len(r['boxes'])>1]):
            if len(picked)<16:picked.setdefault(row['frame_index'],row)
        for row in labels:
            if len(picked)<16:picked.setdefault(row['frame_index'],row)
        assert len(picked)==16
        width,height=stream['width'],stream['height']
        for ordinal,row in sorted(picked.items()):
            boxes=[]
            for proposed in row['boxes']:
                x,y,w,h=proposed['xywh_normalized']
                boxes.append({'publisher_xywh_normalized':proposed['xywh_normalized'],
                    'publisher_box_xyxy_native_px':[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height],
                    'center_native_px':[x*width,y*height],'extends_beyond_unit_image':proposed['extends_beyond_unit_image']})
            neighbors=list(range(max(0,ordinal-1),min(len(pts),ordinal+2)))
            samples.append({'id':f'{alias}_f{ordinal:06d}','alias':alias,'frame_index':ordinal,
                'native_width':width,'native_height':height,'source_pts':str(pts[ordinal]),'source_time_base':str(base),
                'source_timestamp_fraction':str(pts[ordinal]*base),'normalized_source_timestamp_fraction':str((pts[ordinal]-pts[0])*base),
                'native_neighbor_indices':neighbors,'native_neighbor_pts':[str(pts[n]) for n in neighbors],
                'label_source_path':row['source_path'],'label_sha256':row['sha256'],'label_git_blob_sha1':row['git_blob_sha'],
                'boxes':boxes,'review_status':'PENDING_ACTUAL_NATIVE_FRAME_AND_BOX_REVIEW',
                'context_group':None,'development_split':None,'match_id':None,'camera_session_id':None})
        bindings[str(media.relative_to(ROOT))]=record['sha256'];bindings[str(probe_path.relative_to(ROOT))]=sha(probe_path)
        timelines[alias]={'native_frame_count':len(pts),'strictly_increasing_pts':True,'native_width':width,'native_height':height,
            'time_base':str(base),'first_pts':str(pts[0]),'last_pts':str(pts[-1]),'r_frame_rate':stream['r_frame_rate'],
            'avg_frame_rate':stream['avg_frame_rate'],'publisher_label_files':len(labels),
            'frame_rate_is_not_assumed_normal_speed':True}
    result={'schema_version':'1.0','status':'FROZEN32_NATIVE_SOURCE_AND_LABEL_AUDIT_NOT_ADMITTED',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),
        'selection_rule':'PER_ALIAS12_TEMPORAL_RANK_STRATA_PLUS_SMALLEST_LARGEST_FIRST_MULTIBOX_THEN_EARLIEST_FILL_TO16',
        'input_bindings':bindings,'samples':samples,'timelines':timelines,'models_used':False,
        'training_admission':False,'final_test_used':False,'qualification_evidence':False,'source_independence_verified':False,
        'missing_annotations_are_negatives':False,'whole_compilation_grouping_required':True}
    with PLAN.open('x',encoding='utf-8') as out:json.dump(result,out,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'plan_sha256':sha(PLAN),'samples':len(samples),'timelines':timelines}))

def render():
    plan=read(PLAN)
    if sha(Path(__file__))!=plan['script_sha256']:raise ValueError('Frozen renderer changed')
    for path,expected in plan['input_bindings'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Changed frozen audit input')
    if OUTPUT.exists():raise ValueError('Fresh renderer output required')
    launch=resources()
    from PIL import Image,ImageDraw
    after_import=resources()
    OUTPUT.mkdir(parents=True)
    native=OUTPUT/'native';native.mkdir()
    overlays=OUTPUT/'overlays';overlays.mkdir();details=OUTPUT/'details';details.mkdir()
    refs={}
    for alias in ALIASES:
        resources()
        chosen=[row for row in plan['samples'] if row['alias']==alias]
        indices=sorted({n for row in chosen for n in row['native_neighbor_indices']})
        folder=native/alias;folder.mkdir()
        expression='+'.join(f'eq(n\\,{n})' for n in indices)
        # The Python argument is an FFmpeg filter string, no shell evaluation.
        expression=expression.replace('\\\\,','\\,')
        command=['ffmpeg','-nostdin','-v','info','-threads','1','-copyts','-i',str(DEST/(alias+'.mp4')),
            '-vf',f'select={expression},showinfo','-fps_mode','vfr','-threads','1',str(folder/'frame_%04d.png')]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        (OUTPUT/(alias+'_ffmpeg.log')).write_text(result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Native source extraction failed: '+result.stderr[-1800:])
        files=sorted(folder.glob('frame_*.png'))
        actual_pts=re.findall(r'Parsed_showinfo[^\n]*\bn:\s*\d+[^\n]*\bpts:\s*(-?\d+)',result.stderr)
        probe=read(BASE/'native_probe'/(alias+'.json'))
        expected_pts=[str(probe['frames'][n].get('pts',probe['frames'][n]['best_effort_timestamp'])) for n in indices]
        if len(files)!=len(indices) or actual_pts!=expected_pts:raise ValueError('Native extraction frame/PTS order differs')
        refs.update({(alias,n):{'path':str(path.relative_to(ROOT)),'sha256':sha(path),'frame_index':n,'source_pts':pts}
                     for n,path,pts in zip(indices,files,actual_pts,strict=True)})
    records=[]
    for row in plan['samples']:
        resources();alias=row['alias'];ordinal=row['frame_index']
        with Image.open(ROOT/refs[(alias,ordinal)]['path']) as source:
            if source.size!=(row['native_width'],row['native_height']):raise ValueError('Native dimensions changed')
            overlay=source.convert('RGB');draw=ImageDraw.Draw(overlay)
            draw.text((12,12),'ORANGE: unaccepted publisher box proposals; no model; private audit',fill='orange',stroke_width=2,stroke_fill='black')
            for index,box in enumerate(row['boxes']):
                draw.rectangle(box['publisher_box_xyxy_native_px'],outline='orange',width=1)
                draw.text(tuple(box['publisher_box_xyxy_native_px'][:2]),str(index),fill='orange')
            overlay_path=overlays/(row['id']+'.png');overlay.save(overlay_path);overlay.close()
        width=480;height=40+len(row['boxes'])*150
        board=Image.new('RGB',(width,height),'#171a1e');draw=ImageDraw.Draw(board)
        draw.text((8,8),row['id']+' | unmarked native96 patches',fill='white')
        for index,box in enumerate(row['boxes']):
            cx,cy=map(round,box['center_native_px'])
            for column,n in enumerate(row['native_neighbor_indices']):
                with Image.open(ROOT/refs[(alias,n)]['path']) as frame:
                    patch=frame.convert('RGB').crop((cx-48,cy-48,cx+48,cy+48))
                    board.paste(patch,(8+column*150,40+index*150));patch.close()
                draw.text((8+column*150,138+index*150),f'box{index} frame{n}',fill='white')
        detail_path=details/(row['id']+'.png');board.save(detail_path);board.close()
        records.append({**row,'native_frames':[refs[(alias,n)] for n in row['native_neighbor_indices']],
            'native_overlay':{'path':str(overlay_path.relative_to(ROOT)),'sha256':sha(overlay_path)},
            'unmarked_native_detail_board':{'path':str(detail_path.relative_to(ROOT)),'sha256':sha(detail_path)}})
    changed=[p for p,h in plan['input_bindings'].items() if sha(ROOT/p)!=h]
    if changed:raise ValueError('Frozen inputs changed during rendering')
    result={'status':'COMPLETE32_NATIVE_LABEL_REVIEW_AIDS_NO_ADMITTED_TRAINING_LABELS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'plan_sha256':sha(PLAN),'input_bindings':plan['input_bindings'],'resources_before_pil':launch,'resources_after_pil':after_import,
        'resources_at_end':resources(),'native_frames':len(refs),'sample_count':len(records),'samples':records,
        'models_used':False,'labels_exported':False,'training_performed':False,'final_test_used':False,'qualification_evidence':False,'input_hashes_changed':changed}
    with (OUTPUT/'review_aids_manifest.json').open('x',encoding='utf-8') as out:json.dump(result,out,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','input_bindings')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render',action='store_true');args=parser.parse_args()
    render() if args.render else freeze()
