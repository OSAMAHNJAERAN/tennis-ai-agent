from __future__ import annotations
import importlib.util, json, hashlib, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path.cwd().resolve()
RELEASE=ROOT/'artifacts/validation/yolo26_execution_2026-10-15/court_wave64_review_aids_release_v1'
OUT=ROOT/'artifacts/validation/yolo26_execution_2026-10-15/court_wave64_native_review_aids_v1'
DATA=ROOT/'data/external/court_source_wave64_v1'
PLAN=ROOT/'artifacts/validation/yolo26_execution_2026-10-15/court_source_wave_metadata_v2/source_wave_plan.json'
EXPECTED_PLAN='27d72148f4173ab476e9dbaaa5df6b64ce392864fd7150e6d7c0b59e893af65a'
EXPECTED_MANIFEST='8fc0aee1883f10b69dd56e17858bb9d66b5604b65b0a51c0d46f560cf37676a5'
POINT_NAMES=['far_left_doubles','far_right_doubles','near_left_doubles','near_right_doubles','far_left_singles','near_left_singles','far_right_singles','near_right_singles','far_service_left','far_service_right','near_service_left','near_service_right','far_service_center','near_service_center']
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def relative(p): return p.resolve().relative_to(ROOT).as_posix()
def bind(p): return {'path':relative(p),'sha256':sha(p),'bytes':p.stat().st_size}
start=time.perf_counter()
receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PREPARING_NATIVE_REVIEW_AIDS','ram_floor_bytes':1024**3,'disk_floor_bytes':2*1024**3,'qualification_evidence':False,'training_performed':False,'labels_exported':False,'splits_assigned':False,'actual_review_performed_by_rendering':False}
spec=importlib.util.spec_from_file_location('guard',ROOT/'scripts/data/acquire_yolo26_court_source_wave.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
inputs=[]
try:
    if OUT.exists(): raise FileExistsError('Fresh immutable output required')
    if sha(PLAN)!=EXPECTED_PLAN or sha(DATA/'manifest.json')!=EXPECTED_MANIFEST: raise ValueError('Immutable acquisition/plan changed')
    manifest=load(DATA/'manifest.json');plan=load(PLAN)
    if manifest['complete'] is not True or len(manifest['samples'])!=64 or len(manifest['files'])!=64: raise ValueError('Exactly64 complete acquired samples required')
    if manifest['plan_sha256']!=EXPECTED_PLAN: raise ValueError('Wrong plan')
    if {r['id'] for r in manifest['samples']}!={r['id'] for r in manifest['files']}: raise ValueError('Sample/file mismatch')
    allpaths=[PLAN,DATA/'manifest.json',ROOT/'scripts/data/acquire_yolo26_court_source_wave.py',Path(__file__).resolve(),ROOT/'docs/data/YOLO26_ANNOTATION_GUIDE.md']+[ROOT/i['path'] for i in manifest['source_inputs']]+[DATA/i['path'] for i in manifest['files']]
    inputs=[bind(p) for p in dict.fromkeys(allpaths)]
    for i in manifest['source_inputs']:
        if sha(ROOT/i['path'])!=i['sha256']: raise ValueError('Upstream input changed')
    files={i['id']:i for i in manifest['files']}
    for sample in manifest['samples']:
        f=files[sample['id']];p=DATA/f['path']
        if sha(p)!=f['sha256'] or p.stat().st_size!=f['bytes'] or len(sample['publisher_keypoints_xy_px'])!=14: raise ValueError('Image/semantic slots changed')
        if sample['development_split'] is not None or sample['context_group'] is not None: raise ValueError('Pending review must remain unassigned')
    resource=guard.current_resources(OUT);guard.enforce_resources(resource);receipt['resources_before_pil']=resource
    from PIL import Image,ImageDraw,ImageFont
    guard.enforce_resources(guard.current_resources(OUT));receipt['resources_after_pil']=guard.current_resources(OUT)
    OUT.mkdir(parents=True);(OUT/'native_overlays').mkdir();(OUT/'point_detail_boards').mkdir();(OUT/'context_boards').mkdir()
    fontpath='C:/Windows/Fonts/arial.ttf';font=ImageFont.truetype(fontpath,16);small=ImageFont.truetype(fontpath,13);title=ImageFont.truetype(fontpath,19)
    rows=[];boards=[];context=None
    for index,sample in enumerate(manifest['samples']):
        guard.enforce_resources(guard.current_resources(OUT));f=files[sample['id']];path=DATA/f['path']
        with Image.open(path) as decoded:
            decoded.load()
            if decoded.size!=(1280,720): raise ValueError('Decoded native geometry mismatch')
            original=decoded.convert('RGB')
        points=sample['publisher_keypoints_xy_px'];overlay=original.copy();draw=ImageDraw.Draw(overlay)
        for slot,(x,y) in enumerate(points):
            if 0<=x<1280 and 0<=y<720:
                draw.ellipse((x-3,y-3,x+3,y+3),outline='#ff4255',width=2)
                xx=min(1245,max(0,x+5));yy=min(696,max(0,y-20))
                draw.text((xx,yy),str(slot),font=font,fill='yellow',stroke_width=2,stroke_fill='black')
        outpath=OUT/'native_overlays'/f'{index+1:03}_{sample["id"]}.png';overlay.save(outpath,format='PNG')
        detail=Image.new('RGB',(7*144,2*146+40),'#171a20');dd=ImageDraw.Draw(detail)
        dd.text((8,6),f'{index+1:03} {sample["id"]}: PUBLISHER PROPOSALS / NO VISIBILITY ACCEPTED',font=title,fill='white')
        detail_maps=[]
        for slot,(x,y) in enumerate(points):
            cx,cy=round(x),round(y);left,top=cx-48,cy-48;px=(slot%7)*144+24;py=(slot//7)*146+40
            if 0<=x<1280 and 0<=y<720:
                patch=original.crop((left,top,left+96,top+96));detail.paste(patch,(px,py));patch.close()
                detail_maps.append({'slot':slot,'native_crop_xyxy':[left,top,left+96,top+96],'board_patch_xy':[px,py],'scale_xy':[1,1],'padding_if_outside':'BLACK_PIL_CROP_ONLY'})
            else:
                dd.rectangle((px,py,px+95,py+95),outline='red');dd.text((px+3,py+35),'OUTSIDE',font=small,fill='red');detail_maps.append({'slot':slot,'native_crop_xyxy':None,'reason':'PUBLISHER_POINT_OUT_OF_IMAGE_NOT_CLAMPED'})
            dd.text((slot%7*144+6,py+98),f'{slot}: ({x},{y})',font=small,fill='white')
            dd.text((slot%7*144+6,py+115),POINT_NAMES[slot].replace('service','svc'),font=small,fill='#ffc260')
        detailpath=OUT/'point_detail_boards'/f'{index+1:03}_{sample["id"]}.png';detail.save(detailpath,format='PNG');detail.close()
        if index%4==0: context=Image.new('RGB',(1280,2*396),'#171a20')
        xx=(index%2)*640;yy=((index%4)//2)*396;thumb=original.resize((640,360),Image.Resampling.LANCZOS);context.paste(thumb,(xx,yy));thumb.close();cd=ImageDraw.Draw(context);cd.text((xx+7,yy+363),f'{index+1:03} {sample["id"]}',font=title,fill='white')
        row={k:v for k,v in sample.items() if k!='archive_member'};row.update({'source_image':bind(path),'native_review_overlay':bind(outpath),'point_detail_board':bind(detailpath),'point_detail_maps':detail_maps,'native_geometry_decoded_verified':True,'overlay_coordinate_map':'IDENTITY_NATIVE1280x720_NO_RESIZE','actual_visual_review_status':'PENDING','publisher_visibility':'NOT_SUPPLIED','reviewed_keypoints_px':None,'development_split':None,'context_group':None});rows.append(row)
        if index%4==3:
            cp=OUT/'context_boards'/f'context_{index//4+1:02}.png';context.save(cp,format='PNG');context.close();context=None;boards.append({'image':bind(cp),'sample_ids':[r['id'] for r in rows[-4:]],'thumbnail_scale_xy':[.5,.5],'purpose':'CONTEXT_PREVIEW_ONLY_NOT_NATIVE_LANDMARK_ACCEPTANCE'})
        original.close();overlay.close()
    changed=[i['path'] for i in inputs if sha(ROOT/i['path'])!=i['sha256']]
    if changed: raise ValueError('Immutable inputs changed:'+str(changed))
    save(OUT/'review_aids_manifest.json',{'schema_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),'status':'AIDS_READY_PENDING_ACTUAL64_NATIVE_CONTEXT_AND14_INTERSECTION_REVIEW','qualification_evidence':False,'final_test_used':False,'splits_assigned':False,'labels_exported':False,'independent_human_review':False,'source_bindings':inputs,'point_names':POINT_NAMES,'samples':rows,'context_boards':boards,'context_review_constraints':manifest['context_review_constraints'],'rights':manifest['rights'],'review_rules':['Original source image plus native overlay must be actually inspected before intersection visibility acceptance.','Point-detail crops preserve one native pixel per display pixel and contain no target markers.','Do not clamp out-of-image publisher proposals; unknown/occluded slots stay null.','Review all64 source-video contexts and retained prior165 accepted/rejected links before transitive split assignment.','LondonFinals and MonteCarlo remain development-only; distinct video IDs do not establish independent matches.']})
    receipt.update({'status':'COMPLETED64_NATIVE_REVIEW_AIDS_PENDING_ACTUAL_REVIEW','manifest':bind(OUT/'review_aids_manifest.json'),'native_overlays':64,'point_detail_boards':64,'context_boards':16,'input_bindings_unchanged':len(inputs),'changed_inputs':changed,'resources_at_end':guard.current_resources(OUT),'elapsed_seconds':time.perf_counter()-start,'heavy_modules_loaded':[n for n in ['torch','ultralytics','cv2','numpy'] if n in sys.modules]})
except BaseException as exc:
    receipt.update({'status':'FAILED_PRESERVED_PARTIAL_OUTPUT_NO_RETRY','error':type(exc).__name__+': '+str(exc),'resources_at_end':guard.current_resources(OUT),'elapsed_seconds':time.perf_counter()-start});save(RELEASE/'receipt.json',receipt);raise
save(RELEASE/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
