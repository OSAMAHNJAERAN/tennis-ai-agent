"""One bounded acquisition of two author-released private academic audit videos."""
from pathlib import Path
import ctypes
import datetime
import hashlib
import json
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
DEST=ROOT/'data/external/ball_yolo_author_media_audit_v1'
COMMIT='c6992050c74d6e3f6992bac0e314e7b378a0f51c'
SELECTION=[('America_Best_10_1.mp4',7431027,'28a4b6be70940ff17dea5280968050878d698302'),
           ('France_Review_Red_Court_1.mp4',19860317,'c1e452db15ad75887c57568b8b786e51d4d98734')]
PINS={
 'artifacts/validation/yolo26_execution_2026-10-15/ball_bbox_annotation_inventory_v1/prerelease_root_readme.txt':'69789292b396d6fd622e0c482c45d8253d7c20c90085103609aea86dcc08a745',
 'artifacts/validation/yolo26_execution_2026-10-15/ball_bbox_annotation_inventory_v1/prerelease_original_videos_readme.txt':'61ffb72287b51d63bff9912e0eb0c554121afbfffec3f6aa39aefcc08e3e6fae',
 'artifacts/validation/yolo26_execution_2026-10-15/ball_bbox_annotation_inventory_v1/author_original_video_tree.json':'014564a35bc6bc4ac29b65da3be7a1d596807fd4594a197fd6117505c680dacc',
 'artifacts/validation/yolo26_execution_2026-10-15/ball_bbox_annotation_inventory_v1/complete_annotation_inventory.jsonl':'cf94291ae84aee7f4f2e79b3dfa6ec7da4c050eaa2e1b0edcb0abf8092b341f8'}
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in ('total_phys','available_phys','total_page','available_page','total_virtual','available_virtual','available_extended')]

def resources():
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    result={'available_ram_bytes':value.available_phys,'free_disk_bytes':shutil.disk_usage(BASE).free}
    if result['available_ram_bytes']<1024**3 or result['free_disk_bytes']<2*1024**3:
        raise RuntimeError('Unchanged1GiB RAM/2GiB disk media-acquisition admission refused: '+str(result))
    return result

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    receipt={'status':'RUNNING_ONE_FROZEN_TWO_SOURCE_MEDIA_AUDIT_ACQUISITION',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pinned_commit':COMMIT,
      'selection_reason':'Two modest broadcast-named sources supply1195 and1465 annotated frames for native viewpoint/box/process review. No inference-based selection.',
      'selection_before_media':SELECTION,'total_expected_bytes':sum(v[1] for v in SELECTION),
      'fixed_total_response_ceiling_bytes':28_000_000,'input_bindings':PINS,
      'publisher_grant':'Pinned author README explicitly releases dataset for research and academic purposes only. Apply private noncommercial development restriction; no public redistribution.',
      'original_broadcast_rightsholder_verified':False,'rights_limitation':'Publisher research grant retained; underlying original broadcaster authority remains unresolved. No public examples, weights release or commercial clearance.',
      'annotation_policy_verified':False,'source_training_admitted':False,'source_independence_verified':False,
      'final_test_used':False,'splits_assigned':False,'images_viewed':0,'models_imported':False,'files':[]}
    try:
        if DEST.exists():raise ValueError('Preserve earlier attempts; fresh destination required')
        for path,expected in PINS.items():
            if sha(ROOT/path)!=expected:raise ValueError('Changed source grant/inventory binding: '+path)
        receipt['resources_before_network']=resources()
        tree=json.loads((ROOT/list(PINS)[2]).read_text(encoding='utf-8-sig'))
        entries={row['path']:row for row in tree['tree'] if row['type']=='blob'}
        for name,size,blob in SELECTION:
            if entries[name]['size']!=size or entries[name]['sha']!=blob:raise ValueError('Frozen publisher object differs')
        DEST.mkdir(parents=True,exist_ok=False)
        with (BASE/'attempt_start.json').open('x',encoding='utf-8') as out:json.dump(receipt,out,indent=2)
        for name,size,blob in SELECTION:
            resources()
            url=f'https://raw.githubusercontent.com/Zoutianjian/Ball-Yolo-A-Sport-Ball-Dataset/{COMMIT}/tennis/Videos/{name}'
            digest=hashlib.sha256();git=hashlib.sha1(f'blob {size}\0'.encode());count=0
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TennisAcademicDatasetAudit/1.0'}),timeout=45) as response,(DEST/name).open('xb') as target:
                if response.status!=200:raise ValueError('Expected exact public source response')
                if response.headers.get('Content-Length') and int(response.headers['Content-Length'])!=size:raise ValueError('Unexpected source content length')
                while chunk:=response.read(64*1024):
                    count+=len(chunk)
                    if count>size or sum(v['bytes'] for v in receipt['files'])+count>28_000_000:raise ValueError('Frozen response ceiling exceeded')
                    target.write(chunk);digest.update(chunk);git.update(chunk)
                target.flush()
            if count!=size or git.hexdigest()!=blob:raise ValueError('Exact publisher Git blob/byte count mismatch')
            receipt['files'].append({'path':str((DEST/name).relative_to(ROOT)),'url':url,'bytes':count,
                'sha256':digest.hexdigest(),'publisher_git_blob_sha1':blob,'exact_blob_verified':True,'media_decoded':False})
        if any(sha(ROOT/path)!=expected for path,expected in PINS.items()):raise ValueError('Input changed during acquisition')
        receipt.update(status='COMPLETE_TWO_AUTHOR_ORIGINAL_MEDIA_PENDING_NATIVE_PTS_AND_VISUAL_BOX_REVIEW',
                       total_received_bytes=sum(v['bytes'] for v in receipt['files']),resources_after_network=resources())
        code=0
    except Exception as error:
        receipt.update(status='INCOMPLETE_MEDIA_AUDIT_ACQUISITION_PRESERVE_PARTIAL',error=type(error).__name__,detail=str(error));code=1
    with (BASE/'acquisition_receipt.json').open('x',encoding='utf-8') as out:json.dump(receipt,out,indent=2)
    print(json.dumps(receipt));return code

if __name__=='__main__':raise SystemExit(main())
