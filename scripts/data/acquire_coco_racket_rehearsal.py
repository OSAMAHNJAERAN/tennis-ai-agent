"""Bounded COCO train2017 racket appearance rehearsal, disjoint from val2017."""
import json
from pathlib import Path
import random
import sys
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'artifacts/tools/vision_eval_runtime'))
import cv2
import ijson
from scripts.data.acquire_coco_tennis_validation import digest, download


def main():
    validation = ROOT/'data/external/coco_tennis_val2017'
    output = ROOT/'data/external/coco_racket_rehearsal160'
    archive = validation/'annotations_trainval2017.zip'
    old = json.loads((validation/'manifest.json').read_text())
    if not old['complete'] or digest(archive) != old['source_archive_sha256']:
        raise ValueError('Existing COCO source archive changed')
    manifest_path = output/'manifest.json'
    if not manifest_path.exists():
        if output.exists():
            raise ValueError('Output exists without a resumable manifest')
        output.mkdir(parents=True)
        with zipfile.ZipFile(archive) as bundle:
            info = bundle.getinfo('annotations/instances_train2017.json')
            if info.file_size != 469785474 or info.CRC != 959631175:
                raise ValueError('Unexpected annotation member metadata')
            with bundle.open(info) as stream:
                rackets = [row for row in ijson.items(stream,'annotations.item',use_float=True) if row['category_id']==43]
            excluded = {r['image_id'] for r in rackets if r.get('iscrowd',0)}
            ids = sorted({r['image_id'] for r in rackets}-excluded)
            random.Random(17).shuffle(ids)
            selected = set(ids[:160])
            if len(selected)!=160 or selected & set(old['selected_image_ids']):
                raise ValueError('Selection incomplete or overlaps validation images')
            split_ids = {'train':sorted(ids[:128]),'val':sorted(ids[128:160])}
            with bundle.open(info) as stream:
                images = [r for r in ijson.items(stream,'images.item',use_float=True) if r['id'] in selected]
            with bundle.open(info) as stream:
                licenses = list(ijson.items(stream,'licenses.item',use_float=True))
            with bundle.open(info) as stream:
                categories = list(ijson.items(stream,'categories.item',use_float=True))
        if len(images)!=160 or {r['id'] for r in images}!=selected:
            raise ValueError('Selected image metadata incomplete')
        annotations = {'images':sorted(images,key=lambda r:r['id']),
                       'annotations':[r for r in rackets if r['image_id'] in selected],
                       'categories':categories,'licenses':licenses}
        (output/'instances.json').write_text(json.dumps(annotations,separators=(',',':')),encoding='utf-8')
        manifest = {'complete':False,'qualification_evidence':False,'seed':17,
                    'source':'COCO_TRAIN2017','selection':'128 TRAIN / 32 INTERNAL SELECTION; RACKET POSITIVE NONCROWD IMAGES; CHOSEN BEFORE INFERENCE',
                    'pretraining_scope':'ORIGINAL_DETECTOR_TRAINING_DATA; REHEARSAL AND INTERNAL SELECTION, NOT INDEPENDENT TEST',
                    'limitations':['No verified racket-absent scenes','Image-ID separation does not establish original-photo-session separation'],
                    'archive_sha256':digest(archive),'archive_member_crc32':959631175,
                    'instances_sha256':digest(output/'instances.json'),'script_sha256':digest(Path(__file__)),
                    'split_ids':split_ids,'validation_image_id_overlap':[], 'licenses':licenses,'images':[]}
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest = json.loads(manifest_path.read_text())
    if manifest['archive_sha256']!=digest(archive) or manifest['instances_sha256']!=digest(output/'instances.json'):
        raise ValueError('Resume source checksum mismatch')
    annotations=json.loads((output/'instances.json').read_text())
    done={r['id']:r for r in manifest['images']}
    for row in done.values():
        if digest(output/row['path'])!=row['sha256']:
            raise ValueError('Existing image checksum mismatch')
    total=sum(r['bytes'] for r in done.values())
    (output/'images').mkdir(exist_ok=True)
    for item in annotations['images']:
        if item['id'] in done:
            continue
        url=item['coco_url'];parsed=urllib.parse.urlsplit(url);name=item['file_name']
        if Path(name).name!=name or parsed.hostname!='images.cocodataset.org' or parsed.path!='/train2017/'+name or parsed.scheme not in ('http','https'):
            raise ValueError('Unexpected publisher image URL')
        path=output/'images'/name
        size=download(url,path,min(10*1024**2,200*1024**2-total))
        total+=size
        image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(item['height'],item['width']):
            raise ValueError('Downloaded image geometry mismatch')
        manifest['images'].append({'id':item['id'],'path':'images/'+name,'sha256':digest(path),'bytes':size,
                                   'width':item['width'],'height':item['height'],'license':item.get('license'),
                                   'source_url':url,'split':'train' if item['id'] in manifest['split_ids']['train'] else 'val'})
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        if len(manifest['images'])%20==0:
            print(json.dumps({'downloaded_images':len(manifest['images']),'bytes':total}),flush=True)
    manifest.update(complete=True,image_bytes=total)
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'images':len(manifest['images']),'rackets':len(annotations['annotations']),'bytes':total}),flush=True)


if __name__=='__main__':
    main()
