"""Mix broadcast positive crops with a disjoint COCO train2017 rehearsal subset."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import yaml
from scripts.data.acquire_coco_tennis_validation import digest


def main():
    broadcast=ROOT/'data/external/racketvision_racket_crop_pilot01'
    coco=ROOT/'data/external/coco_racket_rehearsal160'
    output=ROOT/'data/external/racketvision_racket_rehearsal_pilot04'
    if output.exists():
        raise ValueError('Require fresh mixed dataset output')
    old=json.loads((broadcast/'manifest.json').read_text())
    new=json.loads((coco/'manifest.json').read_text())
    if not old['complete'] or not new['complete'] or digest(coco/'instances.json')!=new['instances_sha256']:
        raise ValueError('Incomplete source or changed annotations')
    annotations=json.loads((coco/'instances.json').read_text())['annotations']
    records=[]
    for split in ('train','val'):
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    for item in old['records']:
        for field in ('image','label'):
            if digest(broadcast/item[field])!=item[field+'_sha256']:
                raise ValueError('Broadcast crop changed')
            shutil.copyfile(broadcast/item[field],output/item[field])
        records.append({**item,'domain':'broadcast_crop'})
    for item in new['images']:
        if digest(coco/item['path'])!=item['sha256']:
            raise ValueError('COCO image changed')
        split=item['split'];stem=f"coco_train2017_{item['id']:012d}"
        image=output/'images'/split/(stem+'.jpg');label=output/'labels'/split/(stem+'.txt')
        shutil.copyfile(coco/item['path'],image)
        lines=[]
        for ann in annotations:
            if ann['image_id']!=item['id']:
                continue
            if ann['category_id']!=43 or ann.get('iscrowd',0):
                raise ValueError('Unexpected class or crowd annotation')
            x,y,w,h=ann['bbox'];width,height=item['width'],item['height']
            x1,y1,x2,y2=max(0,x),max(0,y),min(width,x+w),min(height,y+h)
            if x2<=x1 or y2<=y1:
                raise ValueError('Invalid racket annotation')
            lines.append(f'38 {(x1+x2)/(2*width):.9f} {(y1+y2)/(2*height):.9f} {(x2-x1)/width:.9f} {(y2-y1)/height:.9f}')
        if not lines:
            raise ValueError('Missing positive racket labels')
        label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        records.append({'split':split,'source_clip':stem,'domain':'coco_train2017','publisher_image_id':item['id'],
                        'rackets':len(lines),'image':str(image.relative_to(output)),'image_sha256':digest(image),
                        'label':str(label.relative_to(output)),'label_sha256':digest(label)})
    groups={split:old['split_clips'][split]+[f'coco_train2017_{i:012d}' for i in new['split_ids'][split]] for split in ('train','val')}
    if set(groups['train'])&set(groups['val']):
        raise ValueError('Mixed train-selection overlap')
    config=yaml.safe_load((broadcast/'data.yaml').read_text());config['path']=output.as_posix()
    (output/'data.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    report={'complete':True,'qualification_evidence':False,'split_clips':groups,
            'selection_scope':'BEST_INTERNAL_FITNESS_ON_147_BROADCAST_CROPS_AND_32_COCO_TRAIN_IMAGES; NO_EXTERNAL_SELECTION',
            'limitations':['Rehearsal uses original detector training distribution','Other classes unlabeled; racket-only candidate',
                           'No verified absent-racket images','Image and publisher match-ID splits do not prove source-session independence'],
            'sources':{'broadcast_manifest_sha256':digest(broadcast/'manifest.json'),'coco_manifest_sha256':digest(coco/'manifest.json')},
            'script_sha256':digest(Path(__file__)),'records':records,
            'counts':{split:{'images':sum(r['split']==split for r in records),'rackets':sum(r['rackets'] for r in records if r['split']==split)} for split in groups}}
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['counts']),flush=True)


if __name__=='__main__':
    main()
