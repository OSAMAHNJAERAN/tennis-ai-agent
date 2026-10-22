from pathlib import Path
import hashlib, json
import cv2
import numpy as np
ROOT=Path.cwd()
folder=ROOT/'artifacts/validation/product_execution/person_mask_diagnostic02'
report=json.loads((folder/'report.json').read_text())
protocol=json.loads((folder/'protocol.json').read_text())
def sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()
checks=[]
for frame in report['frames']:
    mask_path=ROOT/frame['masks']; overlay_path=ROOT/frame['overlay']
    assert sha(mask_path)==frame['masks_sha256'] and sha(overlay_path)==frame['overlay_sha256']
    with np.load(mask_path,allow_pickle=False) as pack:
        masks=pack['masks']; boxes=pack['boxes']; confidence=pack['confidence']
        assert masks.dtype==np.uint8 and masks.shape==(len(frame['person_masks']),1080,1920)
        assert set(np.unique(masks).tolist())<={0,1}
        assert boxes.shape==(len(masks),4) and confidence.shape==(len(masks),)
        assert np.isfinite(boxes).all() and np.isfinite(confidence).all()
        assert (boxes[:,0]>=0).all() and (boxes[:,2]<=1920).all()
        assert (boxes[:,1]>=0).all() and (boxes[:,3]<=1080).all()
        assert all(int(mask.sum())==row['mask_area_pixels'] for mask,row in zip(masks,frame['person_masks'],strict=True))
    overlay=cv2.imread(str(overlay_path)); assert overlay.shape==(1080,1920,3)
    checks.append({'frame_index':frame['frame_index'],'mask_count':len(frame['person_masks']),'hashes_and_binary_contract_valid':True})
# Original/predicted crop pairs are for qualitative boundary review only.
cases=[(0,'far body',(390,170,470,330)),(0,'near body',(1280,800,1420,1020)),
       (120,'far body missed',(390,175,510,315)),(240,'near duplicate',(1755,750,1915,980)),
       (270,'partial body fragments',(1850,735,1920,940))]
canvas=np.full((5*260,800,3),245,dtype=np.uint8)
by_frame={item['frame_index']:item for item in protocol['frames']}
for row,(index,title,bounds) in enumerate(cases):
    images=[cv2.imread(str(ROOT/by_frame[index]['path'])),cv2.imread(str(folder/f'overlay_{index:04d}.png'))]
    for column,image in enumerate(images):
        x1,y1,x2,y2=bounds; crop=image[y1:y2,x1:x2]
        gain=min(380/crop.shape[1],220/crop.shape[0]); resized=cv2.resize(crop,(round(crop.shape[1]*gain),round(crop.shape[0]*gain)),interpolation=cv2.INTER_NEAREST)
        x=column*400+(400-resized.shape[1])//2; y=row*260+34
        canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        cv2.putText(canvas,f'{index}: {title} / '+('original' if column==0 else 'mask'),(column*400+8,row*260+23),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1,cv2.LINE_AA)
preview=folder/'boundary_review.png'; assert cv2.imwrite(str(preview),canvas)
record={'complete':True,'qualification_evidence':False,'report_sha256':sha(folder/'report.json'),
        'protocol_sha256':sha(folder/'protocol.json'),'script_sha256':sha(Path(__file__)),
        'total_frames':len(checks),'total_binary_instance_masks':sum(item['mask_count'] for item in checks),
        'npz_loading_allow_pickle':False,'checks':checks,'boundary_review_sha256':sha(preview),
        'boundary_review_scope':'QUALITATIVE_ORIGINAL_AND_PREDICTION_CROP_PAIRS_NOT_PIXEL_GROUND_TRUTH'}
(folder/'artifact_integrity.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({key:value for key,value in record.items() if key!='checks'},indent=2))
