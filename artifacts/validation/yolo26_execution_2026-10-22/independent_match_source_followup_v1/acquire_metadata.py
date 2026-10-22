"""Bounded public text metadata only; never execute publisher code or fetch media."""
from pathlib import Path
from datetime import datetime,timezone
import urllib.request,urllib.error,json,hashlib
ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).resolve().parent
LIMIT=2*1024*1024
REVISION=json.loads((ART/'playable_head.json').read_text(encoding='utf-8'))['sha']
REPO='willi-menapace/PlayableEnvironments'
records=[]
def fetch(name,url,expected_blob=None):
 record={'url':url,'observed_utc':datetime.now(timezone.utc).isoformat(),'limit_bytes':LIMIT}
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'TennisAcademicMetadataAudit/1.0','Accept':'application/vnd.github+json' if 'api.github.com' in url else 'text/plain'})
  with urllib.request.urlopen(request,timeout=30) as response:
   body=response.read(LIMIT+1)
   record.update(status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'),bytes=len(body))
  if len(body)>LIMIT:raise ValueError('Publisher metadata exceeds bound')
  body.decode('utf-8')
  git_blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
  record.update(sha256=hashlib.sha256(body).hexdigest(),git_blob_sha1=git_blob)
  if expected_blob is not None:
   record.update(expected_git_blob_sha1=expected_blob,git_blob_verified=git_blob==expected_blob)
   if git_blob!=expected_blob:raise ValueError('Exact publisher Git blob differs')
  path=ART/name
  with path.open('xb') as stream:stream.write(body)
  record['path']=path.relative_to(ROOT).as_posix()
  return body
 except Exception as error:
  record.update(error_type=type(error).__name__,error=str(error))
  if isinstance(error,urllib.error.HTTPError):record['status']=error.code
  return None
 finally:records.append(record)
def main():
 tree_body=fetch('playable_tree.json',f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1')
 if tree_body is None:raise RuntimeError('Tree inventory unavailable')
 tree=json.loads(tree_body)
 if tree.get('truncated') is not False:raise RuntimeError('Tree inventory incomplete')
 entries={entry['path']:entry for entry in tree['tree']}
 requested=['LICENSE','README.md','dataset/acquisition/tennis/scripts/download_tennis_v7.sh','dataset/acquisition/tennis/tennis_annotations/tennis_youtube_videos.csv','dataset/acquisition/tennis/tennis_annotations/tennis_youtube_splits_annotations.yaml']
 inventory=[]
 for name in requested:
  entry=entries.get(name)
  inventory.append({'requested_path':name,'in_pinned_tree':entry is not None,'git_entry':entry})
  if entry is None:continue
  if entry['type']!='blob' or entry.get('size',LIMIT+1)>LIMIT:raise ValueError('Requested text path is not bounded blob')
  if name.endswith('.yaml'):
   # Inspect only text shape; no model labels/targets may be acquired here.
   # The source/script identifies this as timeline split metadata. Root can
   # authorize a subsequent individual bound request after a schema inventory.
   inventory[-1]['acquired']=False
   inventory[-1]['reason']='Deferred until time/source-only semantics independently confirmed'
   continue
  body=fetch('playable_'+Path(name).name,f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{name}',entry['sha'])
  inventory[-1]['acquired']=body is not None
 receipt={'schema_version':1,'observed_utc':datetime.now(timezone.utc).isoformat(),'status':'PUBLISHER_TEXT_METADATA_ACQUIRED_NO_MEDIA_OR_QUALIFICATION','repository':REPO,'pinned_revision':REVISION,'tree_sha':tree['sha'],'tree_truncated':tree['truncated'],'requested_inventory':inventory,'requests':records,'limits':{'max_per_response_bytes':LIMIT,'media_archive_images_test_targets_acquired':False,'publisher_code_executed':False,'model_imports_ram_gpu_probes_or_training':False,'split_or_accuracy_qualification':False}}
 with (ART/'acquisition_receipt.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
 print(json.dumps({'status':receipt['status'],'revision':REVISION,'inventory':inventory,'requests':records},indent=2))
if __name__=='__main__':main()
