"""Preserved failed readback verifier: FROZEN paths incorrectly used ROOT.

This file records the executed preparation verifier, not a benchmark workload.
It is intentionally not corrected or rerun. See final_static_review_2026-10-06.py.
"""
import ast,json,hashlib
from pathlib import Path
root=Path.cwd()
unit=root/'artifacts/validation/product_execution/result_storage_latency_measurement_2026-10-06'
helper=unit/'helper.py'
source=helper.read_text(encoding='utf-8')
tree=ast.parse(source)
compile(tree,str(helper),'exec')
values={}
for node in tree.body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
  name=node.targets[0].id
  if name in ('MONITOR','MONITOR_SHA256','FROZEN','FROZEN_RECEIPT','CURRENT','HISTORICAL'):
   if isinstance(node.value,ast.Tuple):
    values[name]=tuple(values[v.id] if isinstance(v,ast.Name) else ast.literal_eval(v) for v in node.value.elts)
   else: values[name]=ast.literal_eval(node.value)
old=json.loads((root/'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/protocol.json').read_text())
new=json.loads((unit/'protocol.json').read_text())
print(json.dumps({'constants':list(values),'oldGuardKeys':list(old),'newGuardKeys':list(new)},sort_keys=True))
for key in ('counts','modes','caseOrder','maximumChildWallSeconds','sampleSeconds','minimumAvailableRamBytes','lowRamConsecutiveSamples','maximumOwnedTreeRssBytes','preflightAvailableRamBytes','minimumFreeDiskBytes','automaticRetry'):
 assert old[key]==new[key],key
for rel,binding in values['FROZEN'].items():
 payload=(root/rel).read_bytes()
 assert len(payload)==binding['bytes'] and hashlib.sha256(payload).hexdigest()==binding['sha256'],rel
print(json.dumps({'receiptConstant':values['FROZEN_RECEIPT'],'monitorConstant':values['MONITOR'],'monitorSha':values.get('MONITOR_SHA256'),'currentPaths':len(values['CURRENT']),'historicalPaths':len(values['HISTORICAL']),'frozenFiles':len(values['FROZEN']),'runExists':(unit/'run').exists(),'preparedBindingsExists':(unit/'prepared_source_bindings.json').exists(),'staticReceipt':json.loads((unit/'static_sanity_receipt.json').read_text())},sort_keys=True))
