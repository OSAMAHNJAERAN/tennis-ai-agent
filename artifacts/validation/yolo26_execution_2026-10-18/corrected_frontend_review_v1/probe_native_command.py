"""Read-only native admission of the exact frontend-generated review command."""
import builtins
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root))
sys.dont_write_bytecode = True
original = builtins.__import__
attempts = []
def guarded(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ultralytics'}:
        attempts.append(name)
        raise AssertionError('No inference permitted in frontend command admission review')
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from src.scoring.corrected_rule_rebuild import _request
command = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
accepted,error = True,None
try:
    _request(command)
except ValueError as caught:
    accepted,error = False,str(caught)
print(json.dumps({'native_accepted':accepted,'native_error':error,'inference_import_attempts':attempts}))
