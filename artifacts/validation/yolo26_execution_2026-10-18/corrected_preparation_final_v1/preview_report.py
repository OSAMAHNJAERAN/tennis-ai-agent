"""Render the actual retained private HTML in an isolated local headless profile."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
TARGET = ART/'report_preview'
TARGET.mkdir(exist_ok=False)
receipt = json.loads((ART/'receipt.json').read_text(encoding='utf-8'))
assert receipt['exit_code'] == 0 and receipt['source_stable']
matches = list((ART/'temp/pytest/test_actual_complete_preparati0').rglob('readable_report.html'))
matches = [path for path in matches if (path.parent/'corrected_preparation_manifest.json').is_file()]
assert len(matches) == 1
report = matches[0]
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
seal = json.loads((report.parent/'corrected_preparation_manifest.json').read_text(encoding='utf-8'))
expected = seal['artifacts']['readable_report.html']['sha256']
assert digest(report) == expected
profile = TARGET/'browser_profile'
screenshot = TARGET/'desktop_report.png'
chrome = Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
assert chrome.is_file() and not profile.exists()
started = time.perf_counter()
command = [str(chrome), '--headless', '--disable-gpu', '--no-first-run',
    '--disable-default-apps', '--disable-background-networking', '--disable-extensions',
    '--disable-component-update', '--no-default-browser-check', '--hide-scrollbars',
    '--host-resolver-rules=MAP * 0.0.0.0', '--user-data-dir='+str(profile),
    '--window-size=1400,6600', '--screenshot='+str(screenshot), report.as_uri()]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
try:
    stdout, stderr = process.communicate(timeout=45)
except BaseException:
    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    process.communicate()
    raise
(TARGET/'browser_stdout.txt').write_bytes(stdout)
(TARGET/'browser_stderr.txt').write_bytes(stderr)
assert process.returncode == 0 and screenshot.is_file()
assert digest(report) == expected
record = {'status': 'RENDERED_PENDING_VISUAL_REVIEW',
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'synthetic_source': True, 'private_unpublished_correction': True,
    'report_path': str(report), 'report_sha256': expected,
    'screenshot_path': str(screenshot), 'screenshot_sha256': digest(screenshot),
    'screenshot_viewport': {'width': 1400, 'height': 6600},
    'isolated_profile_path': str(profile), 'browser_parent_exit_code': process.returncode,
    'elapsed_seconds': time.perf_counter()-started,
    'scope': 'Local HTML layout screenshot only; toy synthetic evidence, no website/authentication/accuracy claim'}
(TARGET/'receipt.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps({key: record[key] for key in ('status', 'screenshot_path', 'elapsed_seconds')}))
