"""Inspect the report tail with one inert anchor added to a separate QA copy."""
import hashlib
import json
from pathlib import Path
import subprocess

ART = Path(__file__).resolve().parent
TARGET = ART/'report_preview'
first = json.loads((TARGET/'receipt.json').read_text(encoding='utf-8'))
source = Path(first['report_path'])
original = source.read_bytes()
assert hashlib.sha256(original).hexdigest() == first['report_sha256']
needle = b'<section><h2>Statistical forecasts'
anchored = b'<section id="qa-forecast"><h2>Statistical forecasts'
assert original.count(needle) == 1
derived = original.replace(needle, anchored, 1)
assert derived.replace(anchored, needle, 1) == original
qa = TARGET/'tail_anchored_report.html'
with qa.open('xb') as stream:
    stream.write(derived)
profile = TARGET/'tail_browser_profile'
screenshot = TARGET/'report_tail.png'
assert not profile.exists() and not screenshot.exists()
command = [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    '--headless', '--disable-gpu', '--no-first-run', '--disable-default-apps',
    '--disable-background-networking', '--disable-extensions', '--disable-component-update',
    '--no-default-browser-check', '--hide-scrollbars', '--host-resolver-rules=MAP * 0.0.0.0',
    '--user-data-dir='+str(profile), '--window-size=1400,5400',
    '--screenshot='+str(screenshot), qa.as_uri()+'#qa-forecast']
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
try:
    stdout, stderr = process.communicate(timeout=45)
except BaseException:
    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    process.communicate()
    raise
(TARGET/'tail_browser_stdout.txt').write_bytes(stdout)
(TARGET/'tail_browser_stderr.txt').write_bytes(stderr)
assert process.returncode == 0 and screenshot.is_file() and source.read_bytes() == original
record = {'status': 'RENDERED_PENDING_VISUAL_REVIEW', 'browser_parent_exit_code': process.returncode,
    'source_report_sha256': first['report_sha256'],
    'qa_copy_sha256': hashlib.sha256(derived).hexdigest(),
    'only_qa_change': 'One inert section ID for scrolling; removing it exactly reproduces original report bytes',
    'screenshot_path': str(screenshot), 'screenshot_sha256': hashlib.sha256(screenshot.read_bytes()).hexdigest(),
    'screenshot_viewport': {'width': 1400, 'height': 5400},
    'synthetic_source': True, 'private_unpublished_correction': True}
(TARGET/'tail_receipt.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps({'status': record['status'], 'screenshot_path': str(screenshot)}))
