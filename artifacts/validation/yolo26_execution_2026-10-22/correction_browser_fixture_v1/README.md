# Local synthetic correction browser fixture

This is a **synthetic software test**, not a tennis inference or independent accuracy example. It uses twelve generated grey 1280×960 frames at 30 FPS, actual H.264 encoding, invented explicitly labeled observations, real native rule/result writers, real account endpoints, and an isolated completed analysis. Original annotation slots are explicitly synthetic unannotated baseline copies. The real correction worker will replace them with freshly rendered overlays and reports.

Preparation passed in 5.70 seconds with all Python source/test hashes unchanged and no Torch/Ultralytics import attempts. Native rule reproduction, source/revision binding, owner result/review/media requests and guest/nonowner rejection passed. The first attempt is retained separately: its test assertion incorrectly expected the filename extension to be omitted from the display label; no product code changed to resolve that test-only mistake. The successful store is `service-v2/`.

## Launch only when the current Python verification is terminal

From the repository root:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -B 'artifacts\validation\yolo26_execution_2026-10-22\correction_browser_fixture_v1\serve.py'
```

This invokes the real `create_app(settings, start_worker=True)` and serves the current `frontend/dist` on `127.0.0.1:5198`. It checks the port before starting and writes its server PID and exact frontend index hash to `runtime.json`. No server or worker was started during fixture preparation. The worker retains the unchanged 1.5 GiB CPU resource admission floor and the selected-player GPU guards; no original inference is queued. The configured unknown-handedness correction branch does not load a model.

The worker can outlive the HTTP process by design. Before shutting down the browser test, record and stop only the verified task-owned server and its `src.api.worker` child. Do not stop any unrelated process or real product server.

## Browser journey

1. Open `http://127.0.0.1:5198/upload` and sign in using `synthetic_credentials.json`. These are disposable synthetic test credentials, not user or deployment secrets.
2. Open `http://127.0.0.1:5198/analysis/match_7a26c0de7a26c0de7a26c0de7a26c0de`. The real application resolves its automatically attached product match and preserves generation query parameters.
3. The display label is `SYNTHETIC SOFTWARE TEST - correction workflow.mp4`. Confirm original/browser playback is available. Under **Review and correct analysis**, enable **Review initial score and serve context** and change **Player 2 initial points** to **15**, keeping Player 1 as server and the second serve.
4. Enter a reason and an explicitly synthetic video evidence reference, check owner authorization and submit. The real worker must rebuild calls, shots, score, causal forecasts, video and reports, then the UI must reopen the exact returned generation. The expected software result is a double fault and one completed point, with receiver score 0–30; the forecast remains uncalibrated with one completed sample.
5. Verify generation-bound replay/exports and open the historical URL below. The old version must remain selected and read-only until explicitly choosing another result.

Original generation:

`be4ae5d9f0b5734bff08916197c6f8ff55581e0f7a71c4021ee8f5f2b97b9c8b`

Historical URL:

`http://127.0.0.1:5198/analysis/match_7a26c0de7a26c0de7a26c0de7a26c0de?generation=be4ae5d9f0b5734bff08916197c6f8ff55581e0f7a71c4021ee8f5f2b97b9c8b`

`request_example.json` records the native source-bound amendment expected from that review. Source PTS are deliberately synthetic fixture-clock references; the actual normalized browser video is CFR and has twelve decoded frames. This cannot qualify capture timing, physical measurements, real match accuracy, long-duration processing or live latency.

## Generation method and retained evidence

`generate.py` clones only the three existing test factory function ASTs into private namespace copies. Exact `fixture` literals become the valid deterministic match ID; one display filename literal becomes the explicit synthetic label. Two additional recorded fixture-only changes actually transcode generated pixels before binding their source hash, and include the original browser video slot. The base→shot→complete factory links are changed only in those private namespaces. All product code, parsers, APIs, rules, storage and generation writers execute unchanged; no fake API, frontend mock, model prediction or external transfer is used.

`fixture_manifest.json` records original/transformed AST hashes, literal locations, actual encoder command/probe, native file records, all source pins, settings and verified preparation scope. `initial_review_source.json`, `request_example.json`, `synthetic_credentials.json` and `fixture_bundle_hashes.json` retain the reproducible inputs and entry points. These establish only that this local synthetic browser fixture is ready; the browser journey itself must still be executed and recorded separately.
