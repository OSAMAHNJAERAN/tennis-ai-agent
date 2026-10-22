"""Load the exact existing selected model assets on CPU with networking disabled."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
output = ROOT / 'artifacts/validation/product_installation'
output.mkdir(parents=True, exist_ok=True)
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
network_attempts = []
def blocked_network(sock, address):
    network_attempts.append(str(address))
    raise RuntimeError('Offline model verification attempted a network connection')
socket.socket.connect = blocked_network
socket.socket.connect_ex = blocked_network

started = time.perf_counter()
report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': False, 'device': 'cpu', 'inferenceRun': False, 'networkAttempts': network_attempts, 'productionQualified': False}
try:
    import torch
    # Keep this check independent of the live worker's GPU/memory occupancy.
    torch.cuda.is_available = lambda: False
    from src.pipeline.phase6_pipeline import Phase6Pipeline
    pipeline = Phase6Pipeline('configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml')
    import lap
    import numpy as np
    cost, assignment, _ = lap.lapjv(np.array([[0., 4.], [4., 0.]]))
    assets = {
        'court': pipeline.court_detector.model,
        'personAndRacket': pipeline.player_detector.model.model,
        'ball': pipeline.ball_detector.model,
        'pose': pipeline.motion_tracker.model.model,
    }
    report['loadedAssets'] = {name: {'parameters': sum(parameter.numel() for parameter in model.parameters()), 'allParametersOnCpu': all(parameter.device.type == 'cpu' for parameter in model.parameters())} for name, model in assets.items()}
    report['racketReusesPersonModel'] = pipeline.racket_tracker.model is pipeline.player_detector.model
    report['assignmentSolverWorks'] = cost == 0 and assignment.tolist() == [0, 1]
    report['fullCourtStateLoading'] = 'strict=True, weights_only=True; constructor completed on the real retained checkpoint'
    report['passed'] = not network_attempts and report['assignmentSolverWorks'] and report['racketReusesPersonModel'] and all(row['allParametersOnCpu'] and row['parameters'] > 0 for row in report['loadedAssets'].values())
except Exception as exc:
    report['error'] = type(exc).__name__ + ': ' + str(exc)
    raise
finally:
    report['elapsedSeconds'] = time.perf_counter() - started
    (output / 'offline_model_asset_load.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))
raise SystemExit(0 if report['passed'] else 1)
