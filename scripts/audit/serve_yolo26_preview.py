"""Isolated loopback UI verification; accepted uploads do not launch a worker.

This deliberately uses separate storage and an explicitly selected frontend
build. It is a browser exercise, never live analysis or a production deployment.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.app import create_app
from src.api.jobs import Settings
import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--storage', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.build.resolve(), args.storage.resolve()):
        if not path.is_relative_to(ROOT):
            parser.error('Preview build and storage must stay inside the workspace.')
    if not (args.build / 'index.html').is_file():
        parser.error('Build the selected frontend first.')
    settings = Settings(root=args.storage.resolve(), config=ROOT / 'configs/production/yolo26.yaml', require_auth=True)
    app = create_app(settings, start_worker=False, frontend_root=args.build.resolve())
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
