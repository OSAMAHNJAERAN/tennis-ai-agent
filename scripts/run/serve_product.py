"""Serve the built website and analysis API from one local origin."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('src.api.app:app', host='127.0.0.1', port=8000)
