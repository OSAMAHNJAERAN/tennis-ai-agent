"""Read-only syntax check before any real image work."""
import ast
from pathlib import Path

unit = Path(__file__).resolve().parent
for leaf in ('benchmark.py', 'candidate.py', 'test_candidate.py'):
    path = unit / leaf
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print(leaf + ': syntax valid')
