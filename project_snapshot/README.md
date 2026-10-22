# Complete supplemental project snapshot — 2026-10-22

The source, tests, configuration, documentation and existing validation reports
are stored at their normal paths in this repository. This snapshot preserves
the additional project files previously excluded by `.gitignore`: models,
datasets, media, training runs, generated outputs, research source snapshots,
and validation fixtures.

The ZIP parts are attached to the repository's
[project-snapshot-2026-10-22 release](https://github.com/OSAMAHNJAERAN/tennis-ai-agent/releases/tag/project-snapshot-2026-10-22).
They are repository release assets, so cloning the source does not download
the entire snapshot. Git contains `index.json`, the compressed path manifest,
and the Python restoration tool.

## Download and verify

Run from the repository root with GitHub CLI and Python 3.11 or newer:

```powershell
gh release download project-snapshot-2026-10-22 --repo OSAMAHNJAERAN/tennis-ai-agent --pattern "objects-*.zip" --dir project_snapshot
python scripts/maintenance/restore_project_snapshot.py
```

Verification checks the manifest and every archive against SHA-256, then checks
every unique archived object's SHA-256, size, and manifest reference.

## Restore original paths

```powershell
python scripts/maintenance/restore_project_snapshot.py --destination .
```

Identical existing files are retained. A different existing file stops the
restore; add `--overwrite` only when replacing it is intended. To restore into
a separate directory instead, use `--destination C:\path\to\restored-project`.

The archives use content-addressed objects to store identical bytes once.
`manifest.jsonl.gz` records every original relative path, byte length, SHA-256,
and archive part. Restoration creates independent files at those original
paths, including duplicate files. No third-party Python packages are needed.

`index.json` records file counts, source and archive sizes, part checksums, and
the excluded runtime directories. Git internals, installed dependencies and
virtual environments, interpreter/package caches, filesystem junctions and
temporary upload working files are excluded. Source environment examples are
retained. This is a point-in-time snapshot of the inventoried files.
