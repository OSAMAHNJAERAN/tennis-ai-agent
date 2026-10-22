# Frozen court wave acquisition checks

The new `scripts/data/acquire_yolo26_court_source_wave.py` is restricted to the root-reviewed v2 plan SHA256 `27d72148f4173ab476e9dbaaa5df6b64ce392864fd7150e6d7c0b59e893af65a` and fresh dataset `data/external/court_source_wave64_v1`. It imports standard-library modules until actual media admission passes. No acquisition has been attempted at the time of these checks; root requested a hold while the ball transform preview runs.

Actual validation: **58 tests passed in 0.30 seconds** (29 selection invariants and 29 acquisition invariants). The JUnit and hash-bound receipt are alongside this note. Tests performed no network transfer, image pixel decoding, model or GPU operations.

The acquirer enforces the unchanged **1 GiB available physical RAM and 2 GiB free disk** media floors before importing the HTTP helper or creating the dataset, again after the HTTP import, and before every member request. Available RAM and owned RSS use standard-library Win32 APIs. Only one member is inflight and the aggregate ceiling is 80,000,000 bytes; the frozen successful wave must receive exactly 52,987,635 range bytes.

Each range verifies HTTP 206 and the exact pinned archive size through the existing hash-pinned `BoundedHTTPFile`. The member parser checks local-header signature, encryption/unsupported flags, compression method, exact name, header sizes/CRC including ZIP64 size extras or signed/unsigned data descriptors, bounded deflate output and publisher CRC. Native PNG signature, IHDR 1280×720 geometry and every PNG chunk CRC are verified without decoding pixels. These structural checks do not substitute for subsequent visual review or prove the annotation quality.

Images are written exclusively, flushed, fsynced and SHA256 checked. Every successfully completed member receives a new immutable receipt. Existing datasets, partial images and attempt directories cause an explicit refusal; no overwrite, silent resume, resampling, retry loop, split assignment or label export occurs. A complete manifest is written only after all 64 images, exact range total, every file SHA256 and all frozen metadata inputs verify. The attempt receipt distinguishes pre-network resource refusal, runtime guard stop and other failures while preserving partial files.

Run the one authorized attempt only after root releases the media hold:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -u -m scripts.data.acquire_yolo26_court_source_wave --attempt artifacts/validation/yolo26_execution_2026-10-15/court_source_wave64_acquisition_attempt_v1
```

No image is suitable for supervised export until native view/context and fourteen-intersection visibility review, conservative compilation/event union with all prior links, and root second pass. All splits remain unassigned. Existing LondonFinals/MonteCarlo components stay DEV; independent match, human ground truth and final-test qualification claims remain unsupported.
