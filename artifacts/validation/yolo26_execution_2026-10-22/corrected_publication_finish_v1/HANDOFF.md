# Corrected publication integrity repair

The saved 18 October v4 candidate had no terminal receipt: its runner stopped after ten progress characters. It was treated as unverified. The former root coupled run likewise had no terminal receipt and its process handle was missing. This repair retained the actual production publisher and tests in `baseline/`, then applied the intended candidate and added temporary-pointer binding after a fresh successful fault reproduction.

The current publisher captures its newly closed database record before native generation-copy callbacks, independently opens that database under a bounded read owner, and streams its root against the already sealed delivery report. Its final callback reconstructs the canonical native manifest and generation token from those trusted producer records. It verifies database, assets, generated result/summary, native manifest and the exact temporary pointer before and after the durable owner callback. Ordinary acknowledgement exceptions reconcile an actual commit; KeyboardInterrupt/SystemExit preserve that commit and propagate.

The pointer fault probe actually returned a new token while the current pointer selected a nonexistent `ffff…` generation. This is historical defect evidence, not a passing result: `pointer_probe/receipt.json`, SHA256 `4cf30cf726112827ee33476703e93d0eac65ae6b1f3fcf37ab86200c71594776`. Five source pins were stable and no inference runtime was imported.

Current source SHA256:

- `src/scoring/corrected_publication.py`: `2be85fad6c748c173cd36245e5d3fe87b58781ead5e46b21d4da6863e2e34e34`.
- `tests/test_corrected_publication.py`: `f9ef03268659f9015ccdf4f12e025f76b4c3c30c99e3b63fa74554d84067660c`.

The focused current-production run passed **9 tests**, zero errors/failures/skips, in **83.971 seconds**, with all **47** selected source/dependency/test/runner pins stable. It covers pre-callback database and native manifest damage, valid-SQLite root mutation before initial byte capture, temporary pointer damage before/inside authorization, both critical post-commit interrupts, ordinary lost acknowledgement/idempotent retry, and final-callback database corruption. Receipt SHA256: `8a17510a50141362e26d983647588db5e34164cca6d5631b11daf5dacd220f41`.

The complete publisher test file now defines **41** cases for the parent's coupled verification; this agent did not claim a separate full 41-case run. Independent read-only closure is retained in the sibling `corrected_publication_review_closure_v1/` directory. All media, observations and committed generations are explicit isolated synthetic fixtures. No model/CUDA inference, independent vision/physical accuracy, long-match or live operation is qualified by these tests.
