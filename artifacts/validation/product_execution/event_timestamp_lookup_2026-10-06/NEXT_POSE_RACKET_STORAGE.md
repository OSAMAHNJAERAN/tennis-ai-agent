# Next capacity intervention: per-frame pose and racket storage

Read-only review on 6 October 2026 found that Phase6Pipeline retains all pose_frames and racket_frames from the observation pass through shot linking, analytics, rendering and late raw exports. Only the renderer and those exports consume the lists. Shot linking uses its separate pose extractor.

Replace each enabled list with an owned append-only identity frame sequence under the existing ExitStack. Preserve disabled literal [], one observation per frame, pose-then-racket producer order, aligned boxes/timestamps and fresh indexed snapshots. Store ordered identity/value pairs rather than JSON object keys: rendering requires integer identity 1 to preserve pose colors. Missing pose remains None; missing racket remains its complete MISSING record. Keep raw nonfinite values until export; sanitize one frame at a time there.

Raw exports currently use json.dump(indent=2), native text newline translation, insertion order and no final newline. JsonArrayArtifact uses compact LF records and would change bytes. Use a narrowly tested lazy encoder adapter or a dedicated byte-identical writer. Do not sanitize the full lazy root/array, which would rematerialize all frames.

Use bounded SQLite page cache and commits; close every cursor before returning. Dispose only the exact owned database and sidecars on success and failure, preserving the primary error and attaching cleanup errors as notes. Existing registration/candidate sequences deliberately retain failed diagnostic databases, so their policy cannot be copied blindly. Broader NumPy-valued test/substitute outputs require an explicit codec decision; current real tracker outputs are native Python values.

Verify original complete raw bytes with disabled, missing, dense, zero, signed-zero and precise-float data; visible skeletons for both identities and a detected racket; object lifetime bounds at increasing frame counts; independent indexed snapshots; and failure injection at setup, append, tracker, rendering, serialization and cleanup. Preserve unrelated files. Rendering capture/writer cleanup currently needs a finally block.

This intervention is not implemented by the timestamp lookup unit. Ground positions, timestamps, motion samples, selected players and trajectory arrays remain resident. Full-match capacity and scientific accuracy remain unqualified.
