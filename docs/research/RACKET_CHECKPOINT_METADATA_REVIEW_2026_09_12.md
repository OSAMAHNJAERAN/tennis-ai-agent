# Racket specialist checkpoint metadata: restricted-loading review

Date: 2026-09-12. **No complete narrow loading path has been proved.** This review supports a bounded design investigation; it does not authorize blanket `getattr` permission, claim the checkpoint impossible to load safely, or establish model/runtime parity. No checkpoint was loaded, no resolver or unpickler was implemented, and no package was installed or modified during this review.

## Inspected artifact and method

The parent task's `artifacts/models/racket/rtmdet_research/inspection.json` records checkpoint revision `a3760773233a0988c9605259743fbdd87c59d3a3`, 411293859 bytes, SHA256 `e6ad74371259d844b11529a64b09052edaec4277ce9ebeeca64d77b9131a19cd`, ZIP CRC verification, and a default weights-only rejection at `HistoryBuffer`. This is an actual restricted-load failure reported by the parent, not a failure newly executed here. [Publisher checkpoint](https://huggingface.co/linfeng302/RacketVision-Models/blob/a3760773233a0988c9605259743fbdd87c59d3a3/checkpoints/epoch_300.pth).

This review read only ZIP `data.pkl` and enumerated it with Python `pickletools.genops`, without invoking pickle reconstruction. The pickle payload is 5194403 bytes, SHA256 `e879a3da8e4b2e1cc88bd0611b763eb2a5367ae5899911d43c07d545a4517b65`, containing 99147 opcodes, 11 GLOBAL declarations, 19 NEWOBJ operations and 61 BUILD operations. Offsets below are zero-based byte positions in this payload.

## Every getattr target

The sole declaration is `GLOBAL '__builtin__ getattr'` at 761557, stored once as memo 7200 at 761578. Memo 7200 has exactly three later reads, and no reassignment. All four immediate call sequences construct a two-item tuple whose first item is memo 7158: the **HistoryBuffer class**, declared at 154273 and memoized at 154320. These are class functions, not methods bound to a HistoryBuffer instance.

| Retrieval | Callable source offset | Class read | Literal name | REDUCE call | Result memo |
|---|---:|---:|---:|---:|---:|
| HistoryBuffer.min | 761557 GLOBAL | 761583 | 761588 | 761607 | 7203 |
| HistoryBuffer.max | 761626 GET 7200 | 761631 | 761636 | 761655 | 7207 |
| HistoryBuffer.current | 761678 GET 7200 | 761683 | 761688 | 761711 | 7211 |
| HistoryBuffer.mean | 761731 GET 7200 | 761736 | 761741 | 761761 | 7215 |

The four returned-function memos have **zero subsequent GET references**. The immediate opcode sequence stores them in the `statistics_methods` dictionary (memo 7198); that dictionary is reused 18 times by later history objects. Thus the observed getattr calls retrieve these four functions for state restoration rather than invoke their statistics computations. This conclusion is specific to the hash above, derived by enumerating every declaration, assignment and reference to the relevant memos; it is not a general statement about arbitrary training checkpoints.

## HistoryBuffer hooks and side effects

The official MMEngine 0.10.7 source defines an ordinary class with no custom `__new__`, `__reduce__`, `__reduce_ex__`, or `__setattr__` implementation. Its two explicit pickle hooks are `__getstate__` and `__setstate__`. `__getstate__` is a serialization hook and is not called merely by restoring these objects. [Complete official class source](https://raw.githubusercontent.com/open-mmlab/mmengine/v0.10.7/mmengine/logging/history_buffer.py).

The 19 NEWOBJ positions are 154326, 761794, 1368732, 1898816, 2474634, 3051770, 3629235, 4206447, 4780785, 4824167, 4870331, 4871565, 4872794, 4874038, 4875269, 4876496, 4877716, 4878950 and 4880145. Each is immediately preceded by the original HistoryBuffer class declaration or memo 7158 plus an empty argument tuple. No other class has a NEWOBJ operation in the enumerated stream.

The installed Torch standard weights-only implementation calls an allowed class's `__new__` for NEWOBJ and calls its `__setstate__` when processing BUILD. Therefore allowlisting HistoryBuffer is not passive metadata acceptance. Its official `__setstate__` removes the statistics dictionary from incoming state, registers four defaults, updates the **class-level shared statistics registry**, then updates the instance dictionary. It does not call the four statistics functions, but it mutates shared class state. The 18 reused dictionaries also mean this mutation/pop behavior needs consideration in any equivalence test. These facts follow from the full official class source and installed `torch/_weights_only_unpickler.py` lines 384–446.

## Remaining globals and NumPy compatibility

All eleven GLOBAL declarations, as read from this exact pickle:

| Offset | Serialized global | Status relevant to this review |
|---:|---|---|
| 20091 | collections.OrderedDict | Standard tensor-checkpoint container; not rejected by the parent's static default-global inspection. |
| 20157 | torch._utils._rebuild_tensor_v2 | Standard tensor reconstruction. |
| 20208 | torch.FloatStorage | Tensor storage. |
| 20725 | torch.LongStorage | Tensor storage. |
| 154273 | mmengine.logging.history_buffer.HistoryBuffer | Actual first rejection; hook behavior reviewed above; package not installed. |
| 154386 | numpy.core.multiarray._reconstruct | Nondefault NumPy array reconstruction; modern installed symbol is `numpy._core.multiarray._reconstruct`. |
| 154427 | numpy.ndarray | Nondefault array type with state restoration. |
| 154455 | _codecs.encode | Existing standard weights-only mapping; used for encoded numeric payloads. |
| 154539 | numpy.dtype | Nondefault dtype constructor. |
| 761557 | __builtin__.getattr | Torch normalizes this legacy name to `builtins.getattr`; four calls exhaustively described above. |
| 4901292 | numpy.core.multiarray.scalar | Nondefault scalar reconstruction; modern symbol is `numpy._core.multiarray.scalar`. |

Installed NumPy is 2.3.1. Symbol identities and names were inspected without loading the checkpoint. The only direct dtype construction sequences are `f8` at 154539/REDUCE 154577 and `i8` at memo-7174 read 511808/REDUCE 511833; both use little-endian dtype state. Their resulting classes are `numpy.dtypes.Float64DType` and `numpy.dtypes.Int64DType`. The scalar reduction at 4901378 uses the f8 dtype memo and an encoded eight-zero-byte payload. Explicit legacy-name aliases would be necessary where modern symbol module names differ. PyTorch documents that dynamically constructed NumPy dtype types can require additional reviewed allowance beyond statically listed GLOBAL names. [PyTorch guidance](https://docs.pytorch.org/docs/2.9/notes/serialization.html#troubleshooting-weights-only).

This enumeration is not a complete proof that every NumPy array BUILD state, tensor storage, size, and callable argument is valid or benign. NumPy's native reconstruction/state hooks would execute if allowed. No broad promise that allowing the listed classes is sufficient follows from inspecting GLOBAL names and relevant local opcode spans.

## What standard weights-only can and cannot express

The installed standard API supports allowlist entries `(callable_or_class, full_serialized_name)`. Its resolver maps that name to the provided object, and REDUCE checks that the callable is allowed before invoking it. This is documented behavior and confirmed in installed `torch/serialization.py` lines 282–294 and `_weights_only_unpickler.py` lines 130–146 and 402–418. [Official API documentation](https://docs.pytorch.org/docs/2.9/notes/serialization.html#torch.serialization.add_safe_globals).

- Merely allowlisting `HistoryBuffer.min/max/current/mean` does **not** satisfy the stream's `builtins.getattr` lookup; those functions are obtained as REDUCE results, not their own GLOBAL declarations.
- Allowlisting Python's real `getattr` would authorize broader attribute lookup than these four observed pairs. The standard allowlist has no built-in per-argument restriction for that callable.
- A separately reviewed resolver, registered through the standard callable/name alias mechanism, could in principle accept only the exact HistoryBuffer class identity and four literal names, returning explicitly mapped official functions and rejecting everything else. That would avoid broad Python getattr authority and would not replace Torch's unpickler. However, it is **new reconstruction code**, not a proven existing safe path. No such resolver was implemented or tested here.
- The official class would still need reviewed availability and its state hook would still execute; NumPy aliases, types and states remain additional gates. Substituting a dummy HistoryBuffer changes reconstruction semantics and has not been established as correct by this review.

**Recommendation:** leave the specialist unexecuted until an explicit bounded loading design is reviewed and validated against this immutable artifact. Prefer a publisher tensor-only state dictionary or safetensors artifact if available. A validated loader would establish access to tensors, not architectural equivalence or accuracy; reference parity for the proposed inference port remains separate. This is a blocker for this specialist candidate only, not for continued tennis-system work.

## Installed serialization source hashes

Root: `C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/torch`.

| File | SHA256 |
|---|---|
| serialization.py | `dc50434201b513a68f3bbb107e0f33a2f45596db31c055aaee1da0d02e75c492` |
| _weights_only_unpickler.py | `7daa1bb27151154c4e5ae46d3c4ce78737450796171cf6e5d5e36de99435c3ba` |

## Resolver tests and approval boundary

The parent implemented a four-function resolver using the standard callable/name alias API and archived the unmodified official HistoryBuffer source (SHA256 `a1cca4a5ba106caa37782877ca70f66d3e760be6a155cd6b4ea8411d0ecbfca1`). Eleven tests pass, including actual standard weights-only reconstruction of synthetic allowed requests and rejection of unreviewed attribute names, owners, instances, subclasses and string subclasses. These tests demonstrate resolver behavior only; they do not prove that restoring the public checkpoint's entire metadata is safe.

Automatic approval review rejected the real tensor-export execution because it would run allowlisted HistoryBuffer state restoration and NumPy reconstruction without a proven safe path or explicit user authorization for that step. The command did not execute and no tensor export was produced. The user was asked for explicit approval while unrelated ball-model work continued. `export_racket_specialist_tensors.py` has an explicit execution gate pending that approval. Do not reroute, remove the gate or deserialize the checkpoint without resolving the approval boundary. The candidate's inference and accuracy remain unmeasured.


Continuation decision: automatic approval review rejected a second attempt because the user discretion reply did not explicitly authorize the disclosed executable metadata restoration. A follow-up again delegated judgment. The chosen path is to leave this RTMDet load blocked and pursue independent model candidates. Its runtime gate remains in place; no restricted or unrestricted real checkpoint load has completed. TOTNet evaluation is a separate checkpoint and data-container-only extraction path.
