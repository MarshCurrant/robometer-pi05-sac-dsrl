# Asset independence

This documents the asset-only fixes for review S1/S2/S3. Training math, rollout
timeouts, and success thresholds are unchanged.

## Provisioning and offline startup

```bash
source scripts/load_env.sh
"$PYTHON_BIN" scripts/download_assets.py --all
"$PYTHON_BIN" scripts/download_assets.py --all --check
```

`pi05` also provisions/checks its `paligemma` dependency. To install only the
tokenizer, run `scripts/download_assets.py paligemma` with the policy interpreter.
The default is `${ASSET_ROOT}/paligemma_tokenizer.model`. `load_env.sh` exports
`PALIGEMMA_TOKENIZER_PATH` with that default; an explicit override takes precedence
at runtime. An override must contain the same pinned bytes, and should be checked
at its actual path by preflight.

The tokenizer comes from the public official HTTPS GCS endpoint, pinned to object
generation `1711547605575873` and SHA-256
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
Provisioning checks the digest before atomic installation and reuses a verified
local file without downloading again. Runtime rechecks the digest and never
fetches PaliGemma from GCS or consults the OpenPI cache. Missing/corrupt files fail
with an actionable error. No `gcsfs` dependency is needed. FAST and FSQ retain
their separate asset requirements; their PaliGemma component uses this local file.

## Preflight interface

`scripts/download_assets.py` exposes offline, read-only helpers:

```python
from scripts.download_assets import check_asset, check_snapshot, file_digest

# spec is one configs/assets.yaml entry, either a dict or DictConfig.
record = check_asset(name, destination, spec)
# Structural check only, for HF directory assets:
checked_files, missing, errors = check_snapshot(name, destination)
# SHA-256 by default; optional "sha1" means Git blob SHA-1, not raw file SHA-1.
sha256 = file_digest(tokenizer_path)
```

`destination` is the asset directory for HF assets, but the complete tokenizer
file path for `paligemma`. Helpers perform no downloads, manifest writes, or
environment changes. `check_snapshot` parses every `*.safetensors.index.json`,
including nested indexes, and checks every referenced shard is a nonempty file.
Malformed/empty weight maps, unsafe shard paths, missing shards, and directories
masquerading as files are rejected. This structural check alone does not parse
safetensors payloads or prove model correctness.

Manifest schema 2 replaces ambiguous `revision` with `expected_revision` and
`verified_revision`. `ok` means required files are present and no content or
revision mismatch is known. **It does not mean the revision is verified.**

HF revision verification rehashes checked files against local HF download
metadata: SHA-256 for LFS ETags, Git blob SHA-1 for regular files. Every checked
file must have supported metadata, matching content, and the same expected
commit. Missing metadata yields `verified_revision: null` and
`revision_status: unverified`, while known content/commit mismatches fail the
check. Evidence trusts the local HF metadata, not a fresh server attestation;
`revision_scope: checked_files` does not certify every file in a snapshot.
Preserve `.cache/huggingface/download` when moving assets to retain this evidence.
Plain copied or Git-cloned directories can be structurally complete but unverified.
`--check` never infers revision identity from configured pins, folder names, or an
old asset manifest. SHA-256 verification covers the entire tokenizer file.

## Success-head completeness

The RoboMeter safetensors loader validates all expected `progress_head.*` and
`success_head.*` state entries, including biases, after key remapping and before
loading any state. The custom-head sidecar path enforces the same requirement.
Missing head entries raise `ValueError`; base-model omissions allowed for PEFT
remain allowed. Complete unchanged weights are valid, so unchanged progress-head
values no longer trigger an interactive debugger. Tensor shapes are still
validated by PyTorch. Completeness does not prove that a checkpoint was trained
or that its success detector is calibrated.

## Verification

```bash
.venv-policy/bin/python -m pytest tests/test_assets.py tests/test_asset_heads.py -q
```

The tests cover missing/malformed/nested indexes, empty shards, content/commit
mismatches, unknown provenance, digest-checked atomic tokenizer installation,
explicit missing/corrupt tokenizer failures, and complete/partial success heads.
Loader tests compile unchanged production functions without importing the GPU
training dependencies. The real-tokenizer tests require the provisioned asset
(otherwise explicitly skip), deny socket connections, and verify fresh-process
cold-cache startup plus exact token/mask equality with the original SentencePiece
encoding and state-binning rules across padding and truncation cases.

On 2026-09-10 the official downloaded tokenizer matched the previous OpenPI cache
byte-for-byte by SHA-256. The local all-assets check exited 0: PaliGemma, Pi05,
RoboMeter, and DINO verified; Qwen and LIBERO complete but revision-unverified.
No training or simulator evaluation was run for these changes.
