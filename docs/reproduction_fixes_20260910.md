# September 10 reference-parity repairs

Reference: `sf73jk43`. Fresh comparison run: `zyiipeu3` in W&B project `efficient-wam`.
The inherited training time-limit/autoreset observation issue is intentionally NOT fixed
in this comparison, as requested. It remains a known scientific limitation.

## What changed

1. Restored the exact 30 scalar data names present in the archived reference journal.
   Restored its four namespace step fields; removed the incorrect shared `env_step`.
   No log cadence, detector aggregation, or SAC metric computation was changed.
2. Provisioned the original PaliGemma tokenizer as a SHA-256-pinned local asset. The
   decoder no longer depends on a previous OpenPI cache or GCS plugin at startup.
   An empty-cache, network-disabled subprocess test exercises real tokenization.
3. Asset checks now expand safetensors indexes and reject missing/empty shards.
   Expected revision and locally verified revision are reported separately.
   Checkpoint loading rejects missing progress/success-head tensors.
4. Corrected the critic documentation: target critics use sampled-two **mean**;
   actor Q reduction uses **minimum**. The SAC implementation itself is unchanged.
5. Evaluation manifests are per environment step and preserve repeated evaluations.
   Checkpoint metadata records environment/update counters and explicitly states that
   it is an algorithm checkpoint, not a replay/simulator snapshot. Loading requires
   `warm_start` or `evaluate`; unsupported exact resume is rejected.
6. Each run retains a source archive with file hashes, policy/reward runtime manifests,
   model_info, asset evidence, and resolved config. External servers' unknown runtime
   is recorded as unknown instead of being inferred from the local interpreter.

## What did not change

- Scientific recipe hash before and after: `856f245e36d0f02b`.
- Task, procedural resets, actor seed/init, action execution and denoising horizons.
- Reward `-1 + progress`, success thresholds, terminal fallback, no extra success bonus.
- Replay sampling, SAC loss, gradient updates, entropy, critic reductions and learning rates.
- Full 240-step simulator evaluation: step zero, then every 25k steps, 20 episodes each.
- 500k low-level-step budget, 25k-step warmup, batch 128, one critic and one actor
  update per newly collected five-action macro transition after warmup.

The SAC model/config, replay buffer, sampler, training rollout worker, detector helpers,
and episode tracker were byte-compared with the pre-fix standalone commit and are unchanged.
Equality with the retained implementation is not proof of equality to unarchived August
working-tree changes. Historical W&B workspace styling was also not archived; only scalar
types, names, axes and cadence can be certified from the journal.

## Validation and comparison

- All 99 tests passed at launch, including separate log clocks, full metric inventory,
  head completeness, cold-cache assets, immutable eval evidence, and checkpoint load intent.
- Python compilation, shell syntax and `git diff --check` passed.
- Pi0.5/RoboMeter/DINO/tokenizer local revision or content evidence passed. Qwen/LIBERO
  completeness passed; their exact revisions remain unverified from local metadata.
- Run directory: `outputs/sf73jk43_repaired_20260910`.
- Launcher log: `outputs/sf73jk43_repaired_20260910.launcher.log`.
- Managed tmux session: `sf73jk43_repaired_20260910`.
- W&B: https://wandb.ai/3268587895-fudan-university-school-of-management/efficient-wam/runs/zyiipeu3

Compare initial eval against the reference 10/20, then periodic simulator success at
matched low-level steps. Compare SAC curves at matched `online/policy/step`, not W&B's
automatic log-event `_step`. A single 20-episode point cannot establish strict statistical
equivalence, and matching initialization does not guarantee matching learning outcomes.

## Initial live result

Step-zero evaluation completed on September 10: **11/20 (55%)**, versus the archived
**10/20 (50%)**. Both use the configured full 240-step horizon. The one-episode difference
is not evidence of a meaningful performance change, nor proof of numerical identity.
The historical true step-zero episode manifest was overwritten by later evaluations,
so an exact per-initial-state comparison against August is not available.

The first eight completed training episodes produced all expected warmup train/buffer
keys and separate namespace axes. Over those buffer rows, the maximum absolute error
in `avg_total_reward - (avg_predicted_progress_reward - 1)` was `5.55e-17`. No shared
`env_step` appeared. Online SAC metrics are intentionally absent until the unchanged
25,000-low-level-step warmup ends; a learning-curve comparison is still pending.
