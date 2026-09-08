# sf73jk43 Reproduction Audit

## Evidence boundary

This repository reconstructs the scientific behavior of W&B run `sf73jk43` from its
resolved configuration, logs, checkpoints, runtime source, and environment metadata.
W&B recorded upstream commit `d5fd0b3`, but the Pi0.5 adapter was an uncommitted working-tree
change. Therefore the project claims **behavior-locked reproduction**, not bitwise source
identity. Every new run writes a resolved YAML, a scientific recipe hash, and provenance.

## Frozen transition semantics

One replay transition is one Pi0.5 macro action:

1. The SAC actor observes DINOv2 features from the main camera plus the 8D robot state.
2. It samples one bounded 32D Gaussian residual.
3. The residual is repeated over Pi0.5's native 10-slot denoising horizon.
4. Frozen Pi0.5 decodes actions and the environment executes the first 5 actions.
5. RoboMeter receives the language instruction and the trajectory history consisting of
   the initial frame plus **every executed low-level main-camera frame**. Its official sample
   builder uniformly limits that history to 8 frames.
6. Replay reward is `-1 + absolute_progress(history)`.
7. Training ignores simulator success. Three consecutive success probabilities above 0.925
   end the training episode. At the time limit only, normalized terminal progress delta above
   `0.0013611419747273127` is the fallback detector.

The actual rollout worker is authoritative for item 5. An earlier generated audit incorrectly
described endpoint-only history; that description is not reproduced here.

## SAC contract

- Five critics, with two sampled target critics and minimum reduction.
- Batch size 128; one actor and one critic update per newly collected macro transition.
- Actor and critic LR `1e-5`; entropy coefficient LR `3e-4`; automatic target entropy `-16`.
- Low-level gamma 0.99 maps once to macro gamma `0.99^5`.
- Polyak coefficient 0.005; no actor or critic gradient clipping.
- Warmup starts at 25,000 low-level steps, equivalent to 5,000 complete macro transitions.
- Pi0.5, RoboMeter, and DINOv2 remain frozen.

## Result interpretation

The archived curve peaks at 100% on 20 episodes at 175k and 225k low-level steps, then falls
to 55% at 300k. This supports learning capability but does not establish monotonic or stable
convergence. Formal comparisons must evaluate all periodic checkpoints on the same held-out
seed set and report uncertainty; selecting only the peak checkpoint is insufficient.
