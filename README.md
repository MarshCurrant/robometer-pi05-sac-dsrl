# RoboMeter Pi0.5 SAC-DSRL

Standalone, behavior-locked reproduction of W&B run `sf73jk43`: online SAC learns a
32-dimensional residual-noise policy around a frozen Pi0.5 LIBERO policy, while RoboMeter-4B
provides dense progress rewards and training-only success detection.

The repository contains the required runtime source. It does **not** clone RoboMeter, RLinf,
OpenPI, LIBERO, or RoboMeter Policy Learning during setup. Large model and simulator assets
are downloaded separately and never committed.

## What is reproduced

The reference experiment uses `libero_spatial/4`:

> pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate

The archived run started at 50% success, reached 100% at 175k and 225k environment steps,
and fell to 55% at 300k (20 evaluation episodes per point). This is evidence that the setup
can improve the policy, but it is also evidence of instability. The complete curve is in
[`reproducibility/sf73jk43/eval_curve.csv`](reproducibility/sf73jk43/eval_curve.csv).

The original entrypoint included uncommitted source. This project therefore promises
behavioral parity against its resolved config, checkpoint, logs, and tests, not a bitwise
checkout. See [`docs/reproduction_audit.md`](docs/reproduction_audit.md).

## Training flow

One online transition follows this sequence:

1. LIBERO provides the main image, wrist image, 8D state, and instruction.
2. Frozen DINOv2 embeds the main image for the SAC actor and five-critic ensemble.
3. SAC samples one bounded 32D residual. The same residual is repeated over Pi0.5's native
   10-slot flow horizon.
4. Frozen Pi0.5 consumes main image, wrist image, state, and instruction, then decodes actions.
5. LIBERO executes the first 5 actions. This is one macro transition.
6. RoboMeter receives the instruction and main-camera history: initial frame plus every
   executed low-level frame. Its official sample builder uniformly limits history to 8 frames.
7. Replay stores `reward = -1 + absolute_progress(history)`. Simulator success is hidden from
   training. RoboMeter detection controls episode termination.
8. After 25k low-level warmup steps, SAC performs one critic update and one actor update for
   each newly collected macro transition, using replay batch size 128.

Evaluation disables RoboMeter termination and uses deterministic SAC actions plus LIBERO's
sticky simulator success over the full 240-step horizon.

## Requirements

- Linux, Python 3.11, `uv`
- NVIDIA GPU with enough memory for frozen Pi0.5, DINOv2, actor, and critic
- A second GPU is recommended for RoboMeter; same-GPU placement is supported and reproduces
  the original single-GPU resource topology when memory permits
- EGL-capable MuJoCo rendering
- A W&B account only when online logging is enabled

The tested policy runtime uses Python 3.11, PyTorch 2.11/CUDA 13, Transformers 4.53.2,
Gymnasium 0.29.1, MuJoCo 3.8.1, and RoboSuite 1.4.1. The uv constraints admit compatible
PyTorch builds because CUDA wheels differ across machines; every run records installed versions.

## Install

```bash
git clone git@github.com:MarshCurrant/robometer-pi05-sac-dsrl.git
cd robometer-pi05-sac-dsrl
bash scripts/setup.sh
source scripts/load_env.sh
```

`setup.sh` creates two isolated environments:

- `.venv-policy`: LIBERO, Pi0.5, DINOv2, SAC, and experiment scripts
- `.venv-reward`: RoboMeter-4B HTTP evaluation server

This separation prevents incompatible Torch/Transformers packages from being imported from
another project's virtualenv. The policy environment is locked by `uv.lock`; the service
environment is locked independently by `environments/reward/requirements.lock`. Setup also
installs OpenPI's version-matched Transformers replacements and verifies them.

## Download assets

Setup permits network access; training defaults to Hugging Face offline mode.

```bash
source scripts/load_env.sh
POSTTRAIN_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com \
  .venv-policy/bin/python scripts/download_assets.py --all
```

Pinned downloads are declared in [`configs/assets.yaml`](configs/assets.yaml):

- `RLinf/RLinf-Pi05-LIBERO-SFT`
- `robometer/Robometer-4B`
- `Qwen/Qwen3-VL-4B-Instruct`
- `facebook/dinov2-base`
- `RLinf/LIBERO-assets`

No task argument is needed. Training is online and does not consume LIBERO demonstrations;
the selected suite/task comes from the experiment YAML. The script writes
`assets/asset_manifest.json` with revisions and completeness checks.

Check without downloading:

```bash
.venv-policy/bin/python scripts/download_assets.py --all --check
.venv-policy/bin/python scripts/preflight.py \
  --config configs/reproduction/sf73jk43.yaml
```

## Reproduce `sf73jk43`

Set W&B identity in `.env` or the shell:

```bash
export WANDB_ENTITY=your-entity
export WANDB_PROJECT=robometer-pi05-sac-dsrl
export POLICY_GPU=0
export REWARD_GPU=1       # use 0 for same-GPU placement
export ROBOMETER_SERVER_PORT=8000
```

Launch with direct Python scripts, not an installed project CLI:

```bash
source scripts/load_env.sh
.venv-policy/bin/python scripts/run_experiment.py \
  --config configs/reproduction/sf73jk43.yaml
```

The launcher starts one isolated RoboMeter server, verifies `/model_info` identifies Qwen3-VL,
runs preflight, starts training, and terminates the server on exit. To use an already running
server, pass `--no-start-server`.

Small smoke run:

```bash
.venv-policy/bin/python scripts/run_experiment.py \
  --config configs/reproduction/sf73jk43.yaml \
  --set training.num_rollouts=100 \
  --set online_algorithm.learning_starts=50 \
  --set eval.eval_num_episodes=2 \
  --set eval.eval_freq=0 \
  --set logging.wandb_offline=true
```

These overrides intentionally change the recipe hash and are not a reference reproduction.

## YAML configuration

All scientific and runtime settings are visible in one YAML. Repeatable `--set key=value`
overrides are supported. There are no hidden Hydra defaults.

Important sections:

- `env`: LIBERO suite/task, reset seed, horizon, and termination semantics
- `dsrl`: Pi0.5 flow steps, 32D residual, native/returned/executed horizons
- `reward_model`: official frame budget, reward semantics, and detector calibration
- `online_algorithm`: SAC batch, critics, entropy, discount, update ratio, and optimizers
- `resources`: policy GPU and RoboMeter server GPU/host/port
- `eval`: deterministic held-out evaluation frequency and episode count
- `logging`: W&B destination and a 27-metric allowlist

`reproduction.recipe_hash` is an output field. Leave it as `pending`; the runner computes it
from scientific settings. Paths, W&B metadata, and output locations are deliberately excluded.

## Extend to another LIBERO task

Copy [`configs/tasks/libero_template.yaml`](configs/tasks/libero_template.yaml) and set:

```yaml
env:
  env_name: libero_object   # spatial, object, goal, 10, or 90
  task_id: 3
  task_name: exact_name_reported_by_LIBERO
reward_model:
  success_detection_threshold: 0.91
  success_detection_duration: 3
  terminal_adjacent_delta_threshold: 0.0015
```

The template inherits the frozen SAC/Pi0.5 contract but marks the run as an extension. Do not
reuse task-4 detector thresholds blindly. Calibration must use episode-disjoint trajectories:

1. Collect stochastic actor rollouts over procedural reset seeds.
2. Preserve simulator success only as an offline calibration label.
3. Choose the success-head threshold and duration on a calibration split.
4. Choose terminal adjacent-delta only as a time-limit fallback.
5. Confirm specificity and premature-trigger rate on a held-out split.
6. Freeze thresholds before online RL and evaluate with simulator success.

The task identity check prevents an incorrect task ID/name pair from starting.

## Resume and evaluate

Resume from an actor/critic checkpoint directory:

```bash
.venv-policy/bin/python scripts/run_experiment.py \
  --config configs/reproduction/sf73jk43.yaml \
  --set training.load_dir=/path/to/checkpoints/175000
```

Evaluate 50 procedural-reset episodes without a RoboMeter server:

```bash
source scripts/load_env.sh
.venv-policy/bin/python scripts/evaluate.py \
  --config configs/reproduction/sf73jk43.yaml \
  --checkpoint /path/to/checkpoints/175000 \
  --episodes 50 --seed 1000
```

Old `sf73jk43` checkpoints contain model/optimizer/counter state but not replay or environment
RNG state. Loading one is valid for evaluation; exact mid-run continuation is not claimed.

## Outputs

Each run creates `outputs/<name>_<timestamp>/` with:

- `source_config.yaml` and fully resolved `resolved_config.yaml`
- `provenance.json` and `reward_semantics_audit.json`
- periodic actor/critic/target/optimizer checkpoints
- evaluation episode manifests and local logs
- W&B files when enabled

W&B is limited to 27 core metrics plus one shared `env_step` axis (28 keys total). The central
outcome is `eval/success_rate`; reward, critic, actor, entropy, detector, replay, and gradient
diagnostics are retained without uploading video.

## Tests

```bash
source scripts/load_env.sh
.venv-policy/bin/python -m pytest
```

Tests lock reward arithmetic, detector semantics, gamma mapping, residual sharing, checkpoint
optimizer identity, procedural reset behavior, configuration inheritance, and metric count.

## License and citation

Project code is MIT licensed. Vendored components retain upstream notices and licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Cite RoboMeter, RLinf, OpenPI, and LIBERO
when using the corresponding components. Model weights and simulator assets are downloaded
from their original repositories and are not covered by this repository's MIT license.
# robometer-pi05-sac-dsrl
