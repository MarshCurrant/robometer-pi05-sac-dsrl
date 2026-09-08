# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import glob
import hashlib
import importlib
import os
import re
import sys
from collections import deque
from typing import Optional, Union

import gym
import numpy as np
import torch
from openpi_client import image_tools
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.libero.robometer_client import (
    OfficialRobometerPool,
    compose_robometer_reward,
    official_subsample_and_pad,
    success_window_detected,
    terminal_progress_contexts,
)
from rlinf.envs.libero.utils import (
    build_interleaved_eval_reset_state_ids,
    distribute_reset_state_ids_round_robin,
    get_benchmark_overridden,
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
    record_completed_episode_task_stats,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger


def _repoint_libero_config(libero_module) -> None:
    """Point LIBERO's cached config at the package that is actually installed.

    LIBERO writes absolute paths into a config file under ``$HOME`` the first
    time it is imported and afterwards only reads it back, so a config left by
    another venv on the same machine silently redirects asset and init-state
    lookups to directories this venv does not have.
    """
    installed_root = os.path.dirname(os.path.abspath(libero_module.__file__))
    try:
        configured_root = libero_module.get_libero_path("benchmark_root")
    except Exception:
        configured_root = None
    if configured_root != installed_root:
        libero_module.set_libero_default_path(installed_root)


logger = get_logger()


def _initial_observation_fingerprint(obs: dict[str, np.ndarray]) -> int:
    """Return a stable, exactly float32-representable visual reset identifier."""
    image = np.ascontiguousarray(obs["agentview_image"])
    digest = hashlib.blake2b(image.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & ((1 << 24) - 1)


def _read_bddl_language_and_goal(bddl_path: str):
    """Parse (:language ...) and a compact (:goal ...) summary from a BDDL file."""
    try:
        with open(bddl_path, "r", encoding="utf-8") as f:
            bddl_text = f.read()
    except OSError:
        return None, None
    lang_m = re.search(r"\(:language\s+([^)]+)\)", bddl_text)
    language = lang_m.group(1).strip() if lang_m else None
    goal_m = re.search(r"\(:goal\s*\n?\s*\(And\s*\(([^)]+)\)\)", bddl_text)
    if goal_m is None:
        goal_m = re.search(r"\(:goal[\s\S]*?\(And\s*\(([^)]+)\)\)", bddl_text)
    goal = goal_m.group(1).strip() if goal_m else None
    return language, goal


libero_type = get_libero_type()

if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    LIBERO_PKG_NAME = f"libero{libero_type}"
    LIBERO_MAIN_MODULE_PATH = f"{LIBERO_PKG_NAME}.{LIBERO_PKG_NAME}"
    try:
        real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
        real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

        try:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
            )
        except ImportError:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_PKG_NAME}.benchmark"
            )

        try:
            real_libero_envs = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.envs"
            )
        except ImportError:
            real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        print(
            f"[Main Process Routing Error] Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Error: {e}"
        )

if libero_type == "pro":
    import liberopro.liberopro as _libero_core
    from liberopro.liberopro.benchmark import Benchmark
elif libero_type == "plus":
    import liberoplus.liberoplus as _libero_core
    from liberoplus.liberoplus.benchmark import Benchmark
else:
    import libero.libero as _libero_core
    from libero.libero.benchmark import Benchmark

# Must run before any benchmark lookup: get_task_init_states() reads the cached
# config, so repointing it later (e.g. when resolving bddl_files) is too late.
_repoint_libero_config(_libero_core)


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        if seed_offset == 0:
            self._log_evaluation_mode()
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        self.init_state_source = str(
            cfg.get("init_state_source", "official_state")
        ).lower()
        if self.init_state_source not in {"official_state", "procedural"}:
            raise ValueError(
                "init_state_source must be 'official_state' or 'procedural', got "
                f"{self.init_state_source!r}"
            )
        if self.init_state_source == "procedural":
            if self.task_id_filter is None or len(set(self.task_id_filter)) != 1:
                raise ValueError(
                    "procedural LIBERO resets require exactly one task_id_filter; "
                    "switching task environments would restart their RNG streams"
                )
            if self.specific_reset_id is not None:
                raise ValueError(
                    "procedural LIBERO resets are incompatible with specific_reset_id"
                )

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.is_eval = cfg.get("is_eval", False)
        self.reward_signal_source = str(
            cfg.get("reward_signal_source", "environment")
        ).lower()
        if self.reward_signal_source not in {"environment", "robometer"}:
            raise ValueError(
                "reward_signal_source must be 'environment' or 'robometer', got "
                f"{self.reward_signal_source!r}"
            )
        self.success_signal_source = str(
            cfg.get("success_signal_source", "environment")
        ).lower()
        if self.success_signal_source not in {"environment", "robometer"}:
            raise ValueError(
                "success_signal_source must be 'environment' or 'robometer', got "
                f"{self.success_signal_source!r}"
            )
        if self.is_eval and self.success_signal_source != "environment":
            raise ValueError(
                "Evaluation must use environment success; RoboMeter is a training "
                "signal, not the objective evaluation label"
            )
        if self.is_eval and self.reward_signal_source != "environment":
            raise ValueError(
                "Evaluation must use environment reward; RoboMeter evaluation is "
                "available through shadow scoring"
            )
        if self.success_signal_source == "robometer" and not self.auto_reset:
            raise ValueError(
                "RoboMeter success detection requires auto_reset=true so detected "
                "episodes do not continue as idle replay"
            )
        default_reset_sampling = (
            "ordered"
            if self.is_eval or cfg.use_ordered_reset_state_ids
            else "random"
        )
        self.reset_sampling = str(
            cfg.get("reset_sampling", default_reset_sampling)
        ).lower()
        if self.reset_sampling not in {"ordered", "random"}:
            raise ValueError(
                "reset_sampling must be 'ordered' or 'random', got "
                f"{self.reset_sampling!r}"
            )

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        if self.is_eval:
            pool = self.reset_state_ids_all[self.seed_offset]
            self._eval_reset_pool = pool[pool >= 0].copy()
        else:
            self._eval_reset_pool = np.array([], dtype=np.int64)
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._procedural_init_fingerprints = np.full(
            self.num_envs, -1, dtype=np.int64
        )

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

        self._robometer_client: OfficialRobometerPool | None = None
        self._robometer_histories: list[list[np.ndarray]] = [
            [] for _ in range(self.num_envs)
        ]
        self._robometer_success_windows: list[deque[float]] = []
        self.robometer_async_relabel = False
        robometer_cfg = cfg.get("robometer", {})
        self.robometer_shadow_enabled = bool(
            robometer_cfg.get("shadow_enabled", False)
        )
        self.robometer_enabled = (
            self.reward_signal_source == "robometer"
            or self.success_signal_source == "robometer"
            or self.robometer_shadow_enabled
        )
        if self.robometer_enabled:
            self.robometer_add_estimated_reward = bool(
                robometer_cfg.get("add_estimated_reward", True)
            )
            self.robometer_frame_sampling = str(
                robometer_cfg.get("frame_sampling", "low_level_steps")
            ).lower()
            if self.robometer_frame_sampling not in {
                "chunk_endpoints",
                "low_level_steps",
            }:
                raise ValueError(
                    "RoboMeter frame_sampling must be chunk_endpoints or "
                    f"low_level_steps, got {self.robometer_frame_sampling!r}"
                )
            self.robometer_success_threshold = float(
                robometer_cfg.get("success_detection_threshold", 0.65)
            )
            self.robometer_success_duration = int(
                robometer_cfg.get("success_detection_duration", 1)
            )
            if self.robometer_success_duration <= 0:
                raise ValueError("RoboMeter success_detection_duration must be positive")
            self.robometer_success_rule = str(
                robometer_cfg.get("success_detection_rule", "majority_window")
            ).lower()
            if self.robometer_success_rule not in {
                "all_consecutive",
                "majority_window",
            }:
                raise ValueError(
                    "RoboMeter success_detection_rule must be all_consecutive or "
                    f"majority_window, got {self.robometer_success_rule!r}"
                )
            self.robometer_terminal_delta_enabled = bool(
                robometer_cfg.get("terminal_adjacent_delta_detection", False)
            )
            self.robometer_terminal_delta_threshold = float(
                robometer_cfg.get("terminal_adjacent_delta_threshold", float("inf"))
            )
            self.robometer_terminal_delta_context_frames = int(
                robometer_cfg.get("terminal_adjacent_delta_context_frames", 4)
            )
            if self.robometer_terminal_delta_context_frames <= 0:
                raise ValueError(
                    "terminal_adjacent_delta_context_frames must be positive"
                )
            self.robometer_use_relative_rewards = bool(
                robometer_cfg.get("use_relative_rewards", False)
            )
            self.robometer_async_relabel = bool(
                robometer_cfg.get("async_relabel", False)
            )
            if self.robometer_async_relabel and self.reward_signal_source != "robometer":
                raise ValueError(
                    "RoboMeter async_relabel requires reward_signal_source=robometer"
                )
            self._robometer_prev_progress = np.zeros(
                self.num_envs, dtype=np.float32
            )
            self._robometer_latest_success_prob = np.zeros(
                self.num_envs, dtype=np.float32
            )
            self._robometer_episode_ids = np.full(
                self.num_envs, -1, dtype=np.int64
            )
            self._robometer_chunk_ids = np.zeros(self.num_envs, dtype=np.int64)
            self._robometer_async_token_meta: dict[str, dict[str, int]] = {}
            self._robometer_async_completed: dict[
                tuple[int, int], dict[int, object]
            ] = {}
            self._robometer_async_next_chunk: dict[tuple[int, int], int] = {}
            self._robometer_async_progress: dict[tuple[int, int], float] = {}
            self._robometer_success_windows = [
                deque(maxlen=self.robometer_success_duration)
                for _ in range(self.num_envs)
            ]
            server_urls = robometer_cfg.get("server_urls", None)
            if server_urls is None:
                server_urls = [
                    str(
                        robometer_cfg.get(
                            "server_url", "http://127.0.0.1:8000"
                        )
                    )
                ]
            self._robometer_client = OfficialRobometerPool(
                list(server_urls),
                max_frames=int(robometer_cfg.get("max_frames", 8)),
                timeout_s=float(robometer_cfg.get("timeout_s", 120.0)),
                connection_retries=int(
                    robometer_cfg.get("connection_retries", 2)
                ),
                strict_qwen3=bool(robometer_cfg.get("strict_qwen3", True)),
                max_pending_batches_per_server=int(
                    robometer_cfg.get("max_pending_batches_per_server", 2)
                ),
            )
            if self.success_signal_source == "robometer":
                logger.info(
                    "LIBERO training signal: official RoboMeter progress + success "
                    "detection (server=%s, max_frames=%s, threshold=%.3f, duration=%d, "
                    "rule=%s, terminal_delta=%s@%.6f/%dframes, relative_reward=%s); "
                    "simulator success is diagnostic-only",
                    self._robometer_client.server_urls,
                    self._robometer_client.max_frames,
                    self.robometer_success_threshold,
                    self.robometer_success_duration,
                    self.robometer_success_rule,
                    self.robometer_terminal_delta_enabled,
                    self.robometer_terminal_delta_threshold,
                    self.robometer_terminal_delta_context_frames,
                    self.robometer_use_relative_rewards,
                )
                if self.robometer_async_relabel:
                    logger.info(
                        "LIBERO RoboMeter uses official asynchronous relabeling: "
                        "rewards are token-matched to source transitions and success "
                        "termination is applied when the ordered callback is observed"
                    )
            elif self.reward_signal_source == "robometer":
                logger.info(
                    "LIBERO reward augmentation enabled: environment sparse reward "
                    "+ official RoboMeter absolute progress (server=%s, "
                    "max_frames=%s, frame_sampling=%s, async_relabel=%s); "
                    "simulator success and reset are unchanged",
                    self._robometer_client.server_urls,
                    self._robometer_client.max_frames,
                    self.robometer_frame_sampling,
                    self.robometer_async_relabel,
                )
            else:
                logger.info(
                    "LIBERO RoboMeter shadow scoring enabled (server=%s, "
                    "max_frames=%s); environment reward, success, and reset are "
                    "unchanged",
                    self._robometer_client.server_urls,
                    self._robometer_client.max_frames,
                )

    def _log_evaluation_mode(self):
        """Log the LIBERO evaluation mode banner (rank 0 env worker only)."""
        libero_type = get_libero_type()
        if libero_type == "pro":
            perturbation = os.environ.get("LIBERO_PERTURBATION", "all")
            logger.info(f"Evaluation Mode: LIBERO-PRO | Perturbation: {perturbation}")
        elif libero_type == "plus":
            suffix = os.environ.get("LIBERO_SUFFIX", "all")
            logger.info(f"Evaluation Mode: LIBERO-PLUS | Suffix: {suffix}")
        else:
            logger.info("Evaluation Mode: Standard LIBERO")

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

                    pkg_name = f"libero{_type_val}"
                    core_name = f"{pkg_name}.{pkg_name}"

                    try:
                        real_pkg = importlib.import_module(pkg_name)
                        real_core = importlib.import_module(core_name)
                        real_bench = importlib.import_module(f"{core_name}.benchmark")
                        real_envs = importlib.import_module(f"{core_name}.envs")

                        sys.modules["libero"] = real_pkg
                        sys.modules["libero.libero"] = real_core
                        sys.modules["libero.libero.benchmark"] = real_bench
                        sys.modules["libero.libero.envs"] = real_envs

                        loaded_path = os.path.dirname(real_core.__file__)
                        os.environ["LIBERO_ASSET_ROOT"] = os.path.join(
                            loaded_path, "assets"
                        )
                        os.environ["LIBERO_BDDL_PATH"] = os.path.join(
                            loaded_path, "bddl_files"
                        )
                        os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(
                            loaded_path, "init_files"
                        )

                        WorkerEnv = real_envs.OffScreenRenderEnv

                    except ImportError as e:
                        print(f"[Worker Env Error] {e}")
                        raise e
                else:
                    from libero.libero.envs import OffScreenRenderEnv as WorkerEnv

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        if variant == "pro":
            import liberopro.liberopro as l_pro

            _repoint_libero_config(l_pro)
            bddl_root = l_pro.get_libero_path("bddl_files")
        elif variant == "plus":
            import liberoplus.liberoplus as l_plus

            _repoint_libero_config(l_plus)
            bddl_root = l_plus.get_libero_path("bddl_files")
        else:
            import libero.libero as l_base

            _repoint_libero_config(l_base)
            bddl_root = l_base.get_libero_path("bddl_files")

        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip()

        task_descriptions = []
        pert_init_folders = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                pert_init_folders.append(
                    self._pert_init_folders[env_id]
                    if hasattr(self, "_pert_init_folders")
                    else ""
                )
                continue

            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)

            final_path = original_path

            if variant == "pro":
                pro_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = ["_lan", "_object", "_swap", "_task"]
                if pro_suffix == "all":
                    filter_perts = valid_perts
                elif pro_suffix is not None:
                    # Map bare name (e.g. "task") to directory suffix (e.g. "_task")
                    normalized = (
                        f"_{pro_suffix}"
                        if not pro_suffix.startswith("_")
                        else pro_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    all_sub_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                        and suite_keyword in d
                        and any(d.endswith(pert) for pert in filter_perts)
                    ]

                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []

                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f)
                            for f in os.listdir(target_dir_path)
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = [
                    "_light",
                    "_language",
                    "_table",
                    "_add",
                    "_tb",
                    "_sample",
                    "_level",
                ]
                if plus_suffix == "all":
                    filter_perts = valid_perts
                elif plus_suffix is not None:
                    normalized = (
                        f"_{plus_suffix}"
                        if not plus_suffix.startswith("_")
                        else plus_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    clean_name = file_name.replace(".bddl", "")
                    for marker in valid_perts:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break

                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                    ]
                    search_dirs = [
                        os.path.join(bddl_root, d)
                        for d in all_dirs
                        if suite_pattern in d.lower().replace("_", "")
                    ]

                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]

                    all_candidates = []
                    for target_dir in search_dirs:
                        matches = [
                            f
                            for f in glob.glob(os.path.join(target_dir, "*.bddl"))
                            if clean_name in os.path.basename(f)
                            and any(
                                pert in os.path.basename(f) for pert in filter_perts
                            )
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": final_path,
                    "seed": (
                        self.seed + env_id
                        if self.init_state_source == "procedural"
                        else self.seed
                    ),
                }
            )
            # LIBERO-PRO: use selected BDDL language (not original suite task.language)
            # and remember the perturbation folder for pruned_init loading.
            pert_folder = os.path.basename(os.path.dirname(os.path.abspath(final_path)))
            pert_init_folders.append(pert_folder)
            if variant == "pro":
                bddl_lang, bddl_goal = _read_bddl_language_and_goal(final_path)
                desc = bddl_lang if bddl_lang else task.language
                task_descriptions.append(desc)
                if self.is_eval:
                    logger.info(
                        "[LIBERO-PRO lang] env=%s pert_folder=%s "
                        "prompt=%r suite_orig=%r bddl_lang=%r goal=%r bddl=%s",
                        env_id,
                        pert_folder,
                        desc,
                        task.language,
                        bddl_lang,
                        bddl_goal,
                        final_path,
                    )
            else:
                task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        self._pert_init_folders = pert_init_folders
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is not None:
            num_tasks = len(self.trial_id_bins)
            validated_tids = []
            for tid in self.task_id_filter:
                if not isinstance(tid, (int, np.integer)):
                    raise ValueError(
                        f"task_id_filter must contain ints, got "
                        f"{type(tid).__name__}: {tid}"
                    )
                tid_int = int(tid)
                if tid_int < 0 or tid_int >= num_tasks:
                    raise ValueError(
                        f"task_id {tid_int} in task_id_filter is out of range "
                        f"[0, {num_tasks - 1}]"
                    )
                validated_tids.append(tid_int)
            validated_tids = sorted(set(validated_tids))

            self._valid_reset_state_ids = []
            for tid in validated_tids:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                self._valid_reset_state_ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(self._valid_reset_state_ids)
        else:
            self._valid_reset_state_ids = None

    def update_reset_state_ids(self):
        if self.reset_sampling == "ordered":
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        elif self._valid_reset_state_ids is not None:
            indices = self._generator.integers(
                low=0, high=len(self._valid_reset_state_ids), size=(num_reset_states,)
            )
            reset_state_ids = self._valid_reset_state_ids[indices]
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        if self.is_eval:
            if self._valid_reset_state_ids is not None:
                reset_state_ids = self._valid_reset_state_ids.copy()
            else:
                reset_state_ids = build_interleaved_eval_reset_state_ids(
                    self.trial_id_bins, self.cumsum_trial_id_bins
                )
            return distribute_reset_state_ids_round_robin(
                reset_state_ids, self.total_num_processes
            )

        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        self._generator_ordered.shuffle(reset_state_ids)

        # Ensure we have enough IDs for all processes by tiling if needed
        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((num_reset_states,), dtype=int)

        if self.is_eval:
            pool = self._eval_reset_pool
            if self.start_idx >= len(pool):
                return np.full((num_reset_states,), -1, dtype=np.int64)
            end = min(self.start_idx + num_reset_states, len(pool))
            n_valid = end - self.start_idx
            result = np.full((num_reset_states,), -1, dtype=np.int64)
            if n_valid > 0:
                result[:n_valid] = pool[self.start_idx : end]
            self.start_idx = end
            return result

        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.seed_offset][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        # LIBERO-PRO: load pruned_init from the selected perturbation folder
        # (e.g. libero_object_task/), not the original suite folder.
        if variant == "pro" and getattr(self, "_pert_init_folders", None):
            import liberopro.liberopro as l_pro

            init_root = l_pro.get_libero_path("init_states")
            init_state = []
            for env_id in env_idx:
                task = self.task_suite.get_task(self.task_ids[env_id])
                folder = self._pert_init_folders[env_id] or task.problem_folder
                pert_init_path = os.path.join(init_root, folder, task.init_states_file)
                states = None
                init_path = None
                used_folder = folder
                if os.path.exists(pert_init_path):
                    loaded = torch.load(pert_init_path, weights_only=False)
                    n = len(loaded) if hasattr(loaded, "__len__") else 0
                    if n == 0:
                        if self.is_eval:
                            logger.warning(
                                "[LIBERO-PRO init] empty pruned_init, skip: %s",
                                pert_init_path,
                            )
                    else:
                        states = loaded
                        init_path = pert_init_path
                if states is None:
                    msg = (
                        "[LIBERO-PRO init] perturbation init missing or empty; "
                        "suite fallback is invalid for eval "
                        f"env={env_id} wanted_folder={folder} "
                        f"file={task.init_states_file} path={pert_init_path}"
                    )
                    if self.is_eval:
                        logger.error(msg)
                        raise RuntimeError(msg)
                    states = self.task_suite.get_task_init_states(self.task_ids[env_id])
                    init_path = f"<suite:{task.problem_folder}/{task.init_states_file}>"
                    used_folder = task.problem_folder
                    logger.warning(
                        "%s; falling back to suite init for training: %s n=%s",
                        msg,
                        init_path,
                        len(states),
                    )
                trial = int(self.trial_ids[env_id])
                if trial >= len(states):
                    trial = trial % len(states)
                init_state.append(states[trial])
                if self.is_eval and env_id == env_idx[0]:
                    logger.info(
                        "[LIBERO-PRO init] env=%s folder=%s trial=%s path=%s n=%s",
                        env_id,
                        used_folder,
                        trial,
                        init_path,
                        len(states),
                    )
            return init_state

        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.sim_success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)
        self._task_success_stats: dict[int, dict[str, int]] = {}
        self._eval_seen_trials: set[tuple[int, int]] = set()

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self.sim_success_once[mask] = False
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self.sim_success_once[:] = False
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        # Only accumulate returns while not yet succeeded
        self.returns += step_reward * (~self.success_once)
        # Record episode_len at first success
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]

        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        if self.success_signal_source == "robometer":
            episode_info["sim_success_once"] = self.sim_success_once.copy()
        if self.is_eval:
            if self.init_state_source == "procedural":
                episode_info["init_state_id"] = (
                    self._procedural_init_fingerprints.copy()
                )
            else:
                episode_info["init_state_id"] = self.reset_state_ids.copy()

        # Use success episode_len for reward if already succeeded, else current elapsed
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    @staticmethod
    def _robometer_frame(raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        image = get_libero_image(raw_obs)
        image = image_tools.resize_with_pad(image, 224, 224)
        image = image_tools.convert_to_uint8(image)
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _reset_robometer_state(self, env_idx: np.ndarray) -> None:
        if not self.robometer_enabled:
            return
        for index in np.asarray(env_idx, dtype=np.int64).tolist():
            self._robometer_episode_ids[index] += 1
            episode_key = (index, int(self._robometer_episode_ids[index]))
            self._robometer_chunk_ids[index] = 0
            self._robometer_async_next_chunk[episode_key] = 0
            self._robometer_async_progress[episode_key] = 0.0
            self._robometer_histories[index] = [
                self._robometer_frame(self.current_raw_obs[index])
            ]
            self._robometer_success_windows[index].clear()
            self._robometer_prev_progress[index] = 0.0
            self._robometer_latest_success_prob[index] = 0.0

    def _score_robometer_chunk(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._robometer_client is None:
            raise RuntimeError("RoboMeter client is not initialized")
        result = self._robometer_client.score_progress_batch(
            [np.stack(history, axis=0) for history in self._robometer_histories],
            self.task_descriptions,
            sample_ids=[
                f"env{index}-step{int(self.elapsed_steps[index])}"
                for index in range(self.num_envs)
            ],
        )
        absolute_progress = np.clip(result.progress, 0.0, 1.0)
        reward = absolute_progress.copy()
        if self.robometer_use_relative_rewards:
            reward -= self._robometer_prev_progress
        self._robometer_prev_progress[:] = absolute_progress

        detected = np.zeros(self.num_envs, dtype=bool)
        for index, probability in enumerate(result.success_prob.tolist()):
            window = self._robometer_success_windows[index]
            window.append(float(probability))
            detected[index] = success_window_detected(
                list(window),
                threshold=self.robometer_success_threshold,
                duration=self.robometer_success_duration,
                rule=self.robometer_success_rule,
            )
        return absolute_progress, reward, result.success_prob, detected

    def _submit_robometer_chunk_async(self) -> list[str]:
        if self._robometer_client is None:
            raise RuntimeError("RoboMeter client is not initialized")
        # Materialize the exact official input before queuing. This bounds each
        # pending request to max_frames instead of retaining a growing episode.
        contexts = [
            official_subsample_and_pad(
                np.stack(history, axis=0), self._robometer_client.max_frames
            )
            for history in self._robometer_histories
        ]
        chunk_ids = self._robometer_chunk_ids.copy()
        sample_ids = [
            f"env{index}-episode{int(self._robometer_episode_ids[index])}"
            f"-chunk{int(chunk_ids[index])}"
            for index in range(self.num_envs)
        ]
        tokens = self._robometer_client.submit_progress_batch(
            contexts,
            self.task_descriptions,
            sample_ids=sample_ids,
            presample=False,
        )
        for env_index, token in enumerate(tokens):
            self._robometer_async_token_meta[token] = {
                "env_index": env_index,
                "episode_id": int(self._robometer_episode_ids[env_index]),
                "chunk_id": int(chunk_ids[env_index]),
            }
        self._robometer_chunk_ids += 1
        return tokens

    def _drain_robometer_async(
        self, *, wait_for_all: bool = False
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, object]],
        np.ndarray,
    ]:
        if self._robometer_client is None:
            raise RuntimeError("RoboMeter client is not initialized")
        for result in self._robometer_client.drain_completed(
            wait_for_all=wait_for_all
        ):
            metadata = self._robometer_async_token_meta.get(result.token)
            if metadata is None:
                raise RuntimeError(f"Unknown async RoboMeter token: {result.token}")
            key = (metadata["env_index"], metadata["episode_id"])
            self._robometer_async_completed.setdefault(key, {})[
                metadata["chunk_id"]
            ] = result

        progress = self._robometer_prev_progress.copy()
        success_prob = self._robometer_latest_success_prob.copy()
        reward_sum = np.zeros(self.num_envs, dtype=np.float32)
        detected = np.zeros(self.num_envs, dtype=bool)
        max_lag = np.zeros(self.num_envs, dtype=np.int32)
        updates: list[dict[str, object]] = []

        # Consume callbacks in source-chunk order for each episode. Dynamic
        # server scheduling may finish requests out of order, but reward deltas
        # and success windows must remain temporal.
        for key in list(self._robometer_async_completed):
            env_index, episode_id = key
            completed = self._robometer_async_completed[key]
            next_chunk = self._robometer_async_next_chunk.get(key, 0)
            while next_chunk in completed:
                result = completed.pop(next_chunk)
                token = result.token
                previous = self._robometer_async_progress.get(key, 0.0)
                absolute = float(np.clip(result.progress, 0.0, 1.0))
                reward = absolute - previous if self.robometer_use_relative_rewards else absolute
                self._robometer_async_progress[key] = absolute
                updates.append(
                    {
                        "token": token,
                        "reward": reward,
                        "operation": (
                            "add"
                            if self.robometer_add_estimated_reward
                            else "replace"
                        ),
                        "progress": absolute,
                        "success_prob": float(result.success_prob),
                        "server_url": result.server_url,
                        "env_index": env_index,
                        "episode_id": episode_id,
                        "chunk_id": next_chunk,
                    }
                )
                self._robometer_async_token_meta.pop(token, None)

                if episode_id == int(self._robometer_episode_ids[env_index]):
                    progress[env_index] = absolute
                    success_prob[env_index] = float(result.success_prob)
                    reward_sum[env_index] += reward
                    lag = int(self._robometer_chunk_ids[env_index]) - next_chunk - 1
                    max_lag[env_index] = max(max_lag[env_index], lag)
                    window = self._robometer_success_windows[env_index]
                    window.append(float(result.success_prob))
                    detected[env_index] |= success_window_detected(
                        list(window),
                        threshold=self.robometer_success_threshold,
                        duration=self.robometer_success_duration,
                        rule=self.robometer_success_rule,
                    )
                next_chunk += 1
            self._robometer_async_next_chunk[key] = next_chunk
            if not completed and key != (
                env_index,
                int(self._robometer_episode_ids[env_index]),
            ):
                self._robometer_async_completed.pop(key, None)

        self._robometer_prev_progress[:] = progress
        self._robometer_latest_success_prob[:] = success_prob
        return progress, reward_sum, success_prob, detected, updates, max_lag

    def flush_robometer_async(self) -> list[dict[str, object]]:
        """Finish every queued label before a replay trajectory is published."""
        if not self.robometer_async_relabel:
            return []
        _, _, _, _, updates, _ = self._drain_robometer_async(wait_for_all=True)
        if self._robometer_async_token_meta:
            raise RuntimeError(
                f"{len(self._robometer_async_token_meta)} RoboMeter labels remain pending"
            )
        return updates

    def _score_robometer_terminal_delta(
        self, eligible: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score the full-episode terminal fallback without early termination."""
        if self._robometer_client is None:
            raise RuntimeError("RoboMeter client is not initialized")
        eligible = np.asarray(eligible, dtype=bool)
        deltas = np.full(self.num_envs, np.nan, dtype=np.float32)
        detected = np.zeros(self.num_envs, dtype=bool)
        env_indices = np.flatnonzero(eligible)
        if env_indices.size == 0:
            return deltas, detected

        contexts: list[np.ndarray] = []
        tasks: list[str] = []
        sample_ids: list[str] = []
        frame_counts: list[int] = []
        scored_env_indices: list[int] = []
        for env_index in env_indices.tolist():
            frames = np.stack(self._robometer_histories[env_index], axis=0)
            if frames.shape[0] <= 1:
                continue
            endpoint_contexts, _ = terminal_progress_contexts(
                frames,
                context_frames=self.robometer_terminal_delta_context_frames,
            )
            contexts.extend(endpoint_contexts)
            tasks.extend([self.task_descriptions[env_index]] * 2)
            sample_ids.extend(
                [
                    f"env{env_index}-terminal-delta-first",
                    f"env{env_index}-terminal-delta-last",
                ]
            )
            frame_counts.append(int(frames.shape[0]))
            scored_env_indices.append(env_index)

        if not contexts:
            return deltas, detected
        result = self._robometer_client.score_progress_batch(
            contexts,
            tasks,
            sample_ids=sample_ids,
            presample=False,
        )
        for pair_index, (env_index, frame_count) in enumerate(
            zip(scored_env_indices, frame_counts, strict=True)
        ):
            first_progress = float(result.progress[2 * pair_index])
            final_progress = float(result.progress[2 * pair_index + 1])
            delta = (final_progress - first_progress) / (frame_count - 1)
            deltas[env_index] = delta
            detected[env_index] = (
                delta > self.robometer_terminal_delta_threshold
            )
        return deltas, detected

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or (
                not self.is_eval and self.init_state_source == "official_state"
            ):
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        if self.init_state_source == "official_state":
            self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus" and self.init_state_source == "official_state":
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            if self.is_eval:
                self._task_success_stats = {}
                self._eval_seen_trials = set()
                self.start_idx = 0
                pool = self.reset_state_ids_all[self.seed_offset]
                self._eval_reset_pool = pool[pool >= 0].copy()
                self.update_reset_state_ids()
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self.reset_state_ids[env_idx] = reset_state_ids
        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]
            if self.init_state_source == "procedural":
                fingerprint = _initial_observation_fingerprint(raw_obs[i])
                self._procedural_init_fingerprints[idx] = fingerprint
                # Evaluation logs use trial_id as the human-readable reset key.
                self.trial_ids[idx] = fingerprint

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        self._reset_robometer_state(np.asarray(env_idx))
        infos = {}
        return obs, infos

    def get_camera_meta(
        self, camera_name: str = "agentview", height: int = 256, width: int = 256
    ) -> dict:
        """Fetch camera intrinsics/extrinsics and depth planes.

        Returns camera calibration from worker 0's robosuite sim: intrinsic
        matrix, cam-to-world transform, and depth near/far.  The agentview
        camera is fixed in the world, so this is constant per episode.
        """
        return self.env.workers[0].get_camera_meta(
            camera_name=camera_name, height=height, width=width
        )

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Render an arbitrary camera at the requested resolution.

        Returns:
            The rendered RGB image, or a ``(rgb, depth)`` tuple when
            *depth* is True.
        """
        return self.env.workers[0].render_camera(
            camera_name=camera_name,
            height=height,
            width=width,
            depth=depth,
        )

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        simulator_terminations = np.asarray(terminations, dtype=bool).copy()
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        if self.robometer_enabled:
            self.sim_success_once |= simulator_terminations
            if self.robometer_frame_sampling == "low_level_steps":
                for index, raw_env_obs in enumerate(raw_obs):
                    self._robometer_histories[index].append(
                        self._robometer_frame(raw_env_obs)
                    )
        if self.success_signal_source == "robometer":
            step_reward = np.zeros(self.num_envs, dtype=np.float32)
            terminations = np.zeros(self.num_envs, dtype=bool)
        else:
            step_reward = self._calc_step_reward(terminations)
            infos = self._record_metrics(step_reward, terminations, infos)
            if self.ignore_terminations:
                infos["episode"]["success_at_end"] = to_tensor(terminations)
                terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos, _ = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        async_submitted_tokens: list[str] | None = None
        async_updates: list[dict[str, object]] = []
        async_lag = np.zeros(self.num_envs, dtype=np.int32)
        if self.robometer_enabled:
            if self.robometer_frame_sampling == "chunk_endpoints":
                for index, raw_env_obs in enumerate(self.current_raw_obs):
                    self._robometer_histories[index].append(
                        self._robometer_frame(raw_env_obs)
                    )
            if self.robometer_async_relabel:
                async_submitted_tokens = self._submit_robometer_chunk_async()
                (
                    progress,
                    reward,
                    success_prob,
                    head_detected,
                    async_updates,
                    async_lag,
                ) = self._drain_robometer_async()
            else:
                progress, reward, success_prob, head_detected = (
                    self._score_robometer_chunk()
                )
            terminal_delta = np.full(self.num_envs, np.nan, dtype=np.float32)
            terminal_detected = np.zeros(self.num_envs, dtype=bool)
            if (
                self.success_signal_source == "robometer"
                and self.robometer_terminal_delta_enabled
            ):
                terminal_eligible = (
                    raw_chunk_truncations.any(dim=1).detach().cpu().numpy()
                    & ~head_detected
                )
                terminal_delta, terminal_detected = (
                    self._score_robometer_terminal_delta(terminal_eligible)
                )
            detected = head_detected | terminal_detected
            reward_tensor = to_tensor(reward)
            detected_tensor = to_tensor(detected).to(torch.bool)
            if self.reward_signal_source == "robometer":
                if not self.robometer_async_relabel:
                    if self.robometer_add_estimated_reward:
                        chunk_rewards[:, -1] = compose_robometer_reward(
                            chunk_rewards[:, -1],
                            reward_tensor,
                            add_estimated_reward=True,
                        )
                    else:
                        chunk_rewards.zero_()
                        chunk_rewards[:, -1] = reward_tensor
            if self.success_signal_source == "robometer":
                raw_chunk_terminations.zero_()
                raw_chunk_terminations[:, -1] = detected_tensor
                # A successful terminal fallback is a termination, not a timeout.
                raw_chunk_truncations[detected_tensor] = False

                infos_list[-1] = self._record_metrics(
                    chunk_rewards[:, -1].detach().cpu().numpy(),
                    detected,
                    infos_list[-1],
                )
            infos_list[-1]["episode"].update(
                {
                    "robometer_progress": to_tensor(progress),
                    "robometer_success_prob": to_tensor(success_prob),
                    "robometer_detected": detected_tensor,
                    "robometer_success_head_detected": to_tensor(head_detected),
                    "robometer_terminal_adjacent_delta": to_tensor(terminal_delta),
                    "robometer_terminal_detected": to_tensor(terminal_detected),
                    "sim_success_once": to_tensor(self.sim_success_once),
                    "robometer_shadow": to_tensor(
                        np.full(
                            self.num_envs,
                            self.robometer_shadow_enabled,
                            dtype=bool,
                        )
                    ),
                }
            )

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1], eval_count_mask = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.robometer_async_relabel:
            infos_list[-1]["robometer_async_relabel"] = {
                "submitted_tokens": async_submitted_tokens,
                "updates": async_updates,
                "lag_chunks": to_tensor(async_lag),
                "pending_samples": int(self._robometer_client.pending_count),
            }

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations

            if eval_count_mask is not None:
                eval_count_mask = torch.tensor(
                    eval_count_mask,
                    dtype=torch.bool,
                    device=past_terminations.device,
                )
                chunk_terminations[:, -1] &= eval_count_mask
                chunk_truncations[:, -1] &= eval_count_mask
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        if self.is_eval:
            return self._handle_eval_auto_reset(dones, _final_obs, infos)
        obs, infos = self._handle_train_auto_reset(dones, _final_obs, infos)
        return obs, infos, None

    def _handle_eval_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        count_mask = record_completed_episode_task_stats(
            env_idx,
            final_info,
            self.task_ids,
            self.trial_ids,
            self.num_envs,
            self._eval_seen_trials,
            self._task_success_stats,
            deduplicate_trials=bool(
                self.cfg.get(
                    "deduplicate_eval_trials",
                    self.specific_reset_id is None,
                )
            ),
        )

        if self.reset_sampling == "random":
            new_reset_state_ids = self._get_random_reset_state_ids(len(env_idx))
        else:
            new_reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
        valid_mask = new_reset_state_ids >= 0
        env_to_reset = env_idx[valid_mask]
        if len(env_to_reset) > 0:
            self.reset_state_ids[env_to_reset] = new_reset_state_ids[valid_mask]
            obs, infos = self.reset(
                env_idx=env_to_reset,
                reset_state_ids=self.reset_state_ids[env_to_reset],
            )
        else:
            obs = _final_obs
            infos = {}

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool) & count_mask
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos, count_mask

    def _handle_train_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        if self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
            obs, infos = self.reset(
                env_idx=env_idx,
                reset_state_ids=self.reset_state_ids[env_idx],
            )
        else:
            obs, infos = self.reset(env_idx=env_idx, reset_state_ids=None)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool)
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        termination_bonus = self.cfg.reward_coef * terminations
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward
