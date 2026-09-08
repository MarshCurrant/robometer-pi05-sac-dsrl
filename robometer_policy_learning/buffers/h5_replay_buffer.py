import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch

from robometer_policy_learning.buffers.base_replay_buffer import (
    BackgroundSampler,
    BaseReplayBuffer,
    Transition,
)
from robometer_policy_learning.buffers.samplers import BaseSampler


class H5ReplayBuffer(BaseReplayBuffer):
    """
    Generic HDF5-based offline replay buffer with optimized caching and batch image loading.

    Assumes datasets follow a robomimic-style layout:
      /data/{demo}/actions
      /data/{demo}/obs/{key}
      /data/{demo}/rewards
      /data/{demo}/dones
      language annotation under /data/{demo}/obs/language

    Subclasses can override protected hooks to adapt to environment-specific variations
    (e.g., action dataset names, special prefixes for episode IDs, rewards/dones presence).
    """

    def __init__(
        self,
        h5_paths: Union[str, List[str]],
        obs_keys: List[str] = None,
        remove_obs_keys: List[str] = None,
        rename_obs_keys: Dict[str, str] = None,
        min_action: float = None,
        max_action: float = None,
        post_transforms: List = None,
        sampler: Optional[BackgroundSampler] = None,
        dataset_weights: List[float] = None,
        hdf5_cache_mode: str = "low_dim",
        hdf5_use_swmr: bool = True,
        # Performance options
        enable_full_caching: bool = True,
        batch_image_loading: bool = True,
        num_image_loading_threads: int = 4,
        pre_convert_to_tensors: bool = True,
        optimize_tensor_dtype: bool = True,
    ):
        super().__init__(
            obs_keys=obs_keys,
            remove_obs_keys=remove_obs_keys,
            rename_obs_keys=rename_obs_keys,
            pre_transforms=[],
            post_transforms=post_transforms,
            sampler=sampler,
        )

        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]

        self.h5_paths = h5_paths
        self.obs_keys = obs_keys
        self.remove_obs_keys = remove_obs_keys or []
        self.rename_obs_keys = rename_obs_keys
        self.min_action = min_action
        self.max_action = max_action
        self.dataset_weights = dataset_weights

        # Caching/perf
        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode
        self.hdf5_use_swmr = hdf5_use_swmr
        self.enable_full_caching = enable_full_caching
        self.batch_image_loading = batch_image_loading
        self.num_image_loading_threads = num_image_loading_threads
        self.pre_convert_to_tensors = pre_convert_to_tensors
        self.optimize_tensor_dtype = optimize_tensor_dtype

        self._hdf5_files: Dict[str, h5py.File] = {}
        self.hdf5_cache: Optional[Dict[str, Dict[str, Any]]] = None

        # Image loading helpers
        from collections import OrderedDict

        self.image_lru_cache = OrderedDict()
        self.max_image_cache_size = 1000
        self.image_loading_executor = None
        self.image_prefetch_queue: List[Any] = []

        # Derived during load
        self.image_keys: List[str] = []
        self.low_dim_keys: List[str] = []
        self._demo_data_lengths: Dict[str, Any] = {}
        self._index: List[Dict[str, Any]] = []
        self._episode_boundaries_index: Dict[str, Tuple[int, int]] = {}
        self.transitions: List[Transition] = []

        # Optional language string mapping; subclasses may populate
        self._demo_id_to_demo_lang_str: Dict[str, str] = {}

        # Default removes
        self.remove_obs_keys.append("object")
        self.remove_obs_keys.append("datagen_info")

        self._load_with_optimizations()
        self._build_lightweight_index_if_needed()

    # --------------------------
    # Subclass hooks
    # --------------------------
    def _list_demos(self, file: h5py.File) -> List[str]:
        return list(file["data"].keys())

    def _get_demo_group(self, file: h5py.File, demo: str) -> h5py.Group:
        return file["data"][demo]

    def _get_obs_group(self, demo_group: h5py.Group) -> h5py.Group:
        return demo_group["obs"]

    def _get_actions_array(self, demo_group: h5py.Group) -> np.ndarray:
        # Default: prefer 'actions'
        if "actions" in demo_group:
            return np.array(demo_group["actions"])
        raise KeyError("actions dataset not found in demo group")

    def _get_rewards_array(self, demo_group: h5py.Group) -> Optional[np.ndarray]:
        return np.array(demo_group["rewards"]) if "rewards" in demo_group else None

    def _get_dones_array(self, demo_group: h5py.Group) -> Optional[np.ndarray]:
        return np.array(demo_group["dones"]) if "dones" in demo_group else None

    def _get_file_prefix(self, file_basename: str, file_idx: int) -> str:
        return f"file_{file_idx}"

    # --------------------------
    # HDF5 helpers
    # --------------------------
    @contextmanager
    def _get_hdf5_file(self, h5_path: str):
        if h5_path not in self._hdf5_files:
            self._hdf5_files[h5_path] = h5py.File(h5_path, "r", swmr=self.hdf5_use_swmr, libver="latest")
        try:
            yield self._hdf5_files[h5_path]
        finally:
            pass

    def _get_all_obs_keys(self, h5_path: str) -> List[str]:
        with self._get_hdf5_file(h5_path) as file:
            demo_keys = self._list_demos(file)
            if not demo_keys:
                raise ValueError(f"No demos found in {h5_path}")
            first_demo = demo_keys[0]
            obs_group = self._get_obs_group(self._get_demo_group(file, first_demo))
            obs_keys = list(obs_group.keys())
        return obs_keys

    # --------------------------
    # Zero obs factory
    # --------------------------
    def _create_zero_observation(self, obs_key: str):
        print(f"Creating zero observation for {obs_key}")
        raise Exception("Creating zero observation for in h5")
        if any(img_kw in obs_key.lower() for img_kw in ["image", "rgb", "camera", "cam"]):
            return np.zeros((128, 128, 3), dtype=np.uint8)
        return np.zeros((1,), dtype=np.float32)

    # --------------------------
    # Image cache helpers
    # --------------------------
    def _add_to_image_cache(self, cache_key: str, data: np.ndarray):
        while len(self.image_lru_cache) >= self.max_image_cache_size:
            self.image_lru_cache.popitem(last=False)
        self.image_lru_cache[cache_key] = data

    def _load_obs_from_file(self, h5_path: str, demo: str, obs_key: str, timestep: int):
        cache_key = f"{h5_path}::{demo}::{obs_key}::{timestep}"
        if cache_key in self.image_lru_cache:
            value = self.image_lru_cache.pop(cache_key)
            self.image_lru_cache[cache_key] = value
            return value
        with self._get_hdf5_file(h5_path) as file:
            data = self._get_obs_group(self._get_demo_group(file, demo))[obs_key][timestep]
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self._add_to_image_cache(cache_key, data)
        return data

    def _batch_load_images(self, image_requests: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        results = {}
        for request in image_requests:
            try:
                image_data = self._load_obs_from_file(
                    request["h5_path"], request["demo"], request["key"], request["timestep"]
                )
                results[request["request_id"]] = image_data
            except Exception:
                results[request["request_id"]] = self._create_zero_observation(request["key"])
        return results

    # --------------------------
    # Loading and caching
    # --------------------------
    def _load_with_optimizations(self):
        if self.obs_keys is None:
            self.obs_keys = self._get_all_obs_keys(self.h5_paths[0])

        # Determine modalities by heuristic
        low_dim_keys: List[str] = []
        rgb_keys: List[str] = []
        for key in self.obs_keys:
            if any(img_kw in key.lower() for img_kw in ["image", "rgb", "camera", "cam"]):
                rgb_keys.append(key)
            else:
                low_dim_keys.append(key)
        self.image_keys = rgb_keys
        self.low_dim_keys = low_dim_keys

        if self.hdf5_cache_mode in ["all", "low_dim"]:
            self._load_with_memory_cache()
        else:
            self._convert_to_transitions()

        self._print_dataset_statistics()
        self.image_loading_executor = None

    def _load_with_memory_cache(self):
        all_cached_data: Dict[str, Dict[str, Any]] = {}
        self._demo_data_lengths = {}

        for file_idx, h5_path in enumerate(self.h5_paths):
            with self._get_hdf5_file(h5_path) as file:
                demos = self._list_demos(file)
                file_basename = os.path.basename(h5_path)
                file_prefix = self._get_file_prefix(file_basename, file_idx)

                for demo in demos:
                    unique_episode_id = f"{file_prefix}_{demo}"
                    demo_key = f"{h5_path}::{unique_episode_id}"
                    group = self._get_demo_group(file, demo)

                    actions = self._get_actions_array(group)
                    actual_data_length = len(actions)
                    self._demo_data_lengths[demo_key] = {
                        "actions_length": actual_data_length,
                        "obs_lengths": {},
                        "original_demo_name": demo,
                    }

                    cached_demo: Dict[str, Any] = {
                        "actions": np.array(actions[:actual_data_length]),
                        "obs": {},
                        "attrs": {"num_samples": actual_data_length},
                        "original_demo_name": demo,
                    }

                    obs_group = self._get_obs_group(group)
                    for obs_key in obs_group.keys():
                        if obs_key in (self.remove_obs_keys or []):
                            continue
                        self._demo_data_lengths[demo_key]["obs_lengths"][obs_key] = len(obs_group[obs_key])
                    # Cache only low-dim keys
                    for key in self.low_dim_keys:
                        if key in obs_group and key not in (self.remove_obs_keys or []):
                            obs_data = obs_group[key]
                            obs_length = min(len(obs_data), actual_data_length)
                            cached_demo["obs"][key] = np.array(obs_data[:obs_length])

                    rewards = self._get_rewards_array(group)
                    if rewards is not None:
                        cached_demo["rewards"] = np.array(rewards[:actual_data_length])

                    dones = self._get_dones_array(group)
                    if dones is not None:
                        cached_demo["dones"] = np.array(dones[:actual_data_length])

                    all_cached_data[demo_key] = cached_demo

        self.hdf5_cache = all_cached_data
        self._build_lightweight_index_if_needed()

    def _convert_to_transitions(self):
        # Fallback path that materializes transitions immediately (not used by default)
        self.transitions = []
        for file_idx, h5_path in enumerate(self.h5_paths):
            with self._get_hdf5_file(h5_path) as file:
                demos = self._list_demos(file)
                file_basename = os.path.basename(h5_path)
                file_prefix = self._get_file_prefix(file_basename, file_idx)

                for demo in demos:
                    unique_episode_id = f"{file_prefix}_{demo}"
                    demo_key = f"{h5_path}::{unique_episode_id}"
                    group = self._get_demo_group(file, demo)
                    actions = self._get_actions_array(group)
                    obs_group = self._get_obs_group(group)

                    n_steps = len(actions)
                    rewards = self._get_rewards_array(group)
                    if rewards is None:
                        rewards = np.zeros(n_steps, dtype=np.float32)
                    dones = self._get_dones_array(group)
                    if dones is None:
                        dones = np.zeros(n_steps, dtype=np.float32)
                        if n_steps > 0:
                            dones[-1] = 1

                    done_indices = np.where(dones)[0]
                    actual_episode_length = n_steps
                    if len(done_indices) > 0:
                        actual_episode_length = int(done_indices[0] + 1)

                    obs_keys = list(obs_group.keys())
                    if self.remove_obs_keys:
                        obs_keys = [k for k in obs_keys if k not in self.remove_obs_keys]

                    obs_list: List[Dict[str, Any]] = []
                    for t in range(actual_episode_length):
                        od = {k: obs_group[k][t] for k in obs_keys}
                        obs_list.append(od)

                    for t in range(actual_episode_length):
                        action = actions[t]
                        if self.min_action is not None and self.max_action is not None:
                            action = (action - self.min_action) / (self.max_action - self.min_action)
                            action = 2 * action - 1
                        next_obs = obs_list[t] if t == actual_episode_length - 1 else obs_list[t + 1]
                        self.transitions.append(
                            Transition(
                                obs=obs_list[t],
                                action=action,
                                reward=rewards[t],
                                next_obs=next_obs,
                                done=dones[t],
                                truncated=False,
                                episode_id=unique_episode_id,
                                step_in_episode=t,
                                max_steps_in_episode=actual_episode_length,
                                timestamp=None,
                                language_instruction=self._demo_id_to_demo_lang_str.get(demo_key, None),
                            )
                        )

    # --------------------------
    # Index and sampling
    # --------------------------
    def _build_lightweight_index_if_needed(self):
        if getattr(self, "_index", None) and len(self._index) > 0:
            return
        self._index = []
        self._episode_boundaries_index = {}
        running = 0
        for demo_key, cached_demo in (self.hdf5_cache or {}).items():
            h5_path, unique_episode_id = demo_key.split("::")
            original_demo_name = cached_demo.get("original_demo_name", unique_episode_id.split("_", 1)[-1])
            n_steps = len(cached_demo.get("actions", []))
            actual_len = n_steps
            dones = cached_demo.get("dones", None)
            if dones is not None and len(dones) > 0:
                done_idx = np.where(dones)[0]
                if len(done_idx) > 0:
                    actual_len = int(done_idx[0] + 1)
            start = running
            for t in range(actual_len):
                self._index.append(
                    {
                        "demo_key": demo_key,
                        "h5_path": h5_path,
                        "original_demo_name": original_demo_name,
                        "episode_id": unique_episode_id,
                        "t": t,
                        "is_last": (t == actual_len - 1),
                        "episode_len": actual_len,
                    }
                )
                running += 1
            self._episode_boundaries_index[unique_episode_id] = (start, running - 1)

    def sample_indices(self, batch_size: int, sampler: "BaseSampler" = None, **kwargs) -> np.ndarray:
        if self.is_empty():
            return np.zeros((0,), dtype=np.int64)
        return np.random.randint(0, len(self._index), size=(batch_size,), dtype=np.int64)

    def _construct_transitions_from_indices(self, indices: np.ndarray) -> List[Transition]:
        meta = [self._index[int(i)] for i in indices]
        transitions: List[Transition] = []
        image_reqs: List[Dict[str, Any]] = []
        req_map: Dict[str, Tuple[int, str, bool]] = {}

        for i, m in enumerate(meta):
            demo_key = m["demo_key"]
            cached_demo = self.hdf5_cache[demo_key]
            t = m["t"]
            actual_len = m["episode_len"]
            is_last = m["is_last"]
            actions = cached_demo["actions"]
            rewards = cached_demo.get("rewards", None)
            dones = cached_demo.get("dones", None)

            obs: Dict[str, Any] = {}
            next_obs: Dict[str, Any] = {}
            for key in set(self.low_dim_keys) - set(self.remove_obs_keys):
                if key in cached_demo["obs"]:
                    arr = cached_demo["obs"][key]
                    if t < len(arr):
                        obs[key] = arr[t]
                    if not is_last and (t + 1) < len(arr):
                        next_obs[key] = arr[t + 1]

            for key in set(self.image_keys) - set(self.remove_obs_keys):
                image_reqs.append(
                    {
                        "request_id": f"{i}:{key}:obs",
                        "h5_path": m["h5_path"],
                        "demo": m["original_demo_name"],
                        "key": key,
                        "timestep": t,
                    }
                )
                req_map[f"{i}:{key}:obs"] = (i, key, False)
                if not is_last:
                    image_reqs.append(
                        {
                            "request_id": f"{i}:{key}:next",
                            "h5_path": m["h5_path"],
                            "demo": m["original_demo_name"],
                            "key": key,
                            "timestep": t + 1,
                        }
                    )
                    req_map[f"{i}:{key}:next"] = (i, key, True)

            action = actions[t]
            if self.min_action is not None and self.max_action is not None:
                action = (action - self.min_action) / (self.max_action - self.min_action)
                action = 2 * action - 1
            reward = rewards[t] if rewards is not None and t < len(rewards) else 0.0
            done = dones[t] if dones is not None and t < len(dones) else (t == actual_len - 1)
            language_instruction = self._demo_id_to_demo_lang_str.get(demo_key, None)
            transitions.append(
                Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs if not is_last else obs,
                    done=done,
                    truncated=False,
                    episode_id=m["episode_id"],
                    step_in_episode=t,
                    max_steps_in_episode=actual_len,
                    timestamp=None,
                    language_instruction=language_instruction,
                )
            )

        if image_reqs:
            results = self._batch_load_images(image_reqs)
            for rid, img in results.items():
                ti, key, is_next = req_map[rid]
                if is_next:
                    if transitions[ti].next_obs is transitions[ti].obs:
                        transitions[ti].next_obs = transitions[ti].obs.copy()
                    transitions[ti].next_obs[key] = img
                else:
                    transitions[ti].obs[key] = img
        return transitions

    def get_contiguous_chunks_optimized(
        self, chunk_size: int, max_chunks: int, obs_as_sequence: bool = True
    ) -> List[List[Transition]]:
        if self.is_empty() or chunk_size <= 0 or max_chunks <= 0:
            return []
        if not hasattr(self, "_valid_chunk_starts") or getattr(self, "_chunk_size_cache", None) != chunk_size:
            valid_starts = []
            for episode_id, (start, end) in self._episode_boundaries_index.items():
                episode_len = end - start + 1
                if episode_len >= chunk_size:
                    valid_starts.extend(range(start, end - chunk_size + 2))
            self._valid_chunk_starts = valid_starts
            self._chunk_size_cache = chunk_size

        valid_starts = self._valid_chunk_starts
        if not valid_starts:
            return []
        import random as _random

        if max_chunks > len(valid_starts):
            # Sample with replacement if we request more chunks than available
            starts = _random.choices(valid_starts, k=max_chunks)
        else:
            # Sample without replacement
            starts = _random.sample(valid_starts, max_chunks)

        chunk_metas: List[List[dict]] = []
        chunks: List[List[Transition]] = []
        image_requests: Dict[Tuple[str, str, str], set] = {}
        assign_list: List[Tuple[int, int, str, bool, int]] = []

        for ci, s in enumerate(starts):
            span_ids = [self._index[i] for i in range(s, s + chunk_size)]
            chunk_metas.append(span_ids)
            transitions: List[Transition] = []
            for pos, m in enumerate(span_ids):
                demo_key = m["demo_key"]
                cached_demo = self.hdf5_cache[demo_key]
                t = m["t"]
                actual_len = m["episode_len"]
                is_last = (pos == chunk_size - 1) or m["is_last"]

                obs: Dict[str, Any] = {}
                next_obs: Dict[str, Any] = {}
                for key in self.low_dim_keys:
                    if key in cached_demo["obs"]:
                        arr = cached_demo["obs"][key]
                        if t < len(arr):
                            obs[key] = arr[t]
                        if not is_last and (t + 1) < len(arr):
                            next_obs[key] = arr[t + 1]

                if obs_as_sequence:
                    needed_ts = [t, t + 1] if not is_last else [t]
                else:
                    if pos == 0:
                        needed_ts = [t]
                    elif pos == chunk_size - 1:
                        needed_ts = [t, t + 1] if not is_last else [t]
                    else:
                        needed_ts = []
                for key in set(self.image_keys) - set(self.remove_obs_keys):
                    g = (m["h5_path"], m["original_demo_name"], key)
                    for ts in needed_ts:
                        image_requests.setdefault(g, set()).add(ts)
                        assign_list.append((ci, pos, key, ts == t + 1, ts))

                actions = cached_demo["actions"]
                action = actions[t]
                if self.min_action is not None and self.max_action is not None:
                    action = (action - self.min_action) / (self.max_action - self.min_action)
                    action = 2 * action - 1
                rewards = cached_demo.get("rewards")
                reward = rewards[t] if rewards is not None and t < len(rewards) else 0.0
                dones = cached_demo.get("dones")
                done = dones[t] if dones is not None and t < len(dones) else (t == actual_len - 1)
                lang = self._demo_id_to_demo_lang_str.get(demo_key, None)
                transitions.append(
                    Transition(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs if not is_last else obs,
                        done=done,
                        truncated=False,
                        episode_id=m["episode_id"],
                        step_in_episode=t,
                        max_steps_in_episode=actual_len,
                        timestamp=None,
                        language_instruction=lang,
                    )
                )
            chunks.append(transitions)

        loaded_images: Dict[Tuple[str, str, str, int], np.ndarray] = {}
        for (h5_path, demo, key), timesteps in image_requests.items():
            if not timesteps:
                continue
            ts_sorted = sorted(timesteps)

            # First, check which timesteps are already in cache
            cache_hits = {}
            cache_misses = []
            for t in ts_sorted:
                cache_key = f"{h5_path}::{demo}::{key}::{t}"
                if cache_key in self.image_lru_cache:
                    # Move to end (most recent)
                    value = self.image_lru_cache.pop(cache_key)
                    self.image_lru_cache[cache_key] = value
                    cache_hits[t] = value
                    loaded_images[(h5_path, demo, key, t)] = value
                else:
                    cache_misses.append(t)

            # Only read from HDF5 for cache misses
            if cache_misses:
                try:
                    with self._get_hdf5_file(h5_path) as f:
                        dset = self._get_obs_group(self._get_demo_group(f, demo))[key]
                        try:
                            dset.id.set_chunk_cache(1024 * 8, 64 * 1024 * 1024, 0.75)
                        except Exception:
                            pass
                        start = cache_misses[0]
                        prev = start
                        for t in cache_misses[1:] + [None]:
                            if t is None or t != prev + 1:
                                data = dset[start : prev + 1]
                                for i, tt in enumerate(range(start, prev + 1)):
                                    img = np.asarray(data[i])
                                    loaded_images[(h5_path, demo, key, tt)] = img
                                    # Add to cache
                                    cache_key = f"{h5_path}::{demo}::{key}::{tt}"
                                    self._add_to_image_cache(cache_key, img)
                                start = t if t is not None else start
                            if t is None:
                                break
                            prev = t
                except Exception:
                    pass

        for ci, pos, key, is_next, t in assign_list:
            m = chunk_metas[ci][pos]
            h5_path = m["h5_path"]
            demo = m["original_demo_name"]
            img = loaded_images.get((h5_path, demo, key, t))

            if img is None:
                img = self._create_zero_observation(key)
            if is_next:
                if chunks[ci][pos].next_obs is chunks[ci][pos].obs:
                    chunks[ci][pos].next_obs = chunks[ci][pos].obs.copy()
                chunks[ci][pos].next_obs[key] = img
            else:
                chunks[ci][pos].obs[key] = img

        return chunks

    # --------------------------
    # Public API
    # --------------------------
    def __len__(self):
        return len(self._index)

    def size(self):
        return len(self._index)

    def is_empty(self):
        return len(self._index) == 0

    def sample(
        self, batch_size: int, sampler: "BaseSampler" = None, device: str = None, dtype=None, **kwargs
    ) -> Dict[str, Any]:
        if self.is_empty():
            return {}
        active_sampler = sampler or self.sampler
        if hasattr(active_sampler, "chunk_size") and hasattr(active_sampler, "_chunk_to_sequence"):
            # Use efficient batch-based chunked sampling
            return self._sample_chunked_batch_efficient(
                batch_size,
                chunk_size=active_sampler.chunk_size,
                obs_as_sequence=active_sampler.obs_as_sequence,
                device=device,
                dtype=dtype,
            )
        return self.sample_batch(batch_size, sampler=active_sampler, device=device, dtype=dtype)

    def _sample_chunked_batch_efficient(
        self, batch_size: int, chunk_size: int, obs_as_sequence: bool, device=None, dtype=None
    ) -> Dict[str, Any]:
        """
        Efficient chunked sampling that uses batch operations like sample_batch.

        Key idea: Sample chunk start indices, expand to all indices in chunks,
        then use batch_from_indices for efficient data loading.
        """
        if self.is_empty():
            return {}

        # Build valid chunk starts if needed (same as before)
        if not hasattr(self, "_valid_chunk_starts") or getattr(self, "_chunk_size_cache", None) != chunk_size:
            valid_starts = []
            for episode_id, (start, end) in self._episode_boundaries_index.items():
                episode_len = end - start + 1
                if episode_len >= chunk_size:
                    valid_starts.extend(range(start, end - chunk_size + 2))
            self._valid_chunk_starts = valid_starts
            self._chunk_size_cache = chunk_size

        if not self._valid_chunk_starts:
            return {}

        # Sample chunk start indices
        num_chunks = min(len(self._valid_chunk_starts), batch_size)
        import random as _random

        chunk_starts = _random.sample(self._valid_chunk_starts, num_chunks)

        # Expand each start to a chunk: [[start, start+1, ..., start+chunk_size-1], ...]
        all_indices = []
        for start in chunk_starts:
            all_indices.extend(range(start, start + chunk_size))

        # Use the efficient batch_from_indices path (same as sample_batch!)
        flat_batch = self.batch_from_indices(np.array(all_indices), device=device, dtype=dtype)

        # Reshape from [B*T, ...] to [B, T, ...] format
        B = num_chunks
        T = chunk_size

        # Reshape actions: [B*T, action_dim] -> [B, T, action_dim]
        actions = flat_batch["action"]
        if isinstance(actions, torch.Tensor):
            action_shape = actions.shape
            actions = actions.view(B, T, *action_shape[1:])
        else:
            actions = np.array(actions).reshape(B, T, -1)
            actions = torch.from_numpy(actions).float() if device or dtype else actions
        # Get gamma from sampler if available (default to 0.99)
        gamma = self.sampler.gamma

        # For chunked sampling, we aggregate rewards/dones across the chunk
        rewards = flat_batch["reward"]
        dones = flat_batch["done"]
        if isinstance(rewards, torch.Tensor):
            rewards_reshaped = rewards.view(B, T)
            dones_reshaped = dones.view(B, T)

            # Apply discount factors: r_t + gamma*r_{t+1} + ... + gamma^{T-1}*r_{t+T-1}
            discount_factors = torch.tensor(
                [gamma**i for i in range(T)], dtype=rewards_reshaped.dtype, device=rewards_reshaped.device
            ).unsqueeze(0)  # Shape: [1, T]
            reward = (rewards_reshaped * discount_factors).sum(dim=1)  # Discounted sum
            done = dones_reshaped.any(dim=1)  # Any done in chunk
        else:
            rewards_reshaped = np.array(rewards).reshape(B, T)
            dones_reshaped = np.array(dones).reshape(B, T)

            # Apply discount factors for numpy path
            discount_factors = np.array([gamma**i for i in range(T)], dtype=np.float32).reshape(1, T)
            reward = torch.from_numpy((rewards_reshaped * discount_factors).sum(axis=1)).float()
            done = torch.from_numpy(dones_reshaped.any(axis=1))

        # For obs/next_obs, take first and last of each chunk
        obs_dict = {}
        next_obs_dict = {}
        for key, values in flat_batch["obs"].items():
            if isinstance(values, torch.Tensor):
                values_reshaped = values.view(B, T, *values.shape[1:])
                obs_dict[key] = values_reshaped[:, 0, ...]  # First timestep
            else:
                # Handle list or non-stackable case
                obs_dict[key] = values[::T] if len(values) == B * T else values

        for key, values in flat_batch["next_obs"].items():
            if isinstance(values, torch.Tensor):
                values_reshaped = values.view(B, T, *values.shape[1:])
                next_obs_dict[key] = values_reshaped[:, -1, ...]  # Last timestep
            else:
                # Handle list or non-stackable case
                next_obs_dict[key] = values[T - 1 :: T] if len(values) == B * T else values

        return {
            "obs": obs_dict,
            "action": actions,
            "reward": reward,
            "next_obs": next_obs_dict,
            "done": done,
            "truncated": torch.zeros_like(done),
            "info": [{} for _ in range(B)],
        }

    def _sample_chunked_batch_fast(self, batch_size: int, chunk_size: int, obs_as_sequence: bool) -> Dict[str, Any]:
        import time
        import sys

        print(
            f"\n=== _sample_chunked_batch_fast CALLED: batch_size={batch_size}, chunk_size={chunk_size} ===",
            file=sys.stderr,
            flush=True,
        )
        _t0_total = time.perf_counter()
        if self.is_empty():
            return {}

        _t0_chunks = time.perf_counter()
        chunks = self.get_contiguous_chunks_optimized(chunk_size, batch_size, obs_as_sequence=obs_as_sequence)
        _t1_chunks = time.perf_counter()
        if not chunks:
            return {}
        _t0_process = time.perf_counter()
        B = len(chunks)
        T = chunk_size
        actions_list = []
        rewards_list = []
        dones_list = []
        truncated_list = []
        obs_acc: Dict[str, List[torch.Tensor]] = {}
        next_obs_acc: Dict[str, List[torch.Tensor]] = {}

        sample_tr = chunks[0][0]
        obs_keys_all = set(sample_tr.obs.keys())
        for k in sample_tr.next_obs.keys():
            obs_keys_all.add(k)
        for k in obs_keys_all:
            obs_acc[k] = []
            next_obs_acc[k] = []

        # Optimization: Collect numpy arrays first, convert to tensor at the end
        # This is much faster than per-element tensor conversions
        for ch in chunks:
            # Collect actions as numpy array
            a_seq = np.stack([t.action if isinstance(t.action, np.ndarray) else np.array(t.action) for t in ch], axis=0)
            actions_list.append(a_seq)

            r_seq = [t.reward for t in ch]
            rewards_list.append(float(np.sum(r_seq)))
            dones_list.append(bool(np.any([t.done for t in ch])))
            truncated_list.append(bool(np.any([t.truncated for t in ch])))

            first = ch[0]
            last = ch[-1]
            for k in obs_keys_all:
                ov = first.obs.get(k, None)
                nv = last.next_obs.get(k, None)
                # Prefer duplicating counterpart over zeros
                if ov is None and nv is not None:
                    ov = nv
                if nv is None and ov is not None:
                    nv = ov
                if ov is None:
                    ov = self._create_zero_observation(k)
                if nv is None:
                    nv = self._create_zero_observation(k)
                if isinstance(ov, np.ndarray) and ov.dtype == np.uint8:
                    ov = ov.astype(np.float32)
                if isinstance(nv, np.ndarray) and nv.dtype == np.uint8:
                    nv = nv.astype(np.float32)
                # Keep as numpy arrays for now
                obs_acc[k].append(ov)
                next_obs_acc[k].append(nv)

        # Convert all numpy arrays to tensors at once (much faster)
        actions = torch.from_numpy(np.stack(actions_list, axis=0)).float()
        reward = torch.as_tensor(rewards_list, dtype=torch.float32)
        done = torch.as_tensor(dones_list, dtype=torch.bool)
        truncated = torch.as_tensor(truncated_list, dtype=torch.bool)

        obs: Dict[str, Any] = {}
        next_obs: Dict[str, Any] = {}
        for k in obs_keys_all:
            items = obs_acc[k]
            if len(items) > 0:
                # Check if all items are numpy arrays with same shape
                if all(isinstance(i, np.ndarray) for i in items) and all(i.shape == items[0].shape for i in items):
                    # Stack numpy arrays and convert to tensor at once
                    obs[k] = torch.from_numpy(np.stack(items, axis=0)).float()
                elif all(isinstance(i, np.ndarray) or isinstance(i, torch.Tensor) for i in items):
                    # Convert any remaining numpy to tensors, then stack
                    items_t = [torch.as_tensor(i) if not torch.is_tensor(i) else i for i in items]
                    if all(tuple(i.shape) == tuple(items_t[0].shape) for i in items_t):
                        obs[k] = torch.stack(items_t, dim=0)
                    else:
                        obs[k] = items_t
                else:
                    obs[k] = items

            items_n = next_obs_acc[k]
            if len(items_n) > 0:
                if all(isinstance(i, np.ndarray) for i in items_n) and all(
                    i.shape == items_n[0].shape for i in items_n
                ):
                    next_obs[k] = torch.from_numpy(np.stack(items_n, axis=0)).float()
                elif all(isinstance(i, np.ndarray) or isinstance(i, torch.Tensor) for i in items_n):
                    items_nt = [torch.as_tensor(i) if not torch.is_tensor(i) else i for i in items_n]
                    if all(tuple(i.shape) == tuple(items_nt[0].shape) for i in items_nt):
                        next_obs[k] = torch.stack(items_nt, dim=0)
                    else:
                        next_obs[k] = items_nt
                else:
                    next_obs[k] = items_n

        result = {
            "obs": self.remap_obs(obs),
            "action": actions,
            "reward": reward,
            "next_obs": self.remap_obs(next_obs),
            "done": done,
            "truncated": truncated,
            "info": [{} for _ in range(B)],
        }

        # Track detailed timing stats
        _t1_total = time.perf_counter()
        if not hasattr(self, "_chunked_batch_stats"):
            self._chunked_batch_stats = {"calls": 0, "total_time": 0, "chunk_time": 0, "process_time": 0}
        self._chunked_batch_stats["calls"] += 1
        self._chunked_batch_stats["total_time"] += _t1_total - _t0_total
        self._chunked_batch_stats["chunk_time"] += _t1_chunks - _t0_chunks
        self._chunked_batch_stats["process_time"] += _t1_total - _t0_process

        if self._chunked_batch_stats["calls"] <= 20:  # First 20 calls
            this_total = (_t1_total - _t0_total) * 1000
            this_chunk = (_t1_chunks - _t0_chunks) * 1000
            this_process = (_t1_total - _t0_process) * 1000
            print(
                f"[CHUNKED_BATCH #{self._chunked_batch_stats['calls']}] "
                f"Total: {this_total:.1f}ms, "
                f"Chunk: {this_chunk:.1f}ms, "
                f"Process: {this_process:.1f}ms"
            )

        return result

    def remap_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # This is for other functions to override
        return obs

    def batch_from_indices(self, indices: np.ndarray, device: str = None, dtype=None) -> Dict[str, Any]:
        if indices is None or len(indices) == 0:
            return {}
        sampled_transitions = self._construct_transitions_from_indices(indices)
        if self.post_transforms:
            out = []
            for t in sampled_transitions:
                for tf in self.post_transforms:
                    t = tf(t)
                out.append(t)
            sampled_transitions = out
        batched = self._batch_transitions(sampled_transitions)
        if batched and (device is not None or dtype is not None):
            batched = self._convert_batch_to_tensors(batched, device, dtype)
        return batched

    def transitions_from_indices(self, indices: np.ndarray) -> List[Transition]:
        if indices is None or len(indices) == 0:
            return []
        return self._construct_transitions_from_indices(indices)

    def sample_batch(
        self, batch_size: int, sampler: "BaseSampler" = None, device: str = None, dtype=None
    ) -> Dict[str, Any]:
        if self.is_empty():
            return {}
        active_sampler = sampler or self.sampler
        indices = None
        if hasattr(active_sampler, "sample_indices"):
            try:
                indices = active_sampler.sample_indices(self, batch_size)
            except Exception:
                indices = None
        if indices is None or len(indices) == 0:
            indices = self.sample_indices(batch_size)
        batch = self.batch_from_indices(indices, device=device, dtype=dtype)
        return batch

    def _batch_transitions(self, transitions: List[Transition]) -> Dict[str, Any]:
        if not transitions:
            return {}
        batched: Dict[str, Any] = {
            "obs": {},
            "action": [],
            "reward": [],
            "next_obs": {},
            "done": [],
            "truncated": [],
            "episode_id": [],
            "step_in_episode": [],
            "timestamp": [],
        }
        all_obs_keys = set()
        for tr in transitions:
            all_obs_keys.update(tr.obs.keys())
            all_obs_keys.update(tr.next_obs.keys())
        for k in all_obs_keys:
            batched["obs"][k] = []
            batched["next_obs"][k] = []
        for tr in transitions:
            for k in all_obs_keys:
                ov = tr.obs.get(k)
                nv = tr.next_obs.get(k)
                # Prefer duplicating counterpart over zeros to avoid mismatched sizes
                if ov is None and nv is not None:
                    ov = nv
                if nv is None and ov is not None:
                    nv = ov
                if ov is None:
                    ov = self._create_zero_observation(k)
                if nv is None:
                    nv = self._create_zero_observation(k)
                if hasattr(ov, "cpu"):
                    ov = ov.cpu().numpy()
                if hasattr(nv, "cpu"):
                    nv = nv.cpu().numpy()
                batched["obs"][k].append(ov)
                batched["next_obs"][k].append(nv)

            action = tr.action
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            batched["action"].append(action)
            batched["reward"].append(tr.reward)
            batched["done"].append(tr.done)
            batched["truncated"].append(tr.truncated)
            batched["episode_id"].append(tr.episode_id)
            batched["step_in_episode"].append(tr.step_in_episode)
            batched["timestamp"].append(tr.timestamp)

        for k in all_obs_keys:
            try:
                obs_items = batched["obs"][k]
                next_items = batched["next_obs"][k]
                if obs_items and all(
                    isinstance(item, np.ndarray) and item.shape == obs_items[0].shape for item in obs_items
                ):
                    batched["obs"][k] = np.stack(obs_items)
                else:
                    batched["obs"][k] = np.array(obs_items, dtype=object)
                if next_items and all(
                    isinstance(item, np.ndarray) and item.shape == next_items[0].shape for item in next_items
                ):
                    batched["next_obs"][k] = np.stack(next_items)
                else:
                    batched["next_obs"][k] = np.array(next_items, dtype=object)
            except Exception:
                batched["obs"][k] = np.array(batched["obs"][k], dtype=object)
                batched["next_obs"][k] = np.array(batched["next_obs"][k], dtype=object)

        try:
            action_items = batched["action"]
            if action_items and all(
                isinstance(item, np.ndarray) and item.shape == action_items[0].shape for item in action_items
            ):
                batched["action"] = np.stack(action_items)
            else:
                batched["action"] = np.array(action_items, dtype=object)
        except Exception:
            batched["action"] = np.array(batched["action"], dtype=object)

        batched["reward"] = np.array(batched["reward"])
        batched["done"] = np.array(batched["done"])
        batched["truncated"] = np.array(batched["truncated"])
        batched["episode_id"] = np.array(batched["episode_id"])
        batched["step_in_episode"] = np.array(batched["step_in_episode"])
        batched["timestamp"] = np.array(batched["timestamp"])

        return batched

    def _convert_batch_to_tensors(self, batch: Dict[str, Any], device: str = None, dtype=None) -> Dict[str, Any]:
        if not batch:
            return {}
        if dtype is None:
            dtype = torch.float32
        tensor_batch: Dict[str, Any] = {}
        tensor_batch["obs"] = {}
        for key, values in batch["obs"].items():
            if values is None:
                continue
            try:
                if isinstance(values, np.ndarray) and values.dtype == object:
                    tensor_list = []
                    for item in values:
                        if item is None:
                            zero_obs = self._create_zero_observation(key)
                            tensor_list.append(torch.tensor(zero_obs, device=device, dtype=dtype))
                        else:
                            if isinstance(item, torch.Tensor):
                                t = item
                                t = t.to(dtype=dtype)
                                if device is not None:
                                    t = t.to(device)
                                tensor_list.append(t)
                            else:
                                if isinstance(item, np.ndarray):
                                    if item.dtype == np.uint8:
                                        item = item.astype(np.float32)
                                    elif item.dtype != np.float32:
                                        item = item.astype(np.float32)
                                tensor_list.append(torch.tensor(item, device=device, dtype=dtype))
                    try:
                        tensor_batch["obs"][key] = torch.stack(tensor_list)
                    except RuntimeError:
                        tensor_batch["obs"][key] = tensor_list
                else:
                    if isinstance(values, torch.Tensor):
                        t = values.to(dtype=dtype)
                        if device is not None:
                            t = t.to(device)
                        tensor_batch["obs"][key] = t
                    else:
                        if isinstance(values, np.ndarray):
                            if values.dtype == np.uint8:
                                values = values.astype(np.float32)
                            elif values.dtype != np.float32:
                                values = values.astype(np.float32)
                        tensor_batch["obs"][key] = torch.tensor(values, device=device, dtype=dtype)
            except Exception:
                zero_obs = self._create_zero_observation(key)
                batch_size = len(batch["action"]) if "action" in batch else 1
                zero_batch = np.stack([zero_obs] * batch_size).astype(np.float32)
                tensor_batch["obs"][key] = torch.tensor(zero_batch, device=device, dtype=dtype)

        tensor_batch["next_obs"] = {}
        for key, values in batch["next_obs"].items():
            if values is None:
                continue
            try:
                if isinstance(values, np.ndarray) and values.dtype == object:
                    tensor_list = []
                    for item in values:
                        if item is None:
                            zero_obs = self._create_zero_observation(key)
                            tensor_list.append(torch.tensor(zero_obs, device=device, dtype=dtype))
                        else:
                            if isinstance(item, torch.Tensor):
                                t = item
                                t = t.to(dtype=dtype)
                                if device is not None:
                                    t = t.to(device)
                                tensor_list.append(t)
                            else:
                                if isinstance(item, np.ndarray):
                                    if item.dtype == np.uint8:
                                        item = item.astype(np.float32)
                                    elif item.dtype != np.float32:
                                        item = item.astype(np.float32)
                                tensor_list.append(torch.tensor(item, device=device, dtype=dtype))
                    try:
                        tensor_batch["next_obs"][key] = torch.stack(tensor_list)
                    except RuntimeError:
                        tensor_batch["next_obs"][key] = tensor_list
                else:
                    if isinstance(values, torch.Tensor):
                        t = values.to(dtype=dtype)
                        if device is not None:
                            t = t.to(device)
                        tensor_batch["next_obs"][key] = t
                    else:
                        if isinstance(values, np.ndarray):
                            if values.dtype == np.uint8:
                                values = values.astype(np.float32)
                            elif values.dtype != np.float32:
                                values = values.astype(np.float32)
                        tensor_batch["next_obs"][key] = torch.tensor(values, device=device, dtype=dtype)
            except Exception:
                zero_obs = self._create_zero_observation(key)
                batch_size = len(batch["action"]) if "action" in batch else 1
                zero_batch = np.stack([zero_obs] * batch_size).astype(np.float32)
                tensor_batch["next_obs"][key] = torch.tensor(zero_batch, device=device, dtype=dtype)

        try:
            if isinstance(batch["action"], np.ndarray) and batch["action"].dtype == object:
                action_list = []
                for item in batch["action"]:
                    if item is None:
                        item = np.zeros_like(batch["action"][0]) if len(batch["action"]) > 0 else np.zeros(7)
                    if isinstance(item, np.ndarray) and item.dtype != np.float32:
                        item = item.astype(np.float32)
                    action_list.append(torch.tensor(item, device=device, dtype=dtype))
                try:
                    tensor_batch["action"] = torch.stack(action_list)
                except RuntimeError:
                    tensor_batch["action"] = action_list
            else:
                actions = batch["action"]
                if isinstance(actions, torch.Tensor):
                    t = actions.to(dtype=dtype)
                    if device is not None:
                        t = t.to(device)
                    tensor_batch["action"] = t
                else:
                    if isinstance(actions, np.ndarray) and actions.dtype != np.float32:
                        actions = actions.astype(np.float32)
                    tensor_batch["action"] = torch.tensor(actions, device=device, dtype=dtype)
        except Exception:
            batch_size = len(batch.get("reward", [])) if "reward" in batch else 1
            action_dim = 7
            zero_actions = np.zeros((batch_size, action_dim), dtype=np.float32)
            tensor_batch["action"] = torch.tensor(zero_actions, device=device, dtype=dtype)

        for key in ["reward", "done", "truncated", "step_in_episode"]:
            if key in batch:
                values = batch[key]
                if key in ["reward"]:
                    target_dtype = torch.float32
                elif key in ["done", "truncated"]:
                    target_dtype = torch.bool
                elif key in ["step_in_episode"]:
                    target_dtype = torch.long
                else:
                    target_dtype = dtype
                if isinstance(values, torch.Tensor):
                    t = values.to(dtype=target_dtype)
                    if device is not None:
                        t = t.to(device)
                    tensor_batch[key] = t
                else:
                    if isinstance(values, np.ndarray):
                        if key in ["reward"] and values.dtype != np.float32:
                            values = values.astype(np.float32)
                        elif key in ["done", "truncated"] and values.dtype != np.bool_:
                            values = values.astype(np.bool_)
                        elif key in ["step_in_episode"] and values.dtype != np.int64:
                            values = values.astype(np.int64)
                    tensor_batch[key] = torch.tensor(values, device=device, dtype=target_dtype)

        for key in ["episode_id", "timestamp"]:
            if key in batch:
                tensor_batch[key] = batch[key]

        return tensor_batch

    def get_episode_boundaries(self) -> Dict[Any, Tuple[int, int]]:
        return dict(self._episode_boundaries_index)

    def _print_dataset_statistics(self):
        if self.is_empty():
            print("📊 Dataset is empty")
            return
        episode_boundaries = self.get_episode_boundaries()
        if not episode_boundaries:
            print("📊 No episodes found in dataset")
            return
        episode_lengths = []
        for episode_id, (start_idx, end_idx) in episode_boundaries.items():
            episode_length = end_idx - start_idx + 1
            episode_lengths.append(episode_length)
        total_transitions = len(self)
        num_episodes = len(episode_lengths)
        avg_episode_length = sum(episode_lengths) / len(episode_lengths)
        min_episode_length = min(episode_lengths)
        max_episode_length = max(episode_lengths)
        print(f"\n📊 DATASET STATISTICS:")
        print(f"  Total transitions: {total_transitions:,}")
        print(f"  Number of episodes: {num_episodes:,}")
        print(f"  Average episode length: {avg_episode_length:.1f} steps")
        print(f"  Min episode length: {min_episode_length} steps")
        print(f"  Max episode length: {max_episode_length} steps")
        if len(self.h5_paths) > 1:
            print(f"  Loaded from {len(self.h5_paths)} dataset files")
        if hasattr(self, "image_keys") and hasattr(self, "low_dim_keys"):
            print(f"  Image observation keys: {len(self.image_keys)}")
            print(f"  Low-dim observation keys: {len(self.low_dim_keys)}")

    def __del__(self):
        if hasattr(self, "_hdf5_files"):
            for fh in self._hdf5_files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._hdf5_files.clear()
        if hasattr(self, "image_loading_executor") and self.image_loading_executor:
            try:
                self.image_loading_executor.shutdown(wait=True)
            except Exception:
                pass

    # --------------------------
    # Abstracts from BaseReplayBuffer
    # --------------------------
    def _add(self, *args, **kwargs):
        raise NotImplementedError("Cannot add to an offline buffer loaded from HDF5.")

    def clear(self):
        try:
            if self.transitions is not None:
                self.transitions.clear()
        except Exception:
            pass
        try:
            if self.hdf5_cache is not None:
                self.hdf5_cache.clear()
        except Exception:
            pass
        try:
            if hasattr(self, "image_lru_cache") and self.image_lru_cache is not None:
                self.image_lru_cache.clear()
        except Exception:
            pass

    def get_all_transitions(self) -> List[Transition]:
        return self.transitions if self.transitions is not None else []
