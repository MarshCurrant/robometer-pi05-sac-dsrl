# Copyright 2026 The RLinf Authors.

"""Small official-semantics RoboMeter client for LIBERO environments.

This intentionally mirrors ``robometer.evals.eval_utils.raw_dict_to_sample``
and ``post_batch_npy`` without importing the RoboMeter training package into
the environment worker process.
"""

from __future__ import annotations

import io
import itertools
import json
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import requests


@dataclass(frozen=True)
class RobometerBatchResult:
    progress: np.ndarray
    success_prob: np.ndarray


@dataclass(frozen=True)
class AsyncRobometerSampleResult:
    """One completed sample from a non-blocking pool submission."""

    token: str
    progress: float
    success_prob: float
    server_url: str


@dataclass(frozen=True)
class _AsyncRobometerAssignment:
    tokens: tuple[str, ...]
    server_url: str
    future: Future


def compose_robometer_reward(
    environment_reward: float | np.ndarray,
    estimated_reward: float | np.ndarray,
    *,
    add_estimated_reward: bool,
) -> float | np.ndarray:
    """Compose RoboMeter progress with the environment reward.

    RoboMeter's official LIBERO SAC recipe uses ``add_estimated_reward=true``:
    the absolute progress estimate is added to LIBERO's sparse ``-1/0`` reward.
    Keeping this operation explicit prevents asynchronous relabeling from
    silently changing the MDP by replacing the sparse reward.
    """
    if add_estimated_reward:
        return environment_reward + estimated_reward
    return estimated_reward


def official_subsample_and_pad(frames: np.ndarray, max_frames: int) -> np.ndarray:
    """Apply RoboMeter's uniform subsampling and right-padding semantics."""
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected frames shaped (T,H,W,C), got {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("RoboMeter cannot score an empty trajectory")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    if frames.shape[0] > max_frames:
        indices = np.linspace(0, frames.shape[0] - 1, max_frames).astype(int)
        frames = frames[indices]
    if frames.shape[0] < max_frames:
        padding = np.repeat(frames[-1:], max_frames - frames.shape[0], axis=0)
        frames = np.concatenate((frames, padding), axis=0)
    return np.ascontiguousarray(frames, dtype=np.uint8)


def terminal_progress_contexts(
    frames: np.ndarray,
    *,
    context_frames: int = 4,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Build the exact endpoint contexts for terminal mean progress delta."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"Expected non-empty (T,H,W,C) frames, got {frames.shape}")
    if context_frames <= 0:
        raise ValueError("context_frames must be positive")
    last = int(frames.shape[0]) - 1
    first_indices = np.zeros(int(context_frames), dtype=np.int64)
    final_indices = np.linspace(
        0, last, int(context_frames), dtype=np.int64
    )
    return (
        [
            np.ascontiguousarray(frames[first_indices], dtype=np.uint8),
            np.ascontiguousarray(frames[final_indices], dtype=np.uint8),
        ],
        [first_indices.tolist(), final_indices.tolist()],
    )


def success_window_detected(
    probabilities: Sequence[float],
    *,
    threshold: float,
    duration: int,
    rule: str,
) -> bool:
    """Evaluate the current success-head window using an explicit rule."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if len(probabilities) != duration:
        return False
    if rule == "all_consecutive":
        return all(float(value) > float(threshold) for value in probabilities)
    if rule == "majority_window":
        votes = sum(float(value) >= float(threshold) for value in probabilities)
        return votes > duration / 2
    raise ValueError(f"Unsupported success detection rule: {rule!r}")


class OfficialRobometerClient:
    """Batch client matching RoboMeter's official ``ProgressSample`` payload."""

    def __init__(
        self,
        server_url: str,
        *,
        max_frames: int = 8,
        timeout_s: float = 120.0,
        connection_retries: int = 2,
        strict_qwen3: bool = True,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.max_frames = int(max_frames)
        self.timeout_s = float(timeout_s)
        self.connection_retries = int(connection_retries)
        if self.connection_retries < 0:
            raise ValueError("connection_retries must be non-negative")
        self.session = requests.Session()
        # requests.Session is not documented as thread-safe. A client owns one
        # single-worker queue, while this lock also protects the rare synchronous
        # terminal request from overlapping that queue.
        self._request_lock = threading.Lock()
        self.model_info = self._load_model_info(strict_qwen3=strict_qwen3)

    def _load_model_info(self, *, strict_qwen3: bool) -> dict[str, Any]:
        response = self.session.get(
            f"{self.server_url}/model_info", timeout=self.timeout_s
        )
        response.raise_for_status()
        model_info = response.json()
        serialized = json.dumps(model_info, sort_keys=True).lower()
        if strict_qwen3 and "qwen3" not in serialized:
            raise RuntimeError(
                f"RoboMeter server {self.server_url} is not backed by Qwen3: "
                f"{serialized[:1000]}"
            )
        return model_info

    @staticmethod
    def _last_values(outputs: dict[str, Any], key: str, field: str) -> np.ndarray:
        container = outputs.get(key) or {}
        sequences = container.get(field) or []
        values = []
        for sequence in sequences:
            if not sequence:
                raise RuntimeError(f"RoboMeter returned an empty {field} sequence")
            values.append(float(sequence[-1]))
        return np.asarray(values, dtype=np.float32)

    def score_progress_batch(
        self,
        trajectories: Sequence[np.ndarray],
        tasks: Sequence[str],
        *,
        sample_ids: Sequence[str] | None = None,
        presample: bool = True,
    ) -> RobometerBatchResult:
        if len(trajectories) != len(tasks):
            raise ValueError("trajectories and tasks must have equal length")
        if not trajectories:
            raise ValueError("At least one trajectory is required")
        if sample_ids is None:
            sample_ids = [str(index) for index in range(len(trajectories))]
        if len(sample_ids) != len(trajectories):
            raise ValueError("sample_ids and trajectories must have equal length")

        files: dict[str, tuple[str, io.BytesIO, str]] = {}
        data: dict[str, str] = {"use_frame_steps": "false"}
        for index, (raw_frames, task, sample_id) in enumerate(
            zip(trajectories, tasks, sample_ids, strict=True)
        ):
            raw_frames = np.asarray(raw_frames)
            if presample:
                frames = official_subsample_and_pad(raw_frames, self.max_frames)
            else:
                if raw_frames.ndim != 4 or raw_frames.shape[0] == 0:
                    raise ValueError(
                        "Prepared RoboMeter contexts must be non-empty (T,H,W,C), "
                        f"got {raw_frames.shape}"
                    )
                frames = np.ascontiguousarray(raw_frames, dtype=np.uint8)
            file_key = f"sample_{index}_trajectory_frames"
            buffer = io.BytesIO()
            np.save(buffer, frames)
            buffer.seek(0)
            files[file_key] = (
                f"{file_key}.npy",
                buffer,
                "application/octet-stream",
            )
            sample = {
                "sample_type": "progress",
                "trajectory": {
                    "frames": {"__numpy_file__": file_key},
                    "frames_shape": list(frames.shape),
                    "task": str(task),
                    "id": str(sample_id),
                    "lang_vector": None,
                    "metadata": {"subsequence_length": int(raw_frames.shape[0])},
                    "video_embeddings": None,
                    "text_embedding": None,
                },
                "data_gen_strategy": None,
                "resample_attempts": 1,
            }
            data[f"sample_{index}"] = json.dumps(sample)

        response = None
        with self._request_lock:
            for attempt in range(self.connection_retries + 1):
                for _, buffer, _ in files.values():
                    buffer.seek(0)
                try:
                    response = self.session.post(
                        f"{self.server_url}/evaluate_batch_npy",
                        files=files,
                        data=data,
                        timeout=self.timeout_s,
                    )
                    break
                except requests.ConnectionError:
                    if attempt >= self.connection_retries:
                        raise
                    self.session.close()
                    self.session = requests.Session()
        assert response is not None
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"RoboMeter request failed at {self.server_url}: "
                f"{response.text[:2000]}"
            ) from exc
        outputs = response.json()
        progress = self._last_values(outputs, "outputs_progress", "progress_pred")
        success_prob = self._last_values(
            outputs, "outputs_success", "success_probs"
        )
        expected = len(trajectories)
        if progress.shape != (expected,) or success_prob.shape != (expected,):
            raise RuntimeError(
                "RoboMeter batch size mismatch: "
                f"expected={expected}, progress={progress.shape}, "
                f"success={success_prob.shape}"
            )
        return RobometerBatchResult(progress=progress, success_prob=success_prob)


class OfficialRobometerPool:
    """Route independent samples across official RoboMeter server replicas."""

    def __init__(
        self,
        server_urls: Sequence[str],
        *,
        max_frames: int = 8,
        timeout_s: float = 120.0,
        connection_retries: int = 2,
        strict_qwen3: bool = True,
        max_pending_batches_per_server: int = 2,
    ) -> None:
        urls = [str(url).rstrip("/") for url in server_urls]
        if not urls:
            raise ValueError("At least one RoboMeter server URL is required")
        if len(set(urls)) != len(urls):
            raise ValueError(f"RoboMeter server URLs must be unique, got {urls}")
        self.clients = [
            OfficialRobometerClient(
                url,
                max_frames=max_frames,
                timeout_s=timeout_s,
                connection_retries=connection_retries,
                strict_qwen3=strict_qwen3,
            )
            for url in urls
        ]
        self.server_urls = urls
        self.max_frames = int(max_frames)
        self.model_info = {
            client.server_url: client.model_info for client in self.clients
        }
        self.max_pending_batches_per_server = int(max_pending_batches_per_server)
        if self.max_pending_batches_per_server <= 0:
            raise ValueError("max_pending_batches_per_server must be positive")
        self._async_executors = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"robometer-{index}")
            for index in range(len(self.clients))
        ]
        self._async_assignments: list[_AsyncRobometerAssignment] = []
        self._async_ready: list[AsyncRobometerSampleResult] = []
        self._token_counter = itertools.count()
        self._async_lock = threading.Lock()
        self._closed = False

    @property
    def pending_count(self) -> int:
        with self._async_lock:
            return sum(len(item.tokens) for item in self._async_assignments)

    def submit_progress_batch(
        self,
        trajectories: Sequence[np.ndarray],
        tasks: Sequence[str],
        *,
        sample_ids: Sequence[str] | None = None,
        presample: bool = True,
    ) -> list[str]:
        """Queue samples per server and return immediately with stable tokens.

        Each server has one worker, so requests to a replica remain ordered and
        at most one request executes there at a time. Results can be drained per
        replica; a slow server therefore does not hold completed results from the
        other replicas behind a batch-wide barrier.
        """
        count = len(trajectories)
        if count != len(tasks):
            raise ValueError("trajectories and tasks must have equal length")
        if count == 0:
            raise ValueError("At least one trajectory is required")
        if sample_ids is None:
            sample_ids = [str(index) for index in range(count)]
        if count != len(sample_ids):
            raise ValueError("sample_ids and trajectories must have equal length")
        if self._closed:
            raise RuntimeError("RoboMeter pool is closed")

        tokens = [f"robometer-{next(self._token_counter)}" for _ in range(count)]
        assignments: list[list[int]] = [[] for _ in self.clients]
        for index in range(count):
            assignments[index % len(self.clients)].append(index)

        active_urls = {
            self.clients[index].server_url
            for index, indices in enumerate(assignments)
            if indices
        }
        self._wait_for_async_capacity(active_urls)

        queued: list[_AsyncRobometerAssignment] = []
        for client_index, indices in enumerate(assignments):
            if not indices:
                continue
            client = self.clients[client_index]
            future = self._async_executors[client_index].submit(
                client.score_progress_batch,
                [trajectories[index] for index in indices],
                [tasks[index] for index in indices],
                sample_ids=[sample_ids[index] for index in indices],
                presample=presample,
            )
            queued.append(
                _AsyncRobometerAssignment(
                    tokens=tuple(tokens[index] for index in indices),
                    server_url=client.server_url,
                    future=future,
                )
            )
        with self._async_lock:
            self._async_assignments.extend(queued)
        return tokens

    @staticmethod
    def _materialize_async_assignment(
        assignment: _AsyncRobometerAssignment,
    ) -> list[AsyncRobometerSampleResult]:
        batch = assignment.future.result()
        if len(assignment.tokens) != len(batch.progress):
            raise RuntimeError(
                "Async RoboMeter result size mismatch: "
                f"tokens={len(assignment.tokens)}, progress={len(batch.progress)}"
            )
        return [
            AsyncRobometerSampleResult(
                token=token,
                progress=float(batch.progress[index]),
                success_prob=float(batch.success_prob[index]),
                server_url=assignment.server_url,
            )
            for index, token in enumerate(assignment.tokens)
        ]

    def _wait_for_async_capacity(self, active_urls: set[str]) -> None:
        """Bound queued video payloads while allowing one chunk of overlap."""
        while True:
            with self._async_lock:
                assignments = list(self._async_assignments)
            counts = {
                url: sum(item.server_url == url for item in assignments)
                for url in active_urls
            }
            overloaded = {
                url
                for url, count in counts.items()
                if count >= self.max_pending_batches_per_server
            }
            if not overloaded:
                return
            candidates = [
                item for item in assignments if item.server_url in overloaded
            ]
            wait([item.future for item in candidates], return_when=FIRST_COMPLETED)
            done = [item for item in candidates if item.future.done()]
            done_ids = {id(item) for item in done}
            ready = []
            for item in done:
                ready.extend(self._materialize_async_assignment(item))
            with self._async_lock:
                self._async_assignments = [
                    item
                    for item in self._async_assignments
                    if id(item) not in done_ids
                ]
                self._async_ready.extend(ready)

    def drain_completed(
        self, *, wait_for_all: bool = False
    ) -> list[AsyncRobometerSampleResult]:
        """Return completed samples without waiting for unrelated replicas."""
        with self._async_lock:
            assignments = list(self._async_assignments)
            results = list(self._async_ready)
            self._async_ready.clear()

        completed_assignments: list[_AsyncRobometerAssignment] = []
        for assignment in assignments:
            if not wait_for_all and not assignment.future.done():
                continue
            completed_assignments.append(assignment)
            results.extend(self._materialize_async_assignment(assignment))

        if completed_assignments:
            completed_ids = {id(item) for item in completed_assignments}
            with self._async_lock:
                self._async_assignments = [
                    item
                    for item in self._async_assignments
                    if id(item) not in completed_ids
                ]
        return results

    def close(self) -> None:
        if self._closed:
            return
        self.drain_completed(wait_for_all=True)
        for executor in self._async_executors:
            executor.shutdown(wait=True, cancel_futures=False)
        for client in self.clients:
            client.session.close()
        self._closed = True

    def score_progress_batch(
        self,
        trajectories: Sequence[np.ndarray],
        tasks: Sequence[str],
        *,
        sample_ids: Sequence[str] | None = None,
        presample: bool = True,
    ) -> RobometerBatchResult:
        count = len(trajectories)
        if count != len(tasks):
            raise ValueError("trajectories and tasks must have equal length")
        if sample_ids is None:
            sample_ids = [str(index) for index in range(count)]
        if count != len(sample_ids):
            raise ValueError("sample_ids and trajectories must have equal length")
        if count == 0:
            raise ValueError("At least one trajectory is required")
        if len(self.clients) == 1:
            return self.clients[0].score_progress_batch(
                trajectories,
                tasks,
                sample_ids=sample_ids,
                presample=presample,
            )

        assignments: list[list[int]] = [[] for _ in self.clients]
        for index in range(count):
            assignments[index % len(self.clients)].append(index)

        progress = np.empty(count, dtype=np.float32)
        success_prob = np.empty(count, dtype=np.float32)

        def score(client_index: int, indices: list[int]):
            client = self.clients[client_index]
            result = client.score_progress_batch(
                [trajectories[index] for index in indices],
                [tasks[index] for index in indices],
                sample_ids=[sample_ids[index] for index in indices],
                presample=presample,
            )
            return indices, result

        active = [
            (client_index, indices)
            for client_index, indices in enumerate(assignments)
            if indices
        ]
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = [executor.submit(score, *item) for item in active]
            for future in futures:
                indices, result = future.result()
                progress[indices] = result.progress
                success_prob[indices] = result.success_prob
        return RobometerBatchResult(
            progress=progress,
            success_prob=success_prob,
        )
