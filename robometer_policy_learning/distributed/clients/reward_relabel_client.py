"""
Client for reward relabeling service.
Batches transitions and sends them to the reward relabeling server.
"""

import threading
import queue
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import grpc

from robometer_policy_learning.distributed.grpc_utils import ndarray_to_bytes
from loguru import logger

from robometer_policy_learning.distributed.protos import reward_relabel_pb2 as pb
from robometer_policy_learning.distributed.protos import reward_relabel_pb2_grpc as pb_grpc
from robometer_policy_learning.distributed.protos import learner_pb2 as learner_pb


@dataclass
class PendingRelabel:
    """Pending transition waiting for reward relabeling."""

    obs: Dict[str, Any]
    action: Any
    reward: float
    next_obs: Dict[str, Any]
    done: bool
    truncated: bool
    episode_id: Any = None
    step_in_episode: int = 0
    timestamp: Optional[float] = None
    language_instruction: Optional[str] = None
    callback: Optional[callable] = None  # Called with (relabeled_reward, transition_data)


@dataclass
class PendingRelabelBatch:
    """Pending batch of transitions with full trajectory context for reward relabeling."""

    transitions: List[PendingRelabel]  # Full trajectory context (0:t)
    batch_start_idx: int = 0  # Start index of the batch within trajectory (inclusive) - used when batch_indices is None
    batch_end_idx: Optional[int] = (
        None  # End index of the batch within trajectory (exclusive) - used when batch_indices is None
    )
    batch_indices: Optional[List[int]] = (
        None  # Specific trajectory indices to relabel (for DSRL mode) - overrides batch_start_idx/batch_end_idx if provided
    )
    episode_id: Any = None
    language_instruction: Optional[str] = None
    callback: Optional[callable] = (
        None  # Called with (success_probs: List[float], batch_data, progress_predictions_by_key: Dict[str, Dict[str, List[float]]])
    )
    trajectory_start_idx: int = 0  # For multi-stage tasks: only use frames from this index onward (0 = full trajectory)


class RewardRelabelClient:
    """
    Client that sends batches of transitions with full trajectory context to reward relabeling server.

    Args:
        address: Address of reward relabeling server (e.g., "localhost:50052")
        max_queue_size: Maximum pending batches in queue
        timeout: Timeout for RPC calls (seconds)
        flush_interval: Maximum time to wait before processing queue (seconds, for background thread)
    """

    def __init__(
        self,
        address: str,
        max_queue_size: int = 100,
        timeout: float = 60.0,  # Longer timeout for full trajectories
        flush_interval: float = 0.1,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        max_retry_backoff_s: float = 5.0,
    ):
        self.address = address
        self.timeout = timeout
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_retry_backoff_s = max_retry_backoff_s

        # Create gRPC channel
        self._create_channel()

        # Queue for pending batches (with trajectory context)
        self.pending_queue: queue.Queue[Optional[PendingRelabelBatch]] = queue.Queue(maxsize=max_queue_size)

        # Background thread for batching and sending
        self.running = False
        self._lock = threading.Lock()
        self._sender_thread = None

        # Statistics
        self.stats = {
            "batches_sent": 0,
            "transitions_sent": 0,
            "errors": 0,
            "queue_full": 0,
        }

    def _create_channel(self):
        self.channel = grpc.insecure_channel(
            self.address,
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),  # 256MB
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self.stub = pb_grpc.RewardRelabelServiceStub(self.channel)

    def _reset_channel(self):
        try:
            self.channel.close()
        except Exception:
            pass
        self._create_channel()

    def start(self):
        """Start background sender thread."""
        with self._lock:
            if self.running:
                return
            self.running = True
            self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._sender_thread.start()

    def stop(self):
        """Stop background sender thread."""
        with self._lock:
            if not self.running:
                return
            self.running = False

        # Signal stop
        self.pending_queue.put(None, timeout=1.0)

        # Wait for thread
        if self._sender_thread:
            self._sender_thread.join(timeout=5.0)

        # Process remaining items in queue
        while not self.pending_queue.empty():
            batch = self.pending_queue.get_nowait()
            if batch is not None:
                self._send_batch(batch)

        # Close channel
        self.channel.close()

    def _sender_loop(self):
        """Background loop that sends batches."""
        while self.running:
            # Get pending batch (blocks until available or timeout)
            try:
                pending_batch = self.pending_queue.get(timeout=self.flush_interval)
            except queue.Empty:
                continue

            # None is sentinel to stop
            if pending_batch is None:
                break

            # Log queue size after removing item (queue.get() removes the item)
            queue_size_after_get = self.pending_queue.qsize()

            # Send batch immediately
            # Note: queue.get() automatically removes the item from the queue
            start_time = time.perf_counter()
            try:
                self._send_batch(pending_batch)
            except Exception as exc:
                self.stats["errors"] += 1
                logger.warning(
                    f"[RewardRelabelClient] Error sending batch (will continue): {type(exc).__name__}: {exc}"
                )
            elapsed_time = time.perf_counter() - start_time

            # Log queue size after processing (may have new items added during processing)
            queue_size_after_process = self.pending_queue.qsize()
            logger.debug(
                f"[RewardRelabelClient] Batch processed: queue_size={queue_size_after_get} -> {queue_size_after_process} "
                f"(decreased by 1, new items may have been added during processing), "
                f"processing_time={elapsed_time:.3f}s"
            )
            # After processing, the batch object goes out of scope and can be garbage collected

    def _send_batch(self, batch: PendingRelabelBatch):
        """Send a batch of transitions with full trajectory context for relabeling."""
        if not batch.transitions:
            return

        # Determine batch range or specific indices
        if batch.batch_indices is not None:
            # DSRL mode: use specific indices
            batch_indices = batch.batch_indices
            batch_size = len(batch_indices)
            batch_start_idx = 0  # Not used when batch_indices is provided
            batch_end_idx = 0  # Not used when batch_indices is provided
        else:
            # Normal mode: use range
            batch_indices = None
            batch_start_idx = batch.batch_start_idx
            batch_end_idx = batch.batch_end_idx if batch.batch_end_idx is not None else len(batch.transitions)
            batch_size = batch_end_idx - batch_start_idx

        # Log batch sending
        if batch_indices is not None:
            logger.success(
                f"[RewardRelabelClient] Sending batch (DSRL mode): "
                f"trajectory_context={len(batch.transitions)}, batch_size={batch_size}, "
                f"batch_indices={batch_indices}, episode_id={batch.episode_id}"
            )
        else:
            logger.success(
                f"[RewardRelabelClient] Sending batch: "
                f"trajectory_context={len(batch.transitions)}, batch_size={batch_size}, "
                f"batch_start_idx={batch_start_idx}, batch_end_idx={batch_end_idx}, episode_id={batch.episode_id}"
            )

        # Convert all transitions in trajectory context to proto format
        # We send the full context (0:t) but only extract rewards for the batch range
        transitions = []
        language_instructions = []
        episode_ids = []
        step_in_episodes = []

        # Use batch-level language instruction
        batch_lang = batch.language_instruction

        for pending in batch.transitions:
            # Convert transition to proto (contains obs/next_obs with images)
            obs_proto = {k: learner_pb.NDArray(data=ndarray_to_bytes(v)) for k, v in pending.obs.items()}
            next_obs_proto = {k: learner_pb.NDArray(data=ndarray_to_bytes(v)) for k, v in pending.next_obs.items()}
            action_proto = learner_pb.NDArray(data=ndarray_to_bytes(pending.action))

            # Convert timestamp to nanoseconds (int64)
            # Handle both seconds (Unix timestamp) and nanoseconds
            if pending.timestamp:
                # If timestamp > 1e12, assume it's already in nanoseconds
                if pending.timestamp > 1e12:
                    timestamp_ns = int(pending.timestamp)
                else:
                    # Convert from seconds to nanoseconds
                    timestamp_ns = int(pending.timestamp * 1e9)
                # Clamp to int64 range to prevent overflow
                max_int64 = 2**63 - 1
                timestamp_ns = min(max(timestamp_ns, 0), max_int64)
            else:
                timestamp_ns = 0

            tr = learner_pb.Transition(
                obs=obs_proto,
                action=action_proto,
                reward_env=pending.reward,
                next_obs=next_obs_proto,
                done=pending.done,
                truncated=pending.truncated,
                episode_id=str(batch.episode_id) if batch.episode_id is not None else "",
                step_in_episode=pending.step_in_episode,
                timestamp_ns=timestamp_ns,
            )
            transitions.append(tr)

            # Use batch-level language instruction, fallback to transition-level
            language_instructions.append(batch_lang or pending.language_instruction or "")

            episode_ids.append(str(batch.episode_id) if batch.episode_id is not None else "")
            step_in_episodes.append(pending.step_in_episode)

        request = pb.RelabelRewardsRequest(
            transitions=transitions,
            language_instructions=language_instructions,
            episode_ids=episode_ids,
            step_in_episodes=step_in_episodes,
            batch_start_idx=batch_start_idx,
            batch_end_idx=batch_end_idx,
            trajectory_start_idx=batch.trajectory_start_idx,
        )

        # Add batch_indices if provided (DSRL mode)
        if batch_indices is not None:
            if hasattr(request, "batch_indices"):
                request.batch_indices.extend(batch_indices)
                logger.debug(f"[RewardRelabelClient] Added batch_indices={batch_indices} to request")
            else:
                logger.error(
                    f"[RewardRelabelClient] ERROR: batch_indices provided ({batch_indices}) but protobuf request doesn't have batch_indices field! "
                    f"Protobuf files need to be regenerated. Falling back to normal mode (batch_start_idx/batch_end_idx)."
                )
                # Fall back to normal mode
                if batch.batch_start_idx is not None and batch.batch_end_idx is not None:
                    request.batch_start_idx = batch.batch_start_idx
                    request.batch_end_idx = batch.batch_end_idx
                else:
                    # Calculate range from batch_indices
                    if batch_indices:
                        request.batch_start_idx = min(batch_indices)
                        request.batch_end_idx = max(batch_indices) + 1
                    else:
                        request.batch_start_idx = 0
                        request.batch_end_idx = len(batch.transitions)

        # Send request with retries (handles server restarts / transient errors)
        response = None
        backoff_s = self.retry_backoff_s
        for attempt in range(self.max_retries + 1):
            try:
                response = self.stub.RelabelRewards(request, timeout=self.timeout)
                break
            except grpc.RpcError as exc:
                self.stats["errors"] += 1
                code = exc.code().name if hasattr(exc, "code") and exc.code() is not None else "UNKNOWN"
                logger.warning(
                    f"[RewardRelabelClient] RPC error (attempt {attempt + 1}/{self.max_retries + 1}, code={code}): {exc}"
                )
                self._reset_channel()
                if attempt < self.max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, self.max_retry_backoff_s)
                    continue
                raise
            except Exception as exc:
                self.stats["errors"] += 1
                logger.warning(
                    f"[RewardRelabelClient] Non-RPC error (attempt {attempt + 1}/{self.max_retries + 1}): {exc}"
                )
                self._reset_channel()
                if attempt < self.max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, self.max_retry_backoff_s)
                    continue
                raise

        if response is None:
            raise RuntimeError("[RewardRelabelClient] No response received after retries")

        # Process responses - extract progress predictions (rewards will be computed from progress)
        if response.ok:
            # logger.success(f"[RewardRelabelClient] Batch relabeled successfully")
            # Extract progress predictions by key
            progress_predictions_by_key = {}
            if hasattr(response, "progress_predictions_by_key") and response.progress_predictions_by_key:
                for image_key, progress_predictions in response.progress_predictions_by_key.items():
                    progress_predictions_by_key[image_key] = {
                        "progress": list(progress_predictions.progress),
                        "success_probs": list(progress_predictions.success_probs),
                    }

            # Extract success probabilities (aggregated from progress_predictions_by_key)
            batch_success_probs = []
            if progress_predictions_by_key:
                # Compute aggregated success probs from progress_predictions_by_key
                first_key = next(iter(progress_predictions_by_key))
                batch_success_probs = progress_predictions_by_key[first_key]["success_probs"]
            else:
                if batch_indices is not None:
                    batch_size = len(batch_indices)
                else:
                    batch_size = batch_end_idx - batch_start_idx
                batch_success_probs = [0.0] * batch_size

            # Create batch for callback - keep full trajectory context for consistency
            # The callback uses batch_indices/batch_start_idx to map to self._trajectory, but we pass
            # the full context for consistency with how the request was sent to the server
            if batch_indices is not None:
                # DSRL mode: pass full trajectory context, use batch_indices to indicate which were processed
                batch_for_callback = PendingRelabelBatch(
                    transitions=batch.transitions,  # Full trajectory context (same as sent to server)
                    batch_indices=batch_indices,  # Keep original indices for callback
                    episode_id=batch.episode_id,
                    language_instruction=batch.language_instruction,
                    callback=batch.callback,
                )
            else:
                # Normal mode: pass full trajectory context, use batch_start_idx/batch_end_idx
                batch_for_callback = PendingRelabelBatch(
                    transitions=batch.transitions,  # Full trajectory context (same as sent to server)
                    batch_start_idx=batch_start_idx,  # Keep original so callback can map to correct step indices
                    batch_end_idx=batch_end_idx,  # Keep original so callback can map to correct step indices
                    episode_id=batch.episode_id,
                    language_instruction=batch.language_instruction,
                    callback=batch.callback,
                )

            # Call callback with success probabilities and progress predictions by key
            # Rewards are computed from progress_predictions_by_key in the wrapper
            if batch.callback:
                # logger.success(
                #     f"[RewardRelabelClient] Calling callback with success probabilities and progress predictions by key"
                # )
                batch.callback(batch_success_probs, batch_for_callback, progress_predictions_by_key)
        else:
            raise Exception(f"Error relabeling batch: {response.message}")

        self.stats["batches_sent"] += 1
        self.stats["transitions_sent"] += batch_size

    def relabel_batch(self, batch: PendingRelabelBatch):
        """
        Queue a batch of transitions with trajectory context for reward relabeling (async).

        Args:
            batch: PendingRelabelBatch containing batch transitions and full trajectory context
        """
        queue_size = self.pending_queue.qsize()
        max_size = self.pending_queue.maxsize

        # Periodic debug logging (if queue > 50% full)
        if queue_size > max_size * 0.8:
            logger.debug(
                f"[RewardRelabelClient] Queue status: size={queue_size}/{max_size} "
                f"({queue_size / max_size * 100:.1f}% full), batches_sent={self.stats['batches_sent']}, "
                f"transitions_sent={self.stats['transitions_sent']}, errors={self.stats['errors']}"
            )

        try:
            self.pending_queue.put(batch, block=False)
        except queue.Full:
            self.stats["queue_full"] += 1
            # Drop oldest batch to make room (avoid crashing when server is down)
            try:
                dropped = self.pending_queue.get_nowait()
                if dropped is not None:
                    logger.warning(
                        "[RewardRelabelClient] Queue full: dropped oldest pending batch to make room"
                    )
            except queue.Empty:
                pass
            try:
                self.pending_queue.put(batch, block=False)
            except queue.Full:
                logger.error(
                    "[RewardRelabelClient] Queue still full after drop; dropping new batch"
                )

    def relabel_batch_sync(self, batch: PendingRelabelBatch):
        """
        Synchronously relabel a batch of transitions (blocks until complete).

        Args:
            batch: PendingRelabelBatch containing batch transitions and full trajectory context
        """
        # Call _send_batch directly (synchronous, blocking)
        self._send_batch(batch)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            "queue_size": self.pending_queue.qsize(),
            "running": self.running,
        }
