from typing import Dict, Any, Iterable, List, Optional
import time
import threading
import queue
import numpy as np
import grpc

from robometer_policy_learning.distributed.grpc_utils import ndarray_to_bytes
from robometer_policy_learning.distributed.protos import learner_pb2 as pb
from robometer_policy_learning.distributed.protos import learner_pb2_grpc as pb_grpc
from loguru import logger


class RemoteReplayBufferClient:
    """
    Minimal client to stream transitions to the Learner's IngestionService.
    This is a write-only client for rollout workers.
    """

    def __init__(self, address: str, max_batch_size: int = 512, max_message_mb: int = 64):
        self.address = address
        # Keepalive and message size options for long-lived streams
        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", max_message_mb * 1024 * 1024),
                ("grpc.max_receive_message_length", max_message_mb * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self.stub = pb_grpc.IngestionServiceStub(self.channel)
        self.max_batch_size = max_batch_size
        self.max_message_bytes = max_message_mb * 1024 * 1024
        # Streaming infrastructure
        self._queue: "queue.Queue[Optional[List[Dict[str, Any]]]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._sender_thread = threading.Thread(target=self._run_stream, daemon=True)
        self._sender_thread.start()

    def _to_proto_transition(self, tr: Dict[str, Any]) -> pb.Transition:
        obs = {k: pb.NDArray(data=ndarray_to_bytes(v)) for k, v in tr["obs"].items()}
        next_obs = {k: pb.NDArray(data=ndarray_to_bytes(v)) for k, v in tr["next_obs"].items()}
        action = pb.NDArray(data=ndarray_to_bytes(tr["action"]))
        info_entries = []
        for k, v in tr.get("info", {}).items():
            if isinstance(v, bytes):
                info_entries.append(pb.MetaEntry(key=k, value=v))
            else:
                info_entries.append(pb.MetaEntry(key=k, value=str(v).encode()))
        return pb.Transition(
            info=info_entries,
            obs=obs,
            action=action,
            reward_env=float(tr.get("reward_env", tr.get("reward", 0.0))),
            next_obs=next_obs,
            done=bool(tr["done"]),
            truncated=bool(tr.get("truncated", False)),
            episode_id=str(tr.get("episode_id", "")),
            step_in_episode=int(tr.get("step_in_episode", 0)),
            timestamp_ns=int(tr.get("timestamp", int(time.time_ns()))),
        )

    def _iter_batches(self):
        """Iterator yielding TransitionBatch for the active stream.
        Consumes lists of transition dicts from the internal queue; None is a sentinel to end the stream.
        """
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                break
            transitions_iter = item
            batch: List[pb.Transition] = []
            batch_bytes = 0
            for tr in transitions_iter:
                # Build proto and estimate size
                proto_tr = self._to_proto_transition(tr)
                tr_bytes = 0
                for _, arr in proto_tr.obs.items():
                    tr_bytes += len(arr.data)
                for _, arr in proto_tr.next_obs.items():
                    tr_bytes += len(arr.data)
                tr_bytes += len(proto_tr.action.data)

                # If adding this transition would exceed the message limit, flush current batch
                if batch and (batch_bytes + tr_bytes > self.max_message_bytes or len(batch) >= self.max_batch_size):
                    yield pb.TransitionBatch(transitions=batch)
                    batch = []
                    batch_bytes = 0

                batch.append(proto_tr)
                batch_bytes += tr_bytes

            if batch:
                yield pb.TransitionBatch(transitions=batch)

    def _run_stream(self):
        """Maintains a long-lived streaming RPC, reconnecting on errors."""
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                # Start stream; this will block until error or stop
                _ = self.stub.StreamTransitions(self._iter_batches())
                # If stream ends normally, small delay before restart
                time.sleep(0.1)
                backoff = 1.0
            except Exception as e:
                logger.warning(f"[STREAM] stream error: {e}; reconnecting in {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)

    def enqueue(self, transitions: List[Dict[str, Any]]):
        """Enqueue transitions to be sent on the persistent stream."""
        if transitions:
            self._queue.put(transitions)

    def stop(self):
        """Stop background streaming and close channel."""
        self._stop_event.set()
        try:
            self._queue.put(None)
        except Exception:
            pass
        try:
            self._sender_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.channel.close()
        except Exception:
            pass
