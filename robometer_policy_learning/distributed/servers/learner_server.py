import threading
import time
import traceback
import pickle
from concurrent import futures
from typing import Optional, Dict, Any

import grpc
import numpy as np

from robometer_policy_learning.distributed.grpc_utils import (
    bytes_to_ndarray,
    ndarray_to_bytes,
    state_dict_to_bytes,
)
from robometer_policy_learning.utils.fingerprint_utils import fingerprint_bytes
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer

# from reward_models import BaseRewardModel
from loguru import logger

# Lazy proto import after grpc_tools.protoc generation (runtime import fallback)
from robometer_policy_learning.distributed.protos import learner_pb2 as pb
from robometer_policy_learning.distributed.protos import learner_pb2_grpc as pb_grpc


class _ParameterStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._version = 0
        self._state_dict_bytes = b""
        self._fingerprint = ""

    def set(self, state_dict: Dict[str, Any]):
        with self._lock:
            self._state_dict_bytes = state_dict_to_bytes(state_dict)
            self._version += 1
            # Compute fingerprint once at publish time
            self._fingerprint = fingerprint_bytes(self._state_dict_bytes) if self._state_dict_bytes else ""
            return self._version, self._fingerprint

    def get(self):
        with self._lock:
            return self._state_dict_bytes, self._version, self._fingerprint


class IngestionService(pb_grpc.IngestionServiceServicer):
    def __init__(self, buffer: ReplayBuffer, server_instance=None):
        # Note: reward relabeling is handled by buffer pre/post transforms configured at construction.
        self.buffer = buffer
        self.server_instance = server_instance

    def StreamTransitions(self, request_iterator, context):
        # Register this connection when streaming starts
        if self.server_instance:
            self.server_instance.register_connection(context)
        accepted = 0
        last_log = time.time()
        dropped = 0
        try:
            for batch in request_iterator:
                for tr in batch.transitions:
                    # If ingest is disabled (e.g., during offline warm-start), drop transitions
                    if self.server_instance and not getattr(self.server_instance, "ingest_enabled", True):
                        dropped += 1
                        continue
                    obs = {k: bytes_to_ndarray(v.data) for k, v in tr.obs.items()}
                    next_obs = {k: bytes_to_ndarray(v.data) for k, v in tr.next_obs.items()}
                    action = bytes_to_ndarray(tr.action.data)

                    # Extract language instruction from info if available (data is already in obs)
                    language_instruction = None
                    for entry in tr.info:
                        if entry.key == "__language_instruction__":
                            language_instruction = entry.value.decode("utf-8")
                            break

                    self.buffer.add(
                        obs=obs,
                        action=action,
                        reward=float(tr.reward_env),
                        next_obs=next_obs,
                        done=tr.done,
                        truncated=tr.truncated,
                        episode_id=tr.episode_id,
                        step_in_episode=tr.step_in_episode,
                        timestamp=tr.timestamp_ns,
                        language_instruction=language_instruction,
                    )
                    accepted += 1
                # periodic ingest log (throttled)
                now = time.time()
                if now - last_log >= 5.0:
                    try:
                        qsize = getattr(self.buffer, "size", lambda: len(self.buffer))()
                    except Exception:
                        qsize = len(self.buffer)
                    # Only annotate as disabled if currently disabled. If previously dropped, report once.
                    if getattr(self.server_instance, "ingest_enabled", True) is False:
                        logger.warning(
                            f"[INGEST] accepted_total={accepted} dropped={dropped} (ingest disabled) online_buffer_len={qsize}"
                        )
                    else:
                        if dropped > 0:
                            logger.info(
                                f"[INGEST] accepted_total={accepted} (previously dropped={dropped}) online_buffer_len={qsize}"
                            )
                        else:
                            logger.info(f"[INGEST] accepted_total={accepted} online_buffer_len={qsize}")
                        # Reset after reporting previous drops once
                        dropped = 0
                    last_log = now
        except Exception as e:
            logger.exception(f"[INGEST][ERROR] {e}\n{traceback.format_exc()}")
            # Unregister connection on error
            if self.server_instance:
                self.server_instance.unregister_connection(context)
            return pb.Ack(ok=False, message=str(e), accepted=accepted)
        finally:
            # Unregister connection when stream ends (normal or error)
            if self.server_instance:
                self.server_instance.unregister_connection(context)
        return pb.Ack(ok=True, message="ok", accepted=accepted)


class PolicyService(pb_grpc.PolicyServiceServicer):
    def __init__(self, param_store: _ParameterStore, policy_info_provider=None):
        self.param_store = param_store
        self._policy_info_provider = policy_info_provider  # callable -> dict

    def GetActor(self, request, context):
        blob, version, fp = self.param_store.get()
        if request.since_version and request.since_version == version:
            # client already up-to-date; still return the same version
            pass
        kwargs = {"state_dict": blob, "version": version}
        try:
            if hasattr(pb.ActorBlob, "DESCRIPTOR") and "fingerprint" in pb.ActorBlob.DESCRIPTOR.fields_by_name:
                kwargs["fingerprint"] = fp or ""
        except Exception:
            pass
        return pb.ActorBlob(**kwargs)

    def GetActorStream(self, request, context):
        blob, version, fp = self.param_store.get()
        chunk_size = 8 * 1024 * 1024  # 8MB
        total = len(blob)
        if total == 0:
            kwargs = {"data": b"", "version": version, "total_size": 0, "seq": 0, "num_chunks": 0}
            try:
                if hasattr(pb.ActorChunk, "DESCRIPTOR") and "fingerprint" in pb.ActorChunk.DESCRIPTOR.fields_by_name:
                    kwargs["fingerprint"] = fp or ""
            except Exception:
                pass
            yield pb.ActorChunk(**kwargs)
            return
        num_chunks = (total + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            kwargs = {
                "data": blob[start:end],
                "version": version,
                "total_size": total,
                "seq": i,
                "num_chunks": num_chunks,
            }
            try:
                if hasattr(pb.ActorChunk, "DESCRIPTOR") and "fingerprint" in pb.ActorChunk.DESCRIPTOR.fields_by_name:
                    kwargs["fingerprint"] = fp or ""
            except Exception:
                pass
            yield pb.ActorChunk(**kwargs)

    def GetPolicyInfo(self, request, context):
        if self._policy_info_provider is None:
            return pb.PolicyInfo(obs_keys=[], chunk_size=1, wandb_run_id="", wandb_project="", wandb_entity="")
        info = self._policy_info_provider()
        chunk_size = info.get("chunk_size", 1)
        if chunk_size is None:
            chunk_size = 1
        return pb.PolicyInfo(
            obs_keys=info.get("obs_keys", []),
            chunk_size=int(chunk_size),
            wandb_run_id=info.get("wandb_run_id", ""),
            wandb_project=info.get("wandb_project", ""),
            wandb_entity=info.get("wandb_entity", ""),
        )


class LearnerServer:
    def __init__(
        self,
        buffer: ReplayBuffer,
        reward_model=None,
        host: str = "0.0.0.0",
        port: int = 50051,
        relabel_mode: str = "pre",
        max_workers: int = 8,
        max_msg_mb: int = 64,
        policy_info_provider=None,
    ):
        self.param_store = _ParameterStore()
        self.buffer = buffer
        # self.reward_model = reward_model
        self.host = host
        self.port = port
        # Gate to allow / disallow online ingestion (e.g., disabled during offline warm-start)
        self.ingest_enabled = True

        # Connection tracking
        import threading

        self.active_connections = set()
        self.connection_lock = threading.Lock()
        self.last_connection_time = {}  # peer -> timestamp
        # Increase message size limits to support large checkpoints
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_receive_message_length", max_msg_mb * 1024 * 1024),
                ("grpc.max_send_message_length", max_msg_mb * 1024 * 1024),
            ],
        )

        pb_grpc.add_IngestionServiceServicer_to_server(
            IngestionService(buffer=self.buffer, server_instance=self), self.server
        )
        pb_grpc.add_PolicyServiceServicer_to_server(
            PolicyService(self.param_store, policy_info_provider=policy_info_provider), self.server
        )
        self.server.add_insecure_port(f"{self.host}:{self.port}")

    def start(self):
        self.server.start()
        logger.success(f"[SERVER] gRPC listening on {self.host}:{self.port}")

    def stop(self, grace: Optional[float] = 5.0):
        self.server.stop(grace)

    def set_actor_state_dict(self, state_dict: Dict[str, Any]):
        version, fp = self.param_store.set(state_dict)
        try:
            logger.info(f"[SERVER] Published policy version={version} fp={fp}")
        except Exception:
            pass

    def set_ingest_enabled(self, enabled: bool):
        self.ingest_enabled = bool(enabled)

    def register_connection(self, context):
        """Register a new connection from a rollout worker."""
        peer = context.peer()
        with self.connection_lock:
            self.active_connections.add(peer)
            self.last_connection_time[peer] = time.time()
            logger.info(f"[CONNECTION] Rollout worker connected: {peer} (total: {len(self.active_connections)})")

    def unregister_connection(self, context):
        """Unregister a connection when it disconnects."""
        peer = context.peer()
        with self.connection_lock:
            if peer in self.active_connections:
                self.active_connections.remove(peer)
                self.last_connection_time.pop(peer, None)
                logger.info(f"[CONNECTION] Rollout worker disconnected: {peer} (total: {len(self.active_connections)})")

    def get_active_connection_count(self):
        """Get the number of active rollout worker connections."""
        with self.connection_lock:
            return len(self.active_connections)

    def has_active_connections(self):
        """Check if there are any active rollout worker connections."""
        return self.get_active_connection_count() > 0
