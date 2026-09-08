from typing import Optional, Dict, Any
import grpc
import time
import torch

from robometer_policy_learning.distributed.grpc_utils import bytes_to_state_dict
from robometer_policy_learning.distributed.protos import learner_pb2 as pb
from robometer_policy_learning.distributed.protos import learner_pb2_grpc as pb_grpc
from loguru import logger


class PolicyClient:
    def __init__(self, address: str, ready_log_prefix: str = "", ready_timeout_per_attempt: float = 5.0):
        # Increase client message limits to handle large checkpoints
        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ],
        )
        self.stub = pb_grpc.PolicyServiceStub(self.channel)
        self.version = 0
        self._ready_log_prefix = ready_log_prefix
        self._address = address
        self._ready_timeout_per_attempt = ready_timeout_per_attempt

    def _wait_for_channel_ready(self):
        backoff = 1.0
        while True:
            try:
                grpc.channel_ready_future(self.channel).result(timeout=self._ready_timeout_per_attempt)
                return
            except Exception:
                if self._ready_log_prefix:
                    logger.info(
                        f"{self._ready_log_prefix} Learner not ready at {self._address}; retrying in {backoff:.1f}s"
                    )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    def fetch_latest(self, block: bool = False) -> Dict[str, Any]:
        if block:
            self._wait_for_channel_ready()
            backoff = 1.0
            while True:
                try:
                    return self.fetch_latest(block=False)
                except Exception:
                    if self._ready_log_prefix:
                        logger.info(f"{self._ready_log_prefix} Weights not available yet; retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 10.0)
        # Try streaming first for very large checkpoints
        try:
            chunks = []
            total = None
            version = None
            fp = None
            for ch in self.stub.GetActorStream(pb.GetActorRequest(since_version=self.version)):
                if version is None:
                    version = int(ch.version)
                    total = int(ch.total_size)
                    try:
                        fp = ch.fingerprint or None
                    except Exception:
                        fp = None
                chunks.append(ch.data)
            if chunks:
                blob = b"".join(chunks)
                self.version = version
                sd = bytes_to_state_dict(blob)
                # Attach fingerprint for caller diagnostics if desired
                if fp is not None:
                    sd["__fingerprint__"] = fp
                return sd
        except Exception:
            # Fallback to unary RPC (smaller models)
            pass
        resp = self.stub.GetActor(pb.GetActorRequest(since_version=self.version))
        self.version = int(resp.version)
        sd = bytes_to_state_dict(resp.state_dict)
        # Attach fingerprint if the field exists in this proto
        try:
            fp = getattr(resp, "fingerprint", "") or None
            if fp is not None:
                sd["__fingerprint__"] = fp
        except Exception:
            pass
        return sd

    def get_policy_info(self, block: bool = False) -> Dict[str, Any]:
        if block:
            self._wait_for_channel_ready()
            backoff = 1.0
            while True:
                try:
                    return self.get_policy_info(block=False)
                except Exception:
                    if self._ready_log_prefix:
                        logger.info(f"{self._ready_log_prefix} Policy info not available; retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 10.0)
        try:
            info = self.stub.GetPolicyInfo(pb.Empty())
            return {
                "obs_keys": list(info.obs_keys),
                "chunk_size": int(info.chunk_size),
                "wandb_run_id": info.wandb_run_id,
                "wandb_project": info.wandb_project,
                "wandb_entity": info.wandb_entity,
            }
        except Exception as e:
            logger.warning(f"Policy info fetch failed: {e}")
            return {
                "obs_keys": None,
                "chunk_size": None,
                "wandb_run_id": None,
                "wandb_project": None,
                "wandb_entity": None,
            }
