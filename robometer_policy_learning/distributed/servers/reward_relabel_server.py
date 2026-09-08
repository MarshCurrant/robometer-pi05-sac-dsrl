"""
Reward relabeling server that processes batches of transitions and returns relabeled rewards.

Supports two reward model types:
- Robometer (default): Qwen-based reward model for progress and success prediction
- RoboReward: Baseline model that predicts discrete progress scores (1-5)
"""

from concurrent import futures
from typing import Optional, Dict, Any, List
import time
import torch
from contextlib import nullcontext

import grpc
import numpy as np

from robometer_policy_learning.distributed.grpc_utils import bytes_to_ndarray
from robometer.utils.setup_utils import setup_batch_collator
from robometer.evals.eval_utils import raw_dict_to_sample
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.embedding_utils import compute_text_embeddings
from robometer.utils.save import load_model_from_hf
from loguru import logger

from robometer_policy_learning.distributed.protos import reward_relabel_pb2 as pb
from robometer_policy_learning.distributed.protos import reward_relabel_pb2_grpc as pb_grpc

# Optional RoboReward import
try:
    from robometer.evals.baselines.roboreward import RoboReward
    HAS_ROBOREWARD = True
except ImportError:
    HAS_ROBOREWARD = False
    RoboReward = None


class RewardRelabelService(pb_grpc.RewardRelabelServiceServicer):
    """
    Service that relabels rewards for batches of transitions using a Qwen reward model.

    This service receives transitions from clients, reconstructs full episode subsequences,
    processes them through the reward model, and returns relabeled rewards.
    """

    def __init__(
        self,
        reward_model,
        exp_config,
        server_instance=None,
        batch_size: int = 32,
        image_keys: List[str] = None,
        language_key: str = "language",
        sentence_model=None,
    ):
        """
        Initialize the reward relabeling service.

        Args:
            reward_model: The Qwen reward model (must have processor and tokenizer attributes)
            exp_config: ExperimentConfig containing model and data configuration
            server_instance: Optional RewardRelabelServer instance for statistics
            batch_size: Batch size for processing subsequences (default: 32)
            image_keys: List of image observation keys to process (default: ["image"])
            language_key: Key name for language/text embeddings (default: "language")
            sentence_model: Sentence transformer model for converting text to embeddings (optional)
        """
        self.reward_model = reward_model
        self.exp_config = exp_config
        self.server_instance = server_instance
        self.batch_size = batch_size
        self.image_keys = image_keys if image_keys is not None else ["image"]
        self.language_key = language_key
        self.sentence_model = sentence_model

        # Setup reward model components
        self.processor = getattr(reward_model, "processor", None)
        self.tokenizer = getattr(reward_model, "tokenizer", None)
        if self.processor is None or self.tokenizer is None:
            raise ValueError(
                "processor and tokenizer must be available on reward_model "
                "(reward_model.processor / reward_model.tokenizer)"
            )

        # Ensure use_multi_image is True for reward relabeling
        if not self.exp_config.data.use_multi_image:
            logger.warning("use_multi_image is False in config. Setting to True for reward relabeling.")
            self.exp_config.data.use_multi_image = True

        # Set up batch collator with inference=True for evaluation
        self.batch_collator = setup_batch_collator(self.processor, self.tokenizer, self.exp_config, is_eval=True)

        # Get max_frames from config (default to 16)
        self.max_frames = self.exp_config.data.max_frames
        logger.info(f"Max frames: {self.max_frames}")

        # Statistics
        self.stats = {
            "batches_processed": 0,
            "transitions_processed": 0,
            "errors": 0,
        }

        # Debug logging counter for periodic queue size logging
        self._debug_log_counter = 0
        self._debug_log_interval = 100  # Log every 100 batches

    def RelabelRewards(self, request, context):
        """
        Relabel rewards for a full trajectory (gRPC RPC handler).

        Processes a full trajectory:
        1. Extracts frames from obs["image"] in each transition
        2. Extracts embeddings from obs (dino_embedding, text_embedding)
        3. Builds subsequences [0:1], [0:2], [0:3], ... by concatenating frames and embeddings
        4. Prepares inputs for the Qwen reward model
        5. Computes relabeled rewards using the model
        6. Returns rewards mapped back to original transitions

        Args:
            request: RelabelRewardsRequest protobuf message containing:
                - transitions: List of Transition protobuf messages (full trajectory)
                - language_instructions: List of language instruction strings (uses first element)
                - episode_ids: List of episode ID strings (uses first element)
            context: gRPC context (unused)

        Returns:
            RelabelRewardsResponse protobuf message containing:
                - rewards: List of relabeled reward values (float) for each transition
                - ok: Boolean indicating success
                - message: Status message string
        """
        start_time = time.perf_counter()
        transitions = request.transitions

        # Extract batch range or specific indices (to avoid reprocessing already-computed subsequences)
        batch_indices = None
        batch_start_idx = 0
        batch_end_idx = 0

        # For multi-stage tasks: only use frames from this index onward (0 = full trajectory)
        trajectory_start_idx = getattr(request, 'trajectory_start_idx', 0) or 0

        if hasattr(request, "batch_indices") and len(request.batch_indices) > 0:
            # DSRL mode: use specific indices
            batch_indices = list(request.batch_indices)
            logger.debug(
                f"[RewardRelabelService] DSRL mode: Processing specific batch_indices: {batch_indices} (len={len(batch_indices)}), "
                f"trajectory_start_idx={trajectory_start_idx}"
            )
        else:
            # Normal mode: use range
            batch_start_idx = request.batch_start_idx if request.batch_start_idx > 0 else 0
            batch_end_idx = request.batch_end_idx if request.batch_end_idx > 0 else len(transitions)
            logger.debug(
                f"[RewardRelabelService] Normal mode: Processing batch range: {batch_start_idx} to {batch_end_idx} (len={batch_end_idx - batch_start_idx}), "
                f"trajectory_start_idx={trajectory_start_idx}"
            )

        # Extract language instruction and episode ID (same for all transitions in a trajectory)
        language_instructions = list(request.language_instructions)
        # logger.info(f"Language instructions: {language_instructions}")
        episode_ids = list(request.episode_ids)
        # logger.info(f"Episode IDs: {episode_ids}")
        language_instruction = language_instructions[0] if language_instructions else ""
        episode_id = episode_ids[0] if episode_ids else None

        # Extract text embedding from obs (should be same for all transitions)
        first_obs = {k: bytes_to_ndarray(v.data) for k, v in transitions[0].obs.items()}

        # Try to get text embedding from observation, or compute from language_instruction string
        text_embedding = None
        if self.language_key in first_obs:
            lang_value = first_obs[self.language_key]
            # Check if it's already an embedding (1D array, likely numeric)
            if (
                isinstance(lang_value, np.ndarray)
                and lang_value.ndim == 1
                and np.issubdtype(lang_value.dtype, np.number)
            ):
                # Already an embedding array
                text_embedding = lang_value
                logger.debug(f"Using precomputed text embedding from observation key '{self.language_key}'")
            else:
                # Not a valid embedding array, fall through to use language_instruction from request
                logger.warning(
                    f"Language key '{self.language_key}' found but value is not a valid embedding array "
                    f"(shape={lang_value.shape if hasattr(lang_value, 'shape') else 'N/A'}, "
                    f"dtype={lang_value.dtype if hasattr(lang_value, 'dtype') else 'N/A'}). "
                    f"Will try to use language_instruction from request."
                )
                text_embedding = None  # Will be computed below

        # If we don't have an embedding yet, try to compute from language_instruction
        if text_embedding is None:
            if language_instruction and self.sentence_model is not None:
                # logger.info(f"Computing text embedding from language_instruction string using sentence_model.")
                text_embedding_tensor = compute_text_embeddings(language_instruction, self.sentence_model)
                text_embedding = text_embedding_tensor.cpu().numpy()
            else:
                raise ValueError(
                    f"Could not get text embedding: "
                    f"language key '{self.language_key}' {'found but not a valid embedding' if self.language_key in first_obs else 'not found in observation'} "
                    f"and cannot compute embedding (sentence_model={'provided' if self.sentence_model is not None else 'not provided'}, "
                    f"language_instruction={'provided' if language_instruction else 'not provided'}). "
                    f"Available keys: {list(first_obs.keys())}"
                )

        # Process each image key separately
        progress_predictions_by_key = {}  # Dict[str, Dict[str, List[float]]] - image_key -> {progress: [...], success_probs: [...]}

        # Check which image keys are available in observations
        available_image_keys = []
        for img_key in self.image_keys:
            if img_key in first_obs:
                available_image_keys.append(img_key)
            else:
                logger.warning(
                    f"Image key '{img_key}' not found in observation. Available keys: {list(first_obs.keys())}"
                )

        if not available_image_keys:
            raise KeyError(
                f"None of the configured image keys {self.image_keys} found in observation. "
                f"Available keys: {list(first_obs.keys())}"
            )

        # Process each available image key
        for image_key in available_image_keys:
            # Extract all frames for this image key from the full trajectory
            all_frames = []
            all_dino_embeddings = []
            use_dino_embeddings = None  # Will be set on first transition

            for i, tr in enumerate(transitions):
                # Extract observations from transition
                obs = {k: bytes_to_ndarray(v.data) for k, v in tr.obs.items()}

                # Extract frame from obs using current image_key
                current_frame = obs[image_key]
                if len(current_frame.shape) == 4:
                    current_frame = current_frame[-1]  # Take last frame if batched
                all_frames.append(current_frame)

                # Extract DINO embedding from obs if available (shared across image keys)
                if use_dino_embeddings is None:
                    # Check on first transition whether dino_embedding is available
                    use_dino_embeddings = "dino_embedding" in obs
                    # if not use_dino_embeddings:
                        # logger.info(
                        #     f"DINO embeddings not found in observations. "
                        #     f"Will compute embeddings from frames using reward model's image encoder. "
                        #     f"Available keys: {list(obs.keys())}"
                        # )

                if use_dino_embeddings:
                    dino_emb = obs["dino_embedding"]
                    if len(dino_emb.shape) > 1:
                        dino_emb = dino_emb[-1]  # Take last if batched
                    all_dino_embeddings.append(dino_emb)

            # Build subsequences: [0:1], [0:2], [0:3], ...
            # Process specific indices if provided (DSRL mode), otherwise use range
            all_samples = []

            if batch_indices is not None:
                # DSRL mode: process specific indices
                for i in batch_indices:
                    # Build subsequence from trajectory_start_idx to current step
                    # trajectory_start_idx > 0 means we only look at the current subtask
                    start = trajectory_start_idx
                    frames_subsequence = np.array(all_frames[start: i + 1])  # Shape: (i+1-start, H, W, C)

                    # Prepare raw data dict for reward model
                    raw_data = dict(
                        frames=frames_subsequence,
                        task=language_instruction,
                        id=episode_id,
                        metadata=dict(subsequence_length=len(frames_subsequence)),
                        text_embedding=text_embedding,
                    )

                    # Add video_embeddings only if DINO embeddings are available
                    if use_dino_embeddings:
                        dino_subsequence = np.array(all_dino_embeddings[start: i + 1])
                        raw_data["video_embeddings"] = dino_subsequence
                    # If not available, reward model will compute embeddings from frames

                    sample = raw_dict_to_sample(
                        raw_data=raw_data,
                        max_frames=self.max_frames,
                        sample_type="progress",
                    )
                    all_samples.append(sample)
            else:
                # Normal mode: process range
                for i in range(batch_start_idx, batch_end_idx):
                    # Build subsequence from trajectory_start_idx to current step
                    # trajectory_start_idx > 0 means we only look at the current subtask
                    start = trajectory_start_idx
                    frames_subsequence = np.array(all_frames[start: i + 1])  # Shape: (i+1-start, H, W, C)

                    # Prepare raw data dict for reward model
                    raw_data = dict(
                        frames=frames_subsequence,
                        task=language_instruction,
                        id=episode_id,
                        metadata=dict(subsequence_length=len(frames_subsequence)),
                        text_embedding=text_embedding,
                    )

                    # Add video_embeddings only if DINO embeddings are available
                    if use_dino_embeddings:
                        dino_subsequence = np.array(all_dino_embeddings[start: i + 1])
                        raw_data["video_embeddings"] = dino_subsequence
                    # If not available, reward model will compute embeddings from frames

                    sample = raw_dict_to_sample(
                        raw_data=raw_data,
                        max_frames=self.max_frames,
                        sample_type="progress",
                    )
                    all_samples.append(sample)

            # print metadata for the first sample
            logger.info(f"all_samples[0].frames.shape: {all_samples[0].trajectory.frames.shape}")
            logger.info(f"all_samples[0].frames_shape: {all_samples[0].trajectory.frames_shape}")
            logger.info(f"all_samples[0].task: {all_samples[0].trajectory.task}")

            # Process subsequences in batches for this image key
            all_success_probs = []
            all_progress = []  # Store progress predictions (rewards computed from progress by client)
            effective_batch_size = self.batch_size if self.batch_size > 0 else 32
            device = self.reward_model.device

            for batch_start in range(0, len(all_samples), effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, len(all_samples))
                batch_samples = all_samples[batch_start:batch_end]

                # Collate batch using batch collator
                batch_inputs = self.batch_collator(batch_samples)

                # Extract progress_inputs and move to device
                progress_inputs = batch_inputs["progress_inputs"]
                progress_inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in progress_inputs.items()
                }

                # Infer is_discrete_mode and num_bins from exp_config
                progress_loss_type = self.exp_config.loss.progress_loss_type
                is_discrete_mode = progress_loss_type.lower() == "discrete"
                if is_discrete_mode:
                    num_bins = self.exp_config.loss.progress_discrete_bins
                else:
                    num_bins = None

                autocast_context = nullcontext()
                if device.type == "cuda":
                    dtype_name = str(self.exp_config.model.torch_dtype).lower()
                    if dtype_name in {"bf16", "bfloat16", "torch.bfloat16"}:
                        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    elif dtype_name in {"fp16", "float16", "half", "torch.float16"}:
                        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)

                with torch.inference_mode(), autocast_context:
                    batch_outputs = compute_batch_outputs(
                        model=self.reward_model,
                        tokenizer=self.tokenizer,
                        batch_inputs=progress_inputs,
                        sample_type="progress",
                        is_discrete_mode=is_discrete_mode,
                        num_bins=num_bins,
                    )

                # Extract progress predictions (rewards will be computed from progress by client)
                # batch_outputs["progress_pred"] is a list of lists, where each inner list contains progress values for each frame
                # For each sample, we want the last progress value (progress at the end of the subsequence)
                progress_batch = []
                if "progress_pred" in batch_outputs:
                    progress_pred_raw = batch_outputs["progress_pred"]
                    # Log shape and min/max instead of full array
                    if isinstance(progress_pred_raw, list):
                        # Convert to numpy array for shape and min/max computation
                        progress_pred_array = np.array(
                            [np.array(seq) if isinstance(seq, list) else np.array([seq]) for seq in progress_pred_raw]
                        )
                        logger.debug(
                            f"[RewardRelabelService] progress_pred: shape={progress_pred_array.shape}, "
                            f"min={np.min(progress_pred_array):.4f}, max={np.max(progress_pred_array):.4f}"
                        )
                    elif isinstance(progress_pred_raw, np.ndarray):
                        logger.debug(
                            f"[RewardRelabelService] progress_pred: shape={progress_pred_raw.shape}, "
                            f"min={np.min(progress_pred_raw):.4f}, max={np.max(progress_pred_raw):.4f}"
                        )

                    for progress_seq in progress_pred_raw:
                        if isinstance(progress_seq, list) and len(progress_seq) > 0:
                            # Take the last progress value in the sequence
                            progress_batch.append(float(progress_seq[-1]))
                        else:
                            progress_batch.append(0.0)
                else:
                    # Fallback: if no progress_pred, use zeros (should not happen with progress sample_type)
                    logger.warning(f"[RewardRelabelService] No progress_pred in batch_outputs, using zeros")
                    progress_batch = [0.0] * len(batch_samples)

                # Extract success probabilities if available
                success_probs_batch = []
                if "outputs_success" in batch_outputs:
                    success_probs_raw = batch_outputs["outputs_success"]["success_probs"]
                    # Log shape and min/max before processing
                    if isinstance(success_probs_raw, list) and len(success_probs_raw) > 0:
                        success_probs_array = np.array(success_probs_raw)
                        logger.debug(
                            f"[RewardRelabelService] success_probs: shape={success_probs_array.shape}, "
                            f"min={np.min(success_probs_array):.4f}, max={np.max(success_probs_array):.4f}"
                        )
                    # Extract last value from each sequence (similar to progress_pred)
                    # success_probs_raw is a list of lists, where each inner list contains success probs for each frame
                    for success_seq in success_probs_raw:
                        if isinstance(success_seq, list) and len(success_seq) > 0:
                            # Take the last success probability in the sequence
                            success_probs_batch.append(float(success_seq[-1]))
                        else:
                            success_probs_batch.append(0.0)
                else:
                    # If no success probabilities, use zeros
                    success_probs_batch = [0.0] * len(progress_batch)

                # Convert to list of floats and accumulate
                all_success_probs.extend([float(s) for s in success_probs_batch])
                all_progress.extend([float(p) for p in progress_batch])

            # Store results for this image key (only for processed range)
            # Progress and success_probs are sent directly (no padding needed)
            progress_predictions_by_key[image_key] = {
                "progress": all_progress,  # Send directly - no padding needed
                "success_probs": all_success_probs,  # Send directly - no padding needed
            }

        # Build response with progress predictions by key
        # Note: rewards field is required by proto but unused - rewards are computed from progress by client
        response = pb.RelabelRewardsResponse(ok=True, message="ok")

        # Add progress predictions by key
        # Note: protobuf map fields return an empty message when accessed - assign fields directly
        for image_key, predictions in progress_predictions_by_key.items():
            map_entry = response.progress_predictions_by_key[image_key]
            map_entry.progress[:] = predictions["progress"]
            map_entry.success_probs[:] = predictions["success_probs"]

        elapsed_time = time.perf_counter() - start_time

        # Calculate actual number of transitions processed
        if batch_indices is not None:
            # DSRL mode: count the number of specific indices processed
            actual_batch_size = len(batch_indices)
            indices_info = f"batch_indices={batch_indices[:5]}{'...' if len(batch_indices) > 5 else ''}"
        else:
            # Normal mode: count the range
            actual_batch_size = batch_end_idx - batch_start_idx
            indices_info = f"batch_start_idx={batch_start_idx}, batch_end_idx={batch_end_idx}"

        self.stats["batches_processed"] += 1
        self.stats["transitions_processed"] += actual_batch_size  # Only count transitions actually processed

        # Periodic debug logging
        self._debug_log_counter += 1
        if self._debug_log_counter >= self._debug_log_interval:
            logger.debug(
                f"[RewardRelabelService] Stats: batches_processed={self.stats['batches_processed']}, "
                f"transitions_processed={self.stats['transitions_processed']}, errors={self.stats['errors']}"
            )
            self._debug_log_counter = 0

        # Log processing time for each batch with actual batch size (not full trajectory context)
        logger.debug(
            f"[RewardRelabelService] Batch processed: trajectory_context={len(transitions)}, "
            f"actual_batch_size={actual_batch_size} ({indices_info}), "
            f"processing_time={elapsed_time:.3f}s, "
            f"time_per_transition={elapsed_time / (actual_batch_size if actual_batch_size > 0 else 1):.4f}s"
        )

        return response

    def StreamRelabelRewards(self, request_iterator, context):
        """
        Stream-based relabeling for better throughput (gRPC streaming RPC handler).

        Processes a stream of RelabelRewardsRequest messages and yields responses.
        This allows for continuous processing of batches without closing the connection.

        Args:
            request_iterator: Iterator of RelabelRewardsRequest protobuf messages
            context: gRPC context (unused)

        Yields:
            RelabelRewardsResponse protobuf messages, one per input request
        """
        for request in request_iterator:
            response = self.RelabelRewards(request, context)
            yield response


class RoboRewardRelabelService(pb_grpc.RewardRelabelServiceServicer):
    """
    Service that relabels rewards using RoboReward baseline model.

    RoboReward predicts discrete progress scores (1-5) which are normalized to (0-1).
    This service provides the same gRPC interface as RewardRelabelService but uses
    the RoboReward baseline instead of the Robometer Qwen model.
    """

    def __init__(
        self,
        roboreward_model: "RoboReward",
        server_instance=None,
        image_keys: List[str] = None,
    ):
        """
        Initialize the RoboReward relabeling service.

        Args:
            roboreward_model: Initialized RoboReward model instance
            server_instance: Optional RewardRelabelServer instance for statistics
            image_keys: List of image observation keys to process (default: ["image"])
        """
        if not HAS_ROBOREWARD:
            raise ImportError(
                "RoboReward is not available. Please install rfm package with RoboReward dependencies."
            )

        self.roboreward_model = roboreward_model
        self.server_instance = server_instance
        self.image_keys = image_keys if image_keys is not None else ["image"]

        # Statistics
        self.stats = {
            "batches_processed": 0,
            "transitions_processed": 0,
            "errors": 0,
        }

        # Debug logging counter
        self._debug_log_counter = 0
        self._debug_log_interval = 100

    def RelabelRewards(self, request, context):
        """
        Relabel rewards using RoboReward (gRPC RPC handler).

        Processes a full trajectory using RoboReward's compute_progress method.

        Args:
            request: RelabelRewardsRequest protobuf message
            context: gRPC context (unused)

        Returns:
            RelabelRewardsResponse protobuf message
        """
        start_time = time.perf_counter()
        transitions = request.transitions

        # Extract batch range or specific indices
        batch_indices = None
        batch_start_idx = 0
        batch_end_idx = 0

        # For multi-stage tasks: only use frames from this index onward (0 = full trajectory)
        trajectory_start_idx = getattr(request, 'trajectory_start_idx', 0) or 0

        if hasattr(request, "batch_indices") and len(request.batch_indices) > 0:
            batch_indices = list(request.batch_indices)
            logger.debug(
                f"[RoboRewardRelabelService] DSRL mode: Processing specific batch_indices: {batch_indices[:5]}..., "
                f"trajectory_start_idx={trajectory_start_idx}"
            )
        else:
            batch_start_idx = request.batch_start_idx if request.batch_start_idx > 0 else 0
            batch_end_idx = request.batch_end_idx if request.batch_end_idx > 0 else len(transitions)
            logger.debug(
                f"[RoboRewardRelabelService] Normal mode: Processing batch range: {batch_start_idx} to {batch_end_idx}, "
                f"trajectory_start_idx={trajectory_start_idx}"
            )

        # Extract language instruction
        language_instructions = list(request.language_instructions)
        language_instruction = language_instructions[0] if language_instructions else ""

        # Extract observations from first transition to check available keys
        first_obs = {k: bytes_to_ndarray(v.data) for k, v in transitions[0].obs.items()}

        # Check which image keys are available
        available_image_keys = []
        for img_key in self.image_keys:
            if img_key in first_obs:
                available_image_keys.append(img_key)
            else:
                logger.warning(
                    f"Image key '{img_key}' not found in observation. Available keys: {list(first_obs.keys())}"
                )

        if not available_image_keys:
            raise KeyError(
                f"None of the configured image keys {self.image_keys} found in observation. "
                f"Available keys: {list(first_obs.keys())}"
            )

        # Process each available image key
        progress_predictions_by_key = {}

        for image_key in available_image_keys:
            # Extract all frames for this image key from the full trajectory
            all_frames = []
            for tr in transitions:
                obs = {k: bytes_to_ndarray(v.data) for k, v in tr.obs.items()}
                current_frame = obs[image_key]
                if len(current_frame.shape) == 4:
                    current_frame = current_frame[-1]  # Take last frame if batched
                all_frames.append(current_frame)

            # Determine which indices to process
            if batch_indices is not None:
                indices_to_process = batch_indices
            else:
                indices_to_process = list(range(batch_start_idx, batch_end_idx))

            # Build subsequences and get predictions
            # RoboReward works best with batched processing
            frames_list = []
            task_descriptions = []

            for i in indices_to_process:
                # Build subsequence from trajectory_start_idx to current step
                # trajectory_start_idx > 0 means we only look at the current subtask
                start = trajectory_start_idx
                frames_subsequence = np.array(all_frames[start: i + 1])
                frames_list.append(frames_subsequence)
                task_descriptions.append(language_instruction)

            # Use batched processing for efficiency
            logger.debug(
                f"[RoboRewardRelabelService] Processing {len(frames_list)} subsequences with RoboReward"
            )

            try:
                # compute_progress_batched returns List[List[float]] - one list per sample
                # Each inner list contains normalized scores (0-1) for each frame
                batch_results = self.roboreward_model.compute_progress_batched(frames_list, task_descriptions)

                # Extract the last prediction from each result (progress at end of subsequence)
                all_progress = []
                for result in batch_results:
                    if result and len(result) > 0:
                        # Take the last value (progress score for this subsequence)
                        all_progress.append(float(result[-1]))
                    else:
                        all_progress.append(0.0)

                # RoboReward doesn't predict success probabilities, use progress as proxy
                # (higher progress = higher success probability)
                all_success_probs = all_progress.copy()

            except Exception as e:
                logger.error(f"[RoboRewardRelabelService] Error in RoboReward inference: {e}")
                self.stats["errors"] += 1
                # Return zeros on error
                all_progress = [0.0] * len(indices_to_process)
                all_success_probs = [0.0] * len(indices_to_process)

            # Store results for this image key
            progress_predictions_by_key[image_key] = {
                "progress": all_progress,
                "success_probs": all_success_probs,
            }

        # Build response
        response = pb.RelabelRewardsResponse(ok=True, message="ok")
        for image_key, predictions in progress_predictions_by_key.items():
            map_entry = response.progress_predictions_by_key[image_key]
            map_entry.progress[:] = predictions["progress"]
            map_entry.success_probs[:] = predictions["success_probs"]

        elapsed_time = time.perf_counter() - start_time

        # Calculate actual number of transitions processed
        if batch_indices is not None:
            actual_batch_size = len(batch_indices)
        else:
            actual_batch_size = batch_end_idx - batch_start_idx

        self.stats["batches_processed"] += 1
        self.stats["transitions_processed"] += actual_batch_size

        # Periodic debug logging
        self._debug_log_counter += 1
        if self._debug_log_counter >= self._debug_log_interval:
            logger.debug(
                f"[RoboRewardRelabelService] Stats: batches_processed={self.stats['batches_processed']}, "
                f"transitions_processed={self.stats['transitions_processed']}, errors={self.stats['errors']}"
            )
            self._debug_log_counter = 0

        logger.debug(
            f"[RoboRewardRelabelService] Batch processed: trajectory_context={len(transitions)}, "
            f"actual_batch_size={actual_batch_size}, processing_time={elapsed_time:.3f}s"
        )

        return response

    def StreamRelabelRewards(self, request_iterator, context):
        """Stream-based relabeling for better throughput."""
        for request in request_iterator:
            response = self.RelabelRewards(request, context)
            yield response


class RewardRelabelServer:
    """
    Server that provides reward relabeling service via gRPC.

    This server hosts either a Robometer Qwen reward model or RoboReward baseline model
    and processes full trajectories to relabel rewards.
    Clients send trajectories (full episodes) and receive relabeled rewards for each transition.

    Supported reward model types:
        - "robometer" (default): Qwen-based reward model for progress and success prediction
        - "roboreward": Baseline model that predicts discrete progress scores (1-5)
    """

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 50052,
        max_workers: int = 8,
        max_msg_mb: int = 256,  # Larger for batches with frames
        batch_size: int = 32,  # Batch size for processing subsequences
        image_keys: List[str] = None,
        language_key: str = "language",
        sentence_model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        reward_model_type: str = "robometer",  # "robometer" or "roboreward"
        roboreward_max_new_tokens: int = 128,  # For RoboReward
        roboreward_use_unsloth: bool = True,  # For RoboReward
    ):
        """
        Initialize the reward relabeling server.

        Args:
            model_path: Path to reward model checkpoint (HuggingFace model ID or local path)
            host: Server host address (default: "0.0.0.0" to accept all connections)
            port: Server port number (default: 50052)
            max_workers: Maximum number of worker threads for gRPC server (default: 8)
            max_msg_mb: Maximum message size in MB (default: 256, larger for batches with frames)
            batch_size: Batch size for processing subsequences (default: 32, only for Robometer)
            image_keys: List of image keys to process (default: ["image"])
            language_key: Key name for language/text embeddings (default: "language", only for Robometer)
            sentence_model_name: Sentence transformer model name (optional, only for Robometer)
            device: Device to load model on (default: cuda if available, else cpu)
            reward_model_type: Type of reward model - "robometer" or "roboreward" (default: "robometer")
            roboreward_max_new_tokens: Max tokens for RoboReward generation (default: 128)
            roboreward_use_unsloth: Whether to use unsloth for RoboReward (default: True)
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.reward_model_type = reward_model_type
        self.host = host
        self.port = port

        logger.info(f"Initializing RewardRelabelServer with model_type={reward_model_type}")
        if image_keys:
            logger.info(f"Server will process image keys: {image_keys}")

        # Create gRPC server
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_receive_message_length", max_msg_mb * 1024 * 1024),
                ("grpc.max_send_message_length", max_msg_mb * 1024 * 1024),
            ],
        )

        if reward_model_type == "roboreward":
            # Initialize RoboReward baseline
            if not HAS_ROBOREWARD:
                raise ImportError(
                    "RoboReward is not available. Please install rfm package with RoboReward dependencies: "
                    "pip install transformers qwen-vl-utils"
                )

            logger.info(f"Loading RoboReward model from {model_path}")
            roboreward_model = RoboReward(
                model_path=model_path,
                max_new_tokens=roboreward_max_new_tokens,
                use_unsloth=roboreward_use_unsloth,
            )
            logger.success("RoboReward model loaded successfully")

            self.reward_model = roboreward_model
            self.exp_config = None  # Not used for RoboReward

            # Add RoboReward service
            self.service = RoboRewardRelabelService(
                roboreward_model=roboreward_model,
                server_instance=self,
                image_keys=image_keys if image_keys is not None else ["image"],
            )

        else:  # Default: Robometer
            # Load Robometer reward model
            logger.info(f"Loading Robometer reward model from {model_path} on {device}")

            exp_config, tokenizer, processor, reward_model = load_model_from_hf(
                model_path=model_path,
                device=device,
            )

            reward_model = reward_model.to(device)
            reward_model.eval()
            logger.success("Robometer reward model loaded successfully")

            self.reward_model = reward_model
            self.exp_config = exp_config

            # Load sentence transformer if specified
            sentence_model = None
            if sentence_model_name:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading sentence transformer model: {sentence_model_name}")
                sentence_model = SentenceTransformer(sentence_model_name)
                logger.success("Sentence transformer model loaded successfully")
            else:
                logger.info("No sentence_model specified. Server will require precomputed text embeddings in observations.")

            # Add Robometer service
            self.service = RewardRelabelService(
                reward_model=self.reward_model,
                exp_config=self.exp_config,
                server_instance=self,
                batch_size=batch_size,
                image_keys=image_keys if image_keys is not None else ["image"],
                language_key=language_key if language_key is not None else "language",
                sentence_model=sentence_model,
            )

        pb_grpc.add_RewardRelabelServiceServicer_to_server(self.service, self.server)
        self.server.add_insecure_port(f"{self.host}:{self.port}")

    def start(self):
        """
        Start the gRPC server.

        Begins listening on the configured host:port and accepts incoming connections.
        The server will continue running until stop() is called.
        """
        self.server.start()
        logger.success(f"[RewardRelabelServer] gRPC listening on {self.host}:{self.port}")

    def stop(self, grace: Optional[float] = 5.0):
        """
        Stop the gRPC server gracefully.

        Args:
            grace: Grace period in seconds to wait for ongoing requests to complete (default: 5.0)
        """
        self.server.stop(grace)

    def get_stats(self):
        """
        Get server statistics.

        Returns:
            Dictionary containing:
                - batches_processed: Total number of batches processed
                - transitions_processed: Total number of transitions processed
                - errors: Total number of errors encountered
        """
        return self.service.stats
