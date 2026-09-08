import os
import gymnasium as gym
from tqdm import tqdm
from loguru import logger as loguru_logger
import numpy as np

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.rollouts.rollout_worker import RolloutWorker
from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.loggers.logger import Logger
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker


class SerialRunner:
    """
    A runner that runs the environment and collects data and sends it to the buffer.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        algorithm: BaseAlgorithm,
        buffer: ReplayBuffer,
        actor: BaseAlgorithm,
        rollout_worker: RolloutWorker,
        num_rollouts: int = 1000,
        eval_freq: int = 250,
        reward_model=None,
        logger: Logger = None,
        eval_kwargs: dict = None,
        save_dir: str = None,
        save_interval: int = None,
        evaluation_worker_class: type[EvaluationWorker] = None,
        eval_at_episode_boundary: bool = False,
        eval_on_first_step: bool = False,
        # Buffer saving options
        save_buffer_on_exit: bool = False,
        save_buffer_every: int = 0,
        save_buffer_images: bool = False,
        save_buffer_image_keys: list = None,
        save_buffer_dir: str = "replay_buffers",
        train_after_episode: bool = False,
    ):
        """
        Args:
            env: Training environment
            eval_env: Evaluation environment (can be same as env for real robots)
            algorithm: RL algorithm
            buffer: Replay buffer
            actor: Actor model
            rollout_worker: Rollout worker for data collection
            num_rollouts: Maximum number of environment steps
            eval_freq: Evaluate every N steps (will wait for episode boundary if eval_at_episode_boundary=True)
            reward_model: Optional reward model
            logger: Logger instance
            eval_kwargs: Additional evaluation kwargs
            save_dir: Directory for checkpoints
            save_interval: Save checkpoint every N steps
            evaluation_worker_class: Custom evaluation worker class
            eval_at_episode_boundary: If True, only evaluate when an episode completes.
                                      This is important for real robots to avoid interrupting trajectories.
                                      When eval_env is the same object as env, this is automatically enabled.
            eval_on_first_step: If True, run evaluation before any training begins (at step 0).
        """
        self.env = env
        self.eval_env = eval_env
        self.algorithm = algorithm
        self.buffer = buffer
        self.actor = actor
        self.rollout_worker = rollout_worker
        self.num_rollouts = num_rollouts
        self.eval_freq = eval_freq
        self.reward_model = reward_model
        self.logger = logger

        # Evaluation settings
        self.eval_kwargs = eval_kwargs or {}

        # If eval_env is the same as env (e.g., real robot), force episode boundary evaluation
        # to avoid interrupting ongoing trajectories
        self.eval_at_episode_boundary = eval_at_episode_boundary or (eval_env is env)
        if self.eval_at_episode_boundary:
            print("Evaluation will only occur at episode boundaries (eval_at_episode_boundary=True)")

        # Whether to run evaluation before any training begins
        self.eval_on_first_step = eval_on_first_step

        # Track if evaluation is pending (waiting for episode boundary)
        self._eval_pending = False

        # Checkpointing settings
        self.save_dir = save_dir
        self.save_interval = save_interval
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        # Buffer saving settings
        self.save_buffer_on_exit = save_buffer_on_exit
        self.save_buffer_every = save_buffer_every
        self.save_buffer_images = save_buffer_images
        self.save_buffer_image_keys = save_buffer_image_keys or ["image"]
        self.save_buffer_dir = save_buffer_dir
        self._last_buffer_save_steps = 0
        self.train_after_episode = train_after_episode
        
        if self.save_buffer_on_exit or self.save_buffer_every > 0:
            os.makedirs(self.save_buffer_dir, exist_ok=True)
            loguru_logger.info(f"Buffer saving enabled: on_exit={save_buffer_on_exit}, every={save_buffer_every} steps")
        if self.train_after_episode:
            loguru_logger.info("Training is enabled only after episode completion.")

        # Initialize evaluation worker if a custom one is not provided
        if evaluation_worker_class is None:
            self.evaluation_worker = EvaluationWorker(
                eval_env=eval_env,
                device=rollout_worker.device,
                logger=logger,
                **self.eval_kwargs,
            )
        else:
            self.evaluation_worker = evaluation_worker_class(
                eval_env=eval_env,
                device=rollout_worker.device,
                logger=logger,
                **self.eval_kwargs,
            )

        # Episode tracking
        self.total_episodes_completed = 0
        self.previous_total_episodes = 0
        self.total_env_steps = 0

    def run(self):
        # Create progress bar for training rollouts
        train_metrics = None
        rollout_metrics = None
        reward_metrics = None

        # Track last evaluation and save points (based on env steps)
        last_eval_steps = 0
        last_save_steps = 0

        # NOTE:
        # - self.rollout_worker.num_rollouts controls how many steps/episodes a *single*
        #   rollout_worker.run() call collects.
        # - self.num_rollouts is interpreted as a *maximum number of environment steps*
        #   for the overall training run (see configs: "this is more like env steps").
        max_env_steps = self.num_rollouts

        # Evaluate on first step if configured (before any training)
        if self.eval_on_first_step and self.eval_env is not None:
            loguru_logger.info("Running initial evaluation before training...")
            eval_metrics = self.evaluate(self.actor)
            if self.logger is not None:
                self.logger.log(eval_metrics, step=0, prefix="eval")
            self._print_detailed_eval_metrics(eval_metrics)

        with tqdm(
            total=max_env_steps,
            desc="Training Rollouts",
            unit="step",
        ) as pbar:
            current_rollouts = 0
            prev_steps_this_episode = 0
            # Stop when we reach the environment step budget.
            while self.total_env_steps < max_env_steps:
                # Check if we have enough data to start training
                buffer_size = len(self.buffer)
                learning_starts = (
                    getattr(self.algorithm.config, "learning_starts", 0) if self.algorithm is not None else 0
                )
                # can_train = buffer_size >= learning_starts
                can_train = self.total_env_steps >= learning_starts
                # collect environment rollouts
                rollout_metrics = self.rollout_worker.run(can_train=can_train)
                # check again
                buffer_size = len(self.buffer)
                learning_starts = (
                    getattr(self.algorithm.config, "learning_starts", 0) if self.algorithm is not None else 0
                )
                # can_train = buffer_size >= learning_starts
                can_train = self.total_env_steps >= learning_starts

                #if rollout_metrics and "num_episodes" in rollout_metrics and rollout_metrics["num_episodes"] > 0:
                #loguru_logger.info(f"total episodes: {self.rollout_worker.total_episodes}")
                self.total_episodes_completed = self.rollout_worker.total_episodes

                # Track total environment steps and steps in this iteration
                steps_this_iter = 0
                if rollout_metrics and "total_steps" in rollout_metrics:
                    prev_steps = self.total_env_steps
                    self.total_env_steps = rollout_metrics["total_steps"]
                    steps_this_iter += self.total_env_steps - prev_steps

                episodes_actually_completed = self.total_episodes_completed > self.previous_total_episodes
                self.previous_total_episodes = self.total_episodes_completed

                # Debug: log when we first reach learning_starts
                if can_train and not hasattr(self, "_training_started_logged"):
                    loguru_logger.info(
                        f"✓ Total env steps {self.total_env_steps} >= {learning_starts} (learning starts), starting training!"
                    )
                    self._training_started_logged = True

                # train the actor with the algorithm (only if we have enough data)
                train_updates = 0
                if self.train_after_episode:
                    # Only train after an episode completes, using steps from that episode
                    if can_train and episodes_actually_completed and steps_this_iter > 0:
                        train_updates = self.total_env_steps - prev_steps_this_episode
                        prev_steps_this_episode = self.total_env_steps
                        if hasattr(self.rollout_worker, "action_exec_len"):
                            train_updates = train_updates // self.rollout_worker.action_exec_len
                else:
                    if can_train:
                        train_updates = 1

                if self.algorithm is not None and train_updates > 0:
                    train_metrics = None
                    if self.train_after_episode:
                        loguru_logger.info(f"Training {train_updates} updates")
                    for _ in range(train_updates):
                        train_metrics = self.algorithm.train_step(logging_prefix="online/policy")
                else:
                    train_metrics = None

                # train the reward model (only if we have enough data)
                #if self.reward_model is not None and train_updates > 0:
                #    reward_metrics = None
                #    for _ in range(train_updates):
                #        reward_metrics = self.reward_model.online_train_step(logging_prefix="online/reward_model")
                #else:
                #    reward_metrics = None

                # update the actor with the algorithm (only if we trained)
                if train_updates > 0:
                    self.rollout_worker.update_actor(self.actor)
                    if self.reward_model is not None:
                        self.buffer.update_reward_model(self.reward_model)

                # Log buffer statistics (including async reward relabeling stats)
                if self.logger is not None and episodes_actually_completed:
                    buffer_stats = self._get_buffer_stats()
                    if buffer_stats:
                        self.logger.log(buffer_stats, step=self.total_env_steps, prefix="buffer")

                # Log rollout metrics to wandb
                if self.logger is not None and rollout_metrics and episodes_actually_completed:
                    self.logger.log(rollout_metrics, step=self.total_env_steps, prefix="train")

                current_rollouts += 1

                # Update progress bar with training metrics
                postfix_dict = {
                    "buffer_size": buffer_size,
                    "episodes": self.total_episodes_completed,
                    "env_steps": self.total_env_steps,
                }

                # Add training status indicator
                if not can_train:
                    # Show both buffer size (transitions) and env steps for clarity
                    postfix_dict["status"] = f"collecting (buffer size:{buffer_size}/{learning_starts})"
                else:
                    postfix_dict["status"] = "✓ training"

                # Add rollout metrics if available
                if rollout_metrics:
                    postfix_dict.update(
                        {
                            k: f"{v:.2f}"
                            for k, v in rollout_metrics.items()
                            if k in ["ep_avg_reward", "ep_avg_progress_reward", "success_rate"]
                        }
                    )

                if train_metrics:
                    postfix_dict.update(
                        {
                            k: f"{v:.4f}"
                            for k, v in train_metrics.items()
                            if k in ["q_values_mean", "actor_loss", "ent_coef"]
                        }
                    )

                pbar.set_postfix(postfix_dict)
                # Update progress bar by actual environment steps collected
                pbar.update(steps_this_iter if steps_this_iter > 0 else self.rollout_worker.num_rollouts)

                # Evaluate the actor (based on environment steps)
                # Check if we should schedule an evaluation
                should_schedule_eval = False
                if self.eval_env is not None:
                    # Schedule eval based on step frequency
                    if self.eval_freq > 0 and (self.total_env_steps - last_eval_steps >= self.eval_freq):
                        should_schedule_eval = True

                if should_schedule_eval:
                    if self.eval_at_episode_boundary:
                        # Mark evaluation as pending - will execute when episode completes
                        if not self._eval_pending:
                            loguru_logger.info(
                                f"[EVAL] Evaluation scheduled at step {self.total_env_steps}, waiting for episode boundary..."
                            )
                        self._eval_pending = True
                    else:
                        # Execute evaluation immediately (original behavior)
                        self._eval_pending = True

                # Execute pending evaluation
                # If eval_at_episode_boundary is True, only eval when an episode actually completed
                should_eval_now = self._eval_pending and (
                    not self.eval_at_episode_boundary or episodes_actually_completed
                )

                # Debug: Log why evaluation isn't happening
                if self._eval_pending and not should_eval_now:
                    loguru_logger.debug(
                        f"[EVAL] Pending but not executing: eval_at_episode_boundary={self.eval_at_episode_boundary}, episodes_completed={episodes_actually_completed}"
                    )

                if should_eval_now:
                    loguru_logger.info(
                        f"Starting evaluation at step {self.total_env_steps} (episode {self.total_episodes_completed})"
                    )
                    eval_metrics = self.evaluate(self.actor)
                    if self.logger is not None:
                        self.logger.log(eval_metrics, step=self.total_env_steps, prefix="eval")
                    self._print_detailed_eval_metrics(eval_metrics)
                    last_eval_steps = self.total_env_steps
                    self._eval_pending = False

                    # If eval_env is the same as training env (e.g., real robot),
                    # force reset the rollout worker state since evaluation used the same env
                    if self.eval_env is self.env:
                        loguru_logger.info("[EVAL] Resetting rollout worker state after evaluation (shared env)")
                        if hasattr(self.rollout_worker, "force_reset"):
                            self.rollout_worker.force_reset()
                        else:
                            # Fallback for rollout workers without force_reset
                            self.rollout_worker.recent_obs = None
                            if hasattr(self.rollout_worker, "needs_reset"):
                                self.rollout_worker.needs_reset = [True] * self.rollout_worker.num_envs

                # Periodic checkpointing and artifact logging (based on environment steps)
                should_save = False
                if self.save_dir is not None and self.save_interval is not None and self.save_interval > 0:
                    if self.total_env_steps - last_save_steps >= self.save_interval:
                        should_save = True

                if should_save:
                    self._save_checkpoint(
                        tag=self.total_env_steps,
                        metadata={
                            "env_steps": self.total_env_steps,
                            "rollouts_completed": current_rollouts,
                            "episodes": self.total_episodes_completed,
                            "algo_step": getattr(self.algorithm, "step_counter", None),
                        },
                    )
                    last_save_steps = self.total_env_steps

                # Periodic buffer saving (overwrites same file each time)
                if self.save_buffer_every > 0:
                    if self.total_env_steps - self._last_buffer_save_steps >= self.save_buffer_every:
                        self._save_buffer(tag="latest", use_timestamp=False)
                        self._last_buffer_save_steps = self.total_env_steps

                # Print detailed metrics only when episodes were actually completed
                if episodes_actually_completed:
                    self._print_detailed_metrics(rollout_metrics, train_metrics)

            # Final checkpoint at the end
            if self.save_dir is not None:
                self._save_checkpoint(
                    tag="final",
                    metadata={
                        "env_steps": self.total_env_steps,
                        "rollouts_completed": current_rollouts,
                        "episodes": self.total_episodes_completed,
                        "algo_step": getattr(self.algorithm, "step_counter", None),
                    },
                )

            # Final buffer save at the end (unique filename with timestamp)
            if self.save_buffer_on_exit:
                self._save_buffer(tag="final", use_timestamp=True)

    def _print_detailed_metrics(self, rollout_metrics, train_metrics):
        """Print detailed metrics."""
        loguru_logger.info(f"\n--- Episode {self.total_episodes_completed} Metrics ---")

        if rollout_metrics:
            loguru_logger.info(f"Average Reward: {rollout_metrics.get('ep_avg_reward', 0):.3f}")
            loguru_logger.info(
                f"Min/Max Reward: {rollout_metrics.get('ep_min_reward', 0):.3f} / {rollout_metrics.get('ep_max_reward', 0):.3f}"
            )
            loguru_logger.info(f"Success Rate: {rollout_metrics.get('success_rate', 0):.1%}")
            if "ep_avg_progress_reward" in rollout_metrics:
                loguru_logger.info(
                    f"Avg Progress (Relabeled) Reward: {rollout_metrics.get('ep_avg_progress_reward', 0):.3f}"
                )
            if "ep_avg_env_reward" in rollout_metrics:
                loguru_logger.info(
                    f"Avg Env Reward: {rollout_metrics.get('ep_avg_env_reward', 0):.3f}"
                )

        if train_metrics:
            loguru_logger.info(f"[TRAIN] Critic Loss: {train_metrics.get('critic_loss', 0):.6f}")
            loguru_logger.info(f"[TRAIN] Actor Loss: {train_metrics.get('actor_loss', 0):.6f}")
            loguru_logger.info(f"[TRAIN] Entropy Coef: {train_metrics.get('ent_coef', 0):.6f}")

        loguru_logger.info("---")  # Separator line

    def _print_detailed_eval_metrics(self, eval_metrics):
        """Print detailed evaluation metrics."""
        loguru_logger.info(f"[EVAL] Avg Reward: {eval_metrics.get('avg_reward', 0):.3f}")
        loguru_logger.info(f"[EVAL] Std Reward: {eval_metrics.get('std_reward', 0):.3f}")
        loguru_logger.info(f"[EVAL] Min Reward: {eval_metrics.get('min_reward', 0):.3f}")
        loguru_logger.info(f"[EVAL] Max Reward: {eval_metrics.get('max_reward', 0):.3f}")
        loguru_logger.info(f"[EVAL] Success Rate: {eval_metrics.get('success_rate', 0):.1%}")
        loguru_logger.info(f"[EVAL] Avg Steps: {eval_metrics.get('avg_steps', 0):.1f}")
        loguru_logger.info("---")  # Separator line

    def evaluate(self, actor: BaseActor):
        """Evaluate the actor using the evaluation worker."""
        return self.evaluation_worker.run(actor)

    def _get_buffer_stats(self) -> dict:
        """Get buffer statistics including average predicted progress reward from buffer transitions."""
        stats = {
            "buffer/total_size": len(self.buffer),
        }

        # Add buffer-specific stats if available
        if hasattr(self.buffer, "get_statistics"):
            try:
                buffer_stats = self.buffer.get_statistics()
                for key, value in buffer_stats.items():
                    if isinstance(value, (int, float)):
                        stats[f"buffer/{key}"] = value
            except Exception:
                pass

        # Compute average predicted progress reward and total reward stats from buffer transitions
        # The relabeled_reward is stored in transition.info["relabeled_reward"]
        # The total reward (env_reward + progress) is in transition.reward
        all_transitions = self.buffer.get_all_transitions()
        progress_rewards = []
        total_rewards = []
        env_rewards = []

        for transition in all_transitions:
            # Collect total reward (combined env + progress)
            if transition.reward is not None:
                total_rewards.append(float(transition.reward))
            
            # Collect progress and env rewards from info if available
            if transition.info is not None:
                relabeled_reward = transition.info.get("relabeled_reward")
                if relabeled_reward is not None:
                    progress_rewards.append(float(relabeled_reward))
                
                env_reward = transition.info.get("env_reward")
                if env_reward is not None:
                    env_rewards.append(float(env_reward))

        # Log progress reward stats
        if progress_rewards:
            stats["buffer/avg_predicted_progress_reward"] = float(np.mean(progress_rewards))
            stats["buffer/min_predicted_progress_reward"] = float(np.min(progress_rewards))
            stats["buffer/max_predicted_progress_reward"] = float(np.max(progress_rewards))
            stats["buffer/num_progress_rewards"] = len(progress_rewards)
        
        # Log total reward stats (env + progress combined)
        if total_rewards:
            stats["buffer/avg_total_reward"] = float(np.mean(total_rewards))
            stats["buffer/min_total_reward"] = float(np.min(total_rewards))
            stats["buffer/max_total_reward"] = float(np.max(total_rewards))
            stats["buffer/std_total_reward"] = float(np.std(total_rewards))
        
        # Log env reward stats
        if env_rewards:
            stats["buffer/avg_env_reward"] = float(np.mean(env_rewards))
            stats["buffer/min_env_reward"] = float(np.min(env_rewards))
            stats["buffer/max_env_reward"] = float(np.max(env_rewards))
        
        # Log relabeling ratio (% of transitions that have been relabeled)
        if len(all_transitions) > 0:
            relabeled_count = len(progress_rewards)
            total_count = len(all_transitions)
            stats["buffer/relabeling_ratio"] = float(relabeled_count / total_count)
            stats["buffer/num_relabeled"] = relabeled_count
            stats["buffer/num_total"] = total_count

        return stats

    def _save_buffer(self, tag: str = "buffer", use_timestamp: bool = False) -> None:
        """Save replay buffer to npz file.
        
        Args:
            tag: Tag for the filename (e.g., "latest", "final", "signal_exit")
            use_timestamp: If True, include timestamp in filename (for unique saves like final)
        """
        from datetime import datetime
        
        if self.buffer is None or len(self.buffer) == 0:
            loguru_logger.warning("Buffer is empty, skipping save")
            return
        
        # Generate filename - periodic saves overwrite, final/signal saves are unique
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_buffer_dir, f"buffer_{tag}_{timestamp}.npz")
        else:
            save_path = os.path.join(self.save_buffer_dir, f"buffer_{tag}.npz")
        
        loguru_logger.info(f"Saving buffer to {save_path}...")
        loguru_logger.info(f"  - Buffer size: {len(self.buffer)}")
        loguru_logger.info(f"  - Episodes: {self.total_episodes_completed}")
        loguru_logger.info(f"  - Env steps: {self.total_env_steps}")
        loguru_logger.info(f"  - Save images: {self.save_buffer_images}")
        
        try:
            self.buffer.save_to_npz(
                save_path,
                save_images=self.save_buffer_images,
                image_keys=self.save_buffer_image_keys if self.save_buffer_images else None,
            )
            loguru_logger.info(f"Buffer saved successfully to {save_path}")
        except Exception as e:
            loguru_logger.error(f"Failed to save buffer: {e}")
            import traceback
            traceback.print_exc()

    def save_buffer_on_signal(self) -> None:
        """Public method to save buffer (can be called from signal handler)."""
        if self.save_buffer_on_exit and self.buffer is not None and len(self.buffer) > 0:
            self._save_buffer(tag="signal_exit", use_timestamp=True)

    def _save_checkpoint(self, tag, metadata: dict | None = None) -> None:
        """Save algorithm state and log as an artifact if a logger is available."""
        if self.save_dir is None:
            return
        checkpoint_dir = os.path.join(self.save_dir, str(tag))
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.algorithm.save(checkpoint_dir)

        if self.logger is not None:
            try:
                aliases = ["latest"]
                try:
                    aliases.append(f"rollout-{int(tag)}")
                except Exception:
                    aliases.append(str(tag))
                self.logger.log_artifact(
                    path=checkpoint_dir,
                    name=f"{self.logger.exp_name}-{tag}",
                    type="checkpoint",
                    metadata=metadata or {},
                    aliases=aliases,
                )
            except Exception:
                # Never break training due to artifact failures
                pass
