# Reference logging contract

The frozen preset contains the 30 scalar data keys observed in the local archived
`sf73jk43` W&B journal. Step fields are coordinates, not additional training metrics.
Nothing here changes reward arithmetic or sampling.

## Axes and cadence

- `online/policy/step`: SAC optimizer update count, logged once per update after warmup.
- `train/step`, `buffer/step`: executed low-level environment steps; logged at episode end.
- `eval/step`: executed training environment steps at evaluation, including step zero.
- W&B's automatic `_step` is a log-event counter, not any of these physical clocks.

As in the archived run, the logger does not upload `define_metric` dashboard overrides.
For comparison in the UI, select the appropriate namespace step field as the horizontal
axis for both runs and use the same smoothing. The archive has no saved panel layout:
historical bar-versus-line styling cannot be inferred from numeric history. Binary
detector flags are scalars, not histogram objects. Losses, rewards and rates can be
viewed as line plots; detector events can be displayed as points or bars in the UI.

## Data keys

| Key | Meaning |
| --- | --- |
| `buffer/avg_env_reward` | Mean raw macro reward over the current replay; normally -1. |
| `buffer/avg_predicted_progress_reward` | Mean stored RoboMeter progress over replay. |
| `buffer/avg_total_reward` | Mean stored training reward over replay, -1 + progress. |
| `buffer/total_size` | Number of macro transitions stored in replay. |
| `eval/avg_steps` | Mean executed low-level steps per evaluation episode. |
| `eval/num_eval_episodes` | Number of evaluated episodes, 20 in the reference. |
| `eval/success_rate` | Fraction achieving simulator success at least once during eval. |
| `online/policy/actor_loss` | SAC actor objective, including enabled entropy term. |
| `online/policy/actor_chosen_q_mean` | Critic estimate of current actor actions at sampled replay states. |
| `online/policy/critic_loss` | Critic TD regression loss on the replay minibatch. |
| `online/policy/ent_coef` | Learned entropy coefficient alpha. |
| `online/policy/ent_coef_loss` | Automatic entropy coefficient optimization loss. |
| `online/policy/progress_reward_mean` | Mean progress in the sampled replay minibatch. |
| `online/policy/q_values_mean` | Current critic predictions for sampled replay state/action pairs. |
| `online/policy/reward_mean` | Mean stored training reward in the sampled replay minibatch. |
| `online/policy/success_prob_mean` | Mean stored RoboMeter success probability in that minibatch. |
| `online/policy/target_q_mean` | Mean TD target, including reward and discounted target value. |
| `online/policy/train_step_total_time_s` | Recorded time of the algorithm training call. |
| `train/ep_avg_env_reward` | Accumulated macro environment reward mean since tracker reset. |
| `train/ep_avg_progress_reward` | Accumulated progress mean since tracker reset. |
| `train/ep_avg_reward` | Latest completed episode raw return for the single environment. |
| `train/ep_avg_success_prob` | Accumulated success-probability mean since tracker reset. |
| `train/ep_overall_reward` | Mean raw episode return over all completed episodes. |
| `train/ep_overall_success_rate` | Mean training detector success over completed episodes. |
| `train/ep_overall_training_reward` | Mean relabeled episode return over completed episodes. |
| `train/robometer_detected` | Whether the just-completed episode triggered either detector, 0/1. |
| `train/robometer_success_head_detected` | Success-head detection in that episode, 0/1. |
| `train/robometer_terminal_detected` | Terminal fallback detection in that episode, 0/1. |
| `train/sim_success_once` | Simulator success in that training episode, monitoring only, 0/1. |
| `train/total_steps` | Episode tracker's cumulative executed low-level steps. |

The names `ep_avg_*` do not all mean "latest episode": the inherited tracker accumulates
progress/success-probability samples. Retaining that behavior is intentional for this
reference preset. Q values are learned discounted values, not success probabilities;
replay reward means are not fresh-policy evaluation scores.

## Known compatibility boundary

The inherited training time-limit/autoreset observation handling is deliberately retained
at the user's request. A time-limit transition may include the automatically reset image;
this affects relabeling/bootstrapping at that boundary. It must be addressed in a separately
versioned experiment, not silently changed in this comparison run. Evaluation's existing
terminal-observation handling is unchanged.
