#!/usr/bin/env python
"""
Multi-stage remote server for real DROID robot that interfaces with DSRL training.
Supports sequential prompts that advance on success, with per-prompt step counting.

Based on Physical Intelligence's official DROID example:
https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py

Uses standard TCP sockets that work with Pinggy and other tunneling services.
Includes keyboard input for marking success/failure and episode timeouts.

This file extends droid_remote_server.py with multi-stage prompt support.

Prerequisites:
1. Install droid package: https://github.com/droid-dataset/droid
2. Install openpi_client: pip install openpi-client
3. Configure camera IDs in command line args

Example usage:
# Terminal 1: Start robot server with multi-stage prompts
python robots/droid_remote_server_multi_stage.py \
    --left-camera-id "24259877" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13062452" \
    --external-camera left \
    --server-port 6000 \
    --prompt-file prompts.txt \
    --max-steps 40

# prompts.txt contains:
# pick up the red block
# place the block on the shelf
# close the drawer

# Terminal 2: Tunnel with Pinggy
ssh -p 443 -R0:localhost:6000 a.pinggy.io

# Terminal 3: Start DSRL training with remote env
python scripts/train_dsrl.py \
    env_name="DROID_remote" \
    remote_env_url="tcp://pinggy-url:port" \
    pi0_checkpoint=/path/to/pi0 \
    ...

Keyboard Controls (in Terminal 1):
  ENTER - Start episode
  's' - Mark current stage as SUCCESS (advances to next prompt)
  'f' - Mark episode as FAILURE (ends episode)
  'c' - Confirm advance to next stage (during transition)
  'q' - Quit server
"""

import argparse
import time
import numpy as np
import socket
import pickle
import struct
from typing import Dict, Any, List, Optional
import threading
import sys
from robometer_policy_learning.robots.droid_remote_server import extract_observation
import select
import os

# Import shared utilities
import importlib.util
spec = importlib.util.spec_from_file_location(
    "remote_server_utils", 
    os.path.join(os.path.dirname(__file__), "remote_server_utils.py")
)
remote_server_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_server_utils)

# Use shared utilities
MultiStageEpisodeState = remote_server_utils.MultiStageEpisodeState
keyboard_listener = remote_server_utils.keyboard_listener
send_msg = remote_server_utils.send_msg
recv_msg = remote_server_utils.recv_msg

# DROID specific imports
try:
    from droid.robot_env import RobotEnv
    from openpi_client import image_tools
    from PIL import Image
    DROID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DROID environment not found: {e}")
    print("Robot functionality will not work. Install with:")
    print("  pip install droid")
    print("  pip install openpi-client")
    DROID_AVAILABLE = False

# DROID data collection frequency (from official example)
DROID_CONTROL_FREQUENCY = 15 # num times per second
MIN_SUCCESS_CHECK_THRESHOLD = 8 # need to execute at least this many action chunks after the previous success check


def load_prompts_from_file(filepath: str) -> List[str]:
    """
    Load prompts from a text file (one prompt per line).
    
    Args:
        filepath: Path to text file containing prompts
        
    Returns:
        List of prompt strings (stripped, empty lines skipped)
    """
    prompts = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    prompts.append(line)
        
        if not prompts:
            raise ValueError(f"No prompts found in {filepath}")
        
        print(f"✓ Loaded {len(prompts)} prompt(s) from {filepath}")
        for i, prompt in enumerate(prompts, 1):
            print(f"  Stage {i}: {prompt}")
        
        return prompts
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading prompts from {filepath}: {e}")


def init_robot(left_camera_id: str, right_camera_id: str, wrist_camera_id: str, 
               external_camera: str = "left"):
    """
    Initialize connection to DROID robot.
    
    Args:
        left_camera_id: Left camera serial number
        right_camera_id: Right camera serial number
        wrist_camera_id: Wrist camera serial number
        external_camera: Which external camera to use ("left" or "right")
    
    Returns:
        env: DROID RobotEnv instance
        camera_config: Dict with camera IDs
    """
    if not DROID_AVAILABLE:
        raise RuntimeError(
            "DROID not installed. Please install:\n"
            "  pip install droid\n"
            "  pip install openpi-client"
        )
    
    print(f"Connecting to DROID robot...")
    print(f"  Left camera: {left_camera_id}")
    print(f"  Right camera: {right_camera_id}")
    print(f"  Wrist camera: {wrist_camera_id}")
    print(f"  Using {external_camera} camera for policy")
    
    # Initialize the Panda environment
    # Using joint velocity action space is very important (from official example)
    env = RobotEnv(
        action_space="joint_velocity",
        gripper_action_space="position"
    )
    
    print("✓ DROID robot connected")
    
    camera_config = {
        'left_camera_id': left_camera_id,
        'right_camera_id': right_camera_id,
        'wrist_camera_id': wrist_camera_id,
        'external_camera': external_camera,
    }
    
    return env, camera_config


def handle_prompt_transition(
    env: RobotEnv,
    episode_state: MultiStageEpisodeState,
    prompts: List[str],
    current_prompt_idx: int,
    max_steps: Optional[int],
    reset_between_stages: bool = False,
    reset_function: str = "reset",
) -> int:
    """
    Handle transition to next prompt after success.
    
    Args:
        episode_state: Multi-stage episode state
        prompts: List of all prompts
        current_prompt_idx: Current prompt index
        max_steps: Max steps per prompt
        
    Returns:
        New prompt index (loops to 0 if at end)
    """
    num_prompts = len(prompts)
    is_last_prompt = (current_prompt_idx == num_prompts - 1)
    
    print("\n" + "="*60)
    print(f"✓ Stage {current_prompt_idx + 1}/{num_prompts} completed!")
    print(f"  Completed: {prompts[current_prompt_idx]}")
    
    if is_last_prompt:
        # This should not happen - last stage should end episode instead of calling this function
        raise ValueError(
            "handle_prompt_transition called for last stage - episode should end instead. "
            "This indicates a bug in the calling code."
        )
    else:
        print(f"\n➡️  Next stage: {prompts[current_prompt_idx + 1]}")
        next_idx = current_prompt_idx + 1
    
    # Advance to next prompt
    print(f"\n✓ Advancing to stage {next_idx + 1}/{num_prompts}")
    print(f"  New prompt: {prompts[next_idx]}")
    
    if reset_between_stages:
        print("Resetting DROID robot for new stage...")
        if reset_function == "reset_rewardfm_partial":
            env.reset_rewardfm_partial()
        elif reset_function == "reset_rewardfm":
            env.reset_rewardfm()
        else:  # fallback to reset if invalid function
            env.reset()
        #time.sleep(0.5)  # Allow robot to settle
        raw_obs = env.get_observation()
        save_image = True  # Save image
    else:
        print("Continuing to next stage...")
        # Stage continuation without reset: don't reset robot, just get current observation
        raw_obs = env.get_observation()
        save_image = False  # Don't save image for continuation
    
    # Reset per-prompt step counter
    episode_state.reset_prompt_step_count()

    with episode_state.lock:
        episode_state.step_count = 0
    
    
    # Clear success flag for next stage
    episode_state.success = None


    current_prompt = prompts[next_idx]
    return current_prompt, next_idx, raw_obs, save_image


def handle_client(conn, addr, env, camera_config, prompts: List[str], 
                  max_steps: Optional[int], 
                  resolution: int, episode_state: MultiStageEpisodeState,
                  reset_between_stages: bool = False, reset_function: str = "reset"):
    """Handle a single client connection with keyboard input support."""
    print(f"Handling client from {addr}")
    num_actions_since_last_success_check = 0

    def apply_operator_requested_reset() -> bool:
        """Apply reset requested from keyboard thread in this main loop thread."""
        if episode_state.consume_env_reset_request():
            if prompt_idx < len(prompts) - 1:
                return False
            print("\nApplying operator-requested reset...")
            env.reset()
            time.sleep(0.5)  # Allow robot to settle before next observation
            return True
        return False

    try:
        while True:
            cmd = recv_msg(conn)
            if cmd is None:
                print("Client disconnected")
                break
            
            if cmd['type'] == 'RESET':
                print("\n" + "="*60)
                print("="*60)
                
                # Reset episode state
                episode_state.reset()
                
                # Reset prompt index to start from first stage
                prompt_idx = 0
                
                # Initialize variables for reset handling
                save_image = False
                print("Resetting DROID robot for new episode...")
                env.reset()
                time.sleep(0.5)  # Allow robot to settle
                raw_obs = env.get_observation()
                save_image = True  # Save image
                wait_for_enter = True  # Wait for Enter on new episodes
                
                # Format observation
                current_prompt = prompts[prompt_idx]
                formatted_obs = extract_observation(
                    raw_obs,
                    camera_config,
                    current_prompt,
                    resolution,
                    save_to_disk=save_image
                )

                if (formatted_obs['observation/exterior_image_1_left'] == 0).all():
                    print("Error: Exterior image is all zeros")
                    sys.exit(1)
                
                print(f"\nStage {prompt_idx + 1}/{len(prompts)}: {current_prompt}")
                print(f"Max steps per stage: {max_steps if max_steps else 'unlimited'}")
                
                # Wait for operator to press Enter before starting episode (only for new episodes)
                if wait_for_enter:
                    print("\n" + "─"*60)
                    print("🤖 Robot is ready. Set up the scene if needed.")
                    print("─"*60)
                    input(">>> Press ENTER to start episode...")
                    print("─"*60)
                    print("Episode starting! Press 's' for success, 'f' for failure")
                    print("─"*60 + "\n")
                else:
                    print("\n" + "─"*60)
                    print("Episode continuing! Press 's' for success, 'f' for failure")
                    print("─"*60 + "\n")
                
                # Advertise server capabilities
                info = {
                    'supports_action_chunking': True,
                    'stage': prompt_idx + 1,
                    'total_stages': len(prompts),
                    'prompt': current_prompt,
                }
                
                # Send back to client (epigsode starts now)
                print(f"prompt for send_msg {formatted_obs['prompt']}")
                send_msg(conn, {**formatted_obs, 'info': info})

            elif cmd['type'] == 'SUCCESS_CHECK':
                print("Received SUCCESS_CHECK command")
                blocked = False
                if num_actions_since_last_success_check < MIN_SUCCESS_CHECK_THRESHOLD:
                    print(f"Blocking success check at {num_actions_since_last_success_check} since last. Need to execute at least {MIN_SUCCESS_CHECK_THRESHOLD} actions.")
                    num_actions_since_last_success_check += 1
                    has_stages_left = True
                    raw_obs = env.get_observation()
                    blocked = True
                else:
                    print(f"Success check allowed at {num_actions_since_last_success_check} since last. Executing success check.")
                    num_actions_since_last_success_check = 0
                    # Check if the current stage is the last stage
                    has_stages_left = (prompt_idx < len(prompts) - 1)
                    # if it's not the last stage, advance to the next stage
                    if has_stages_left:
                        print("Advancing to next stage")
                        current_prompt, prompt_idx, raw_obs, save_image = handle_prompt_transition(
                            env, episode_state, prompts, prompt_idx, max_steps, reset_between_stages, reset_function
                        )

                # last stage: end episode  
                if not has_stages_left:
                    # don't reset here because client will send RESET next
                    print("Last stage completed - ending episode")
                    send_msg(conn, {"done": True})
                # not last stage: send formatted observation to client and continue episode
                else:
                    print("Sending formatted observation to client and continuing episode")
                    formatted_obs = extract_observation(
                        raw_obs,
                        camera_config,
                        current_prompt,
                        resolution,
                        save_to_disk=save_image
                    )
                    send_msg(conn, {**formatted_obs, 'done': False, 'blocked': blocked})
                    print(f"prompt for send_msg {formatted_obs['prompt']}")

            elif cmd['type'] == 'STEP':
                num_actions_since_last_success_check += 1
                action = cmd['action']
                need_obs = cmd.get('need_obs', True)
                
                # Convert to numpy and detect if this is a chunk
                action = np.array(action, dtype=np.float32)
                is_chunk = (action.ndim == 2)  # Single: (8,), Chunk: (N, 8)
                
                if is_chunk:
                    # Execute multiple actions (chunked execution)
                    actions = action
                    
                    print(f"\n[CHUNK] Executing {len(actions)} actions...")
                    
                    num_steps_executed = 0
                    done = False
                    truncated = False
                    success = False
                    formatted_obs = None
                    
                    try:
                        for i, single_action in enumerate(actions):
                            start_time = time.time()
                            
                            episode_state.increment_step()  # Increments both episode and prompt step counters
                            num_steps_executed += 1
                            current_step = episode_state.get_step_count()
                            prompt_step = episode_state.get_prompt_step_count()
                            
                            # Binarize gripper
                            if single_action[-1] > 0.5:
                                single_action = np.concatenate([single_action[:-1], np.ones((1,))])
                            else:
                                single_action = np.concatenate([single_action[:-1], np.zeros((1,))])
                            
                            # Clip action
                            single_action = np.clip(single_action, -1, 1)
                            
                            # Execute action
                            env.step(single_action)
                            
                            # Check episode status (with per-prompt timeout)
                            done, truncated, success, status_info = episode_state.get_status(
                                prompt_max_steps=max_steps
                            )

                            # Apply reset in main thread if keyboard requested it.
                            # If this was a manual reset (no terminal status), stop this
                            # chunk early and return fresh observation.
                            if apply_operator_requested_reset():
                                if not (done or truncated):
                                    status_info["manual_reset"] = True
                                break
                            
                            # Print progress
                            is_last = (i == len(actions) - 1)
                            stage_info = f"Stage {prompt_idx + 1}/{len(prompts)}"
                            print(f"  Step {current_step} (prompt: {prompt_step}) [{stage_info}] ({i+1}/{len(actions)})", end="")
                            
                            if done or truncated:
                                status_str = " - SUCCESS" if success else " - FAILURE"
                                print(status_str)
                                break
                            elif is_last:
                                print(" (chunk done)")
                            else:
                                print()
                            
                            # Match DROID frequency
                            elapsed_time = time.time() - start_time
                            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                        
                        # Handle timeout confirmation
                        if truncated and status_info.get('timeout', False):
                            print("\n" + "─"*60)
                            print("⏱️  TIMEOUT reached! Was the task or subtask successful?")
                            print("─"*60)
                            print(">>> Press 's' for SUCCESS or 'f' for FAILURE")
                            print("─"*60)
                            
                            while episode_state.success is None:
                                time.sleep(0.1)
                            
                            _, _, success, status_info = episode_state.get_status()
                            print("─"*60 + "\n")
                        
                        # Handle stage success - advance to next prompt
                        stage_transitioned = False
                        if success and status_info.get('stage_complete', False):
                            # Check if this is the last stage - if so, end episode instead of looping
                            is_last_stage = (prompt_idx == len(prompts) - 1)
                            if is_last_stage:
                                # Last stage completed - end episode, don't loop back
                                print("\n" + "="*60)
                                print(f"🎉 All stages completed! Episode finished.")
                                print("="*60)
                                done = True
                                truncated = False
                                success = True
                                # Keep current prompt for final observation
                                current_prompt = prompts[prompt_idx]
                                # Get final observation
                                raw_obs = env.get_observation()
                                stage_transitioned = True
                                # reset
                                env.reset()
                            else:
                                # Not last stage - advance to next
                                current_prompt, prompt_idx, raw_obs, save_image = handle_prompt_transition(
                                    env, episode_state, prompts, prompt_idx, max_steps, reset_between_stages, reset_function
                                )
                                # If failure was marked during transition, end episode
                                if episode_state.success is False:
                                    done = True
                                    truncated = True
                                    success = False
                                else:
                                    # Stage transition successful - episode continues
                                    done = False
                                    truncated = False
                                    success = False
                                stage_transitioned = True
                        
                        # Get final observation if not already obtained from stage transition
                        if not stage_transitioned:
                            raw_obs = env.get_observation()
                            current_prompt = prompts[prompt_idx]
                        
                        formatted_obs = extract_observation(
                            raw_obs,
                            camera_config,
                            current_prompt,
                            resolution,
                            save_to_disk=False
                        )
                        
                        # Determine reward
                        if success and not status_info.get('stage_complete', False):
                            reward = 1.0  # Final episode success
                        elif success and status_info.get('stage_complete', False):
                            if prompt_idx == len(prompts) - 1:
                                reward = 1.0  # Last stage success - full reward
                            else:
                                reward = 0.0  # Stage success, episode continues
                        else:
                            reward = 0.0
                        
                        # Send response
                        print(f"prompt for send_msg {formatted_obs['prompt']}")
                        send_msg(conn, {
                            'observation/exterior_image_1_left': formatted_obs['observation/exterior_image_1_left'],
                            'observation/wrist_image_left': formatted_obs['observation/wrist_image_left'],
                            'observation/joint_position': formatted_obs['observation/joint_position'],
                            'observation/gripper_position': formatted_obs['observation/gripper_position'],
                            'prompt': formatted_obs['prompt'],
                            'reward': float(reward),
                            'done': bool(done),
                            'truncated': bool(truncated),
                            'success': bool(success),
                            'num_steps': num_steps_executed,
                            'info': {
                                **status_info,
                                'stage': prompt_idx + 1,
                                'total_stages': len(prompts),
                                'prompt': current_prompt,
                            },
                        })
                        
                    except Exception as e:
                        print(f"Error executing action chunk: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        current_prompt = prompts[prompt_idx] if prompt_idx < len(prompts) else prompts[0]
                        send_msg(conn, {
                            'observation/exterior_image_1_left': np.zeros((resolution, resolution, 3), dtype=np.uint8),
                            'observation/wrist_image_left': np.zeros((resolution, resolution, 3), dtype=np.uint8),
                            'observation/joint_position': np.zeros(7, dtype=np.float32),
                            'observation/gripper_position': np.zeros(1, dtype=np.float32),
                            'prompt': current_prompt,
                            'reward': 0.0,
                            'done': False,
                            'truncated': True,
                            'success': False,
                            'num_steps': num_steps_executed,
                            'info': {'error': str(e)},
                        })
                
                else:
                    # Execute single action
                    start_time = time.time()
                    
                    episode_state.increment_step()  # Increments both episode and prompt step counters
                    current_step = episode_state.get_step_count()
                    prompt_step = episode_state.get_prompt_step_count()
                    num_steps_executed = 1
                    
                    # Action is [joint_velocity (7), gripper_position (1)]
                    
                    # Binarize gripper action (from official example)
                    if action[-1] > 0.5:
                        action = np.concatenate([action[:-1], np.ones((1,))])
                    else:
                        action = np.concatenate([action[:-1], np.zeros((1,))])
                    
                    # Clip action to [-1, 1]
                    action = np.clip(action, -1, 1)
                    
                    # Execute action on robot
                    try:
                        env.step(action)
                        
                        # Check episode status from operator/timeout (with per-prompt timeout)
                        done, truncated, success, status_info = episode_state.get_status(
                            prompt_max_steps=max_steps
                        )

                        # Apply reset in main thread if keyboard requested it.
                        if apply_operator_requested_reset() and not (done or truncated):
                            status_info["manual_reset"] = True
                        
                        # Get observation
                        raw_obs = env.get_observation()
                        current_prompt = prompts[prompt_idx]
                        
                        # Format observation
                        formatted_obs = extract_observation(
                            raw_obs,
                            camera_config,
                            current_prompt,
                            resolution,
                            save_to_disk=False
                        )

                        if (formatted_obs['observation/exterior_image_1_left'] == 0).all():
                            print("Error: Exterior image is all zeros")
                            status_info['error'] = True

                        # Handle timeout confirmation
                        if truncated and status_info.get('timeout', False):
                            print("\n" + "─"*60)
                            print("⏱️  TIMEOUT reached! Was the task successful?")
                            print("─"*60)
                            print(">>> Press 's' for SUCCESS or 'f' for FAILURE")
                            print("─"*60)
                            
                            while episode_state.success is None:
                                time.sleep(0.1)
                            
                            _, _, success, status_info = episode_state.get_status()
                            print("─"*60 + "\n")
                        
                        # Handle stage success - advance to next prompt
                        stage_transitioned = False
                        if success and status_info.get('stage_complete', False):
                            # Check if this is the last stage - if so, end episode instead of looping
                            is_last_stage = (prompt_idx == len(prompts) - 1)
                            if is_last_stage:
                                # Last stage completed - end episode, don't loop back
                                print("\n" + "="*60)
                                print(f"🎉 All stages completed! Episode finished.")
                                print("="*60)
                                done = True
                                truncated = False
                                success = True
                                # Keep current prompt for final observation
                                current_prompt = prompts[prompt_idx]
                                # Get final observation
                                raw_obs = env.get_observation()
                                stage_transitioned = True
                            else:
                                # Not last stage - advance to next
                                current_prompt, prompt_idx, raw_obs, save_image = handle_prompt_transition(
                                    env, episode_state, prompts, prompt_idx, max_steps, reset_between_stages, reset_function
                                )
                                stage_transitioned = True
                        
                        # Get observation if not already obtained from stage transition
                        if not stage_transitioned:
                            raw_obs = env.get_observation()
                            current_prompt = prompts[prompt_idx]
                        
                        formatted_obs = extract_observation(
                            raw_obs,
                            camera_config,
                            current_prompt,
                            resolution,
                            save_to_disk=False
                        )
                        
                        # Determine reward
                        if success and not status_info.get('stage_complete', False):
                            reward = 1.0  # Final episode success
                        elif success and status_info.get('stage_complete', False):
                            if prompt_idx == len(prompts) - 1:
                                reward = 1.0  # Last stage success - full reward
                            else:
                                reward = 0.0  # Stage success, episode continues
                        else:
                            reward = 0.0
                        
                        # Print step info
                        status_str = ""
                        if truncated or done:
                            status_str = f" - {'SUCCESS' if success else 'FAILURE'}"
                        stage_info = f"Stage {prompt_idx + 1}/{len(prompts)}"
                        print(f"Step {current_step} (prompt: {prompt_step}) [{stage_info}]{status_str}")
                        
                        # Send response
                        print(f"prompt for send_msg {formatted_obs['prompt']}")
                        send_msg(conn, {
                            'observation/exterior_image_1_left': formatted_obs['observation/exterior_image_1_left'],
                            'observation/wrist_image_left': formatted_obs['observation/wrist_image_left'],
                            'observation/joint_position': formatted_obs['observation/joint_position'],
                            'observation/gripper_position': formatted_obs['observation/gripper_position'],
                            'prompt': formatted_obs['prompt'],
                            'reward': float(reward),
                            'done': bool(done),
                            'truncated': bool(truncated),
                            'success': bool(success),
                            'num_steps': num_steps_executed,
                            'info': {
                                **status_info,
                                'stage': prompt_idx + 1,
                                'total_stages': len(prompts),
                                'prompt': current_prompt,
                            },
                        })
                        
                        # Sleep to match DROID control frequency (from official example)
                        elapsed_time = time.time() - start_time
                        if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

                        if status_info.get('error', False):
                            print("status error goingn to exit")
                            exit()
                    except Exception as e:
                        print(f"Error executing action: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        current_prompt = prompts[prompt_idx] if prompt_idx < len(prompts) else prompts[0]
                        # Send error response
                        print(f"prompt for send_msg {current_prompt}")
                        send_msg(conn, {
                            'observation/exterior_image_1_left': np.zeros((resolution, resolution, 3), dtype=np.uint8),
                            'observation/wrist_image_left': np.zeros((resolution, resolution, 3), dtype=np.uint8),
                            'observation/joint_position': np.zeros(7, dtype=np.float32),
                            'observation/gripper_position': np.zeros(1, dtype=np.float32),
                            'prompt': current_prompt,
                            'reward': 0.0,
                            'done': True,
                            'truncated': True,
                            'success': False,
                            'num_steps': num_steps_executed,
                            'info': {'error': str(e)},
                        })
            
            elif cmd['type'] == 'CLOSE':
                print("Closing connection...")
                break
    
    except Exception as e:
        print(f"Error handling client: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


def run_droid_remote_server(prompts: List[str], left_camera_id: str, right_camera_id: str,
                            wrist_camera_id: str, external_camera: str, server_port: int,
                            max_steps: Optional[int], resolution: int, reset_between_stages: bool = False,
                            reset_function: str = "reset"):
    """
    Run a remote server that exposes the DROID robot via socket interface.
    Works with Pinggy and other tunneling services.
    
    Args:
        prompts: List of prompt strings for multi-stage tasks
        left_camera_id: Left camera serial number
        right_camera_id: Right camera serial number
        wrist_camera_id: Wrist camera serial number
        external_camera: Which external camera to use
        server_port: Port for server to listen on
        max_steps: Maximum steps per prompt before timeout
        resolution: Image resolution
        reset_between_stages: Whether to reset robot between stages
        reset_function: Reset function to use ('reset', 'reset_rewardfm_partial', or 'reset_rewardfm')
    """
    # Initialize robot
    env, camera_config = init_robot(
        left_camera_id,
        right_camera_id,
        wrist_camera_id,
        external_camera
    )
    
    # Create multi-stage episode state manager
    episode_state = MultiStageEpisodeState(max_steps=max_steps)
    
    # Start keyboard listener thread with multi-stage support
    keyboard_thread = threading.Thread(
        target=keyboard_listener,
        args=(episode_state,),
        kwargs={'multi_stage': True, 'env': env},
        daemon=True
    )
    keyboard_thread.start()
    
    # Setup socket server
    host = '0.0.0.0'
    port = server_port
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    # Set a timeout so we can check for keyboard interrupt periodically
    server_socket.settimeout(1.0)
    
    print(f"Starting DROID remote server on {host}:{port}...")
    print("Ready to accept connections (works with Pinggy tunnels)")
    
    try:
        while True:
            try:
                print("Waiting for connection...")
                conn, addr = server_socket.accept()
                print(f"Connection accepted from {addr}")
                
                # Handle client (blocking, one at a time for robot safety)
                handle_client(conn, addr, env, camera_config, prompts, 
                            max_steps, resolution, episode_state, reset_between_stages, reset_function)
            except socket.timeout:
                # This allows KeyboardInterrupt to be caught
                continue
            
    except KeyboardInterrupt:
        print("\nServer stopping due to keyboard interrupt...")
    finally:
        server_socket.close()
        print("Server closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run DROID robot as multi-stage remote server for DSRL training"
    )
    # Hardware parameters (from official example)
    parser.add_argument(
        "--left-camera-id",
        type=str,
        required=True,
        help="Left camera serial number (e.g., '24259877')"
    )
    parser.add_argument(
        "--right-camera-id",
        type=str,
        required=True,
        help="Right camera serial number (e.g., '24514023')"
    )
    parser.add_argument(
        "--wrist-camera-id",
        type=str,
        required=True,
        help="Wrist camera serial number (e.g., '13062452')"
    )
    parser.add_argument(
        "--external-camera",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Which external camera to use for policy"
    )
    
    # Server parameters
    parser.add_argument(
        "--server-port",
        type=int,
        default=6000,
        help="Port for remote server to listen on"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to text file with prompts (one per line). If not provided, --prompt will be used."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Image resolution (will resize if needed)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum steps per prompt before timeout (default: 600)"
    )
    parser.add_argument(
        "--reset-between-stages",
        action="store_true",
        default=False,
        help="If set, reset robot between stages. If not set (default), robot continues without reset between stages."
    )
    parser.add_argument(
        "--reset-function",
        type=str,
        default="reset",
        choices=["reset", "reset_rewardfm_partial", "reset_rewardfm"],
        help="Reset function to use: 'reset' (default), 'reset_rewardfm_partial', or 'reset_rewardfm'"
    )
    
    args = parser.parse_args()
    
    # Load prompts from file or use single prompt
    if args.prompt_file:
        try:
            prompts = load_prompts_from_file(args.prompt_file)
        except Exception as e:
            print(f"Error loading prompts from file: {e}")
            # Fallback to single prompt if provided
            if args.prompt:
                print(f"Falling back to single prompt: {args.prompt}")
                prompts = [args.prompt]
            else:
                raise
    else:
        raise ValueError("Either --prompt-file must be provided")
    
    print("=" * 60)
    print("DROID Robot Multi-Stage Remote Server for DSRL Training")
    print("=" * 60)
    print(f"Left camera: {args.left_camera_id}")
    print(f"Right camera: {args.right_camera_id}")
    print(f"Wrist camera: {args.wrist_camera_id}")
    print(f"External camera: {args.external_camera}")
    print(f"Server: 0.0.0.0:{args.server_port}")
    print(f"Number of stages: {len(prompts)}")
    print(f"Max steps per stage: {args.max_steps}")
    print(f"Reset between stages: {args.reset_between_stages}")
    print(f"Reset function: {args.reset_function}")
    print("=" * 60)
    
    run_droid_remote_server(
        prompts=prompts,
        left_camera_id=args.left_camera_id,
        right_camera_id=args.right_camera_id,
        wrist_camera_id=args.wrist_camera_id,
        external_camera=args.external_camera,
        server_port=args.server_port,
        max_steps=args.max_steps,
        resolution=args.resolution,
        reset_between_stages=args.reset_between_stages,
        reset_function=args.reset_function
    )


if __name__ == "__main__":
    main()

