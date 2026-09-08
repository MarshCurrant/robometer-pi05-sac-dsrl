#!/usr/bin/env python
"""
Remote server for USC WidowX robot that interfaces with DSRL training.
Uses standard TCP sockets that work with Pinggy and other tunneling services.
Includes keyboard input for marking success/failure and episode timeouts.

Example usage:
# Terminal 1: Start robot server
python robots/widowx_remote_server.py \
    --robot-ip localhost \
    --robot-port 5556 \
    --server-port 6000 \
    --prompt "pick up the red block" \
    --max-steps 40 

# Terminal 2: Tunnel with Pinggy
ssh -p 443 -R0:localhost:6000 a.pinggy.io

# Terminal 3: Start DSRL training (example: bridge config + tunnel host/port)
uv run python scripts/train_dsrl.py \
    config_name=dsrl_bridge_config.yaml \
    remote_robot.host=<tunnel-host> \
    remote_robot.port=<tunnel-port> \
    pi0_checkpoint=/path/to/pi0 \
    ...

Keyboard Controls (in Terminal 1):
  ENTER - Start episode
  's' - Mark current episode as SUCCESS
  'f' - Mark current episode as FAILURE  
  'q' - Quit server
"""

import argparse
import time
import numpy as np
import cv2
import threading
import socket
import os
from typing import Dict, Any

# Import shared utilities
import importlib.util

spec = importlib.util.spec_from_file_location(
    "remote_server_utils", os.path.join(os.path.dirname(__file__), "remote_server_utils.py")
)
remote_server_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_server_utils)

# Use shared utilities
EpisodeState = remote_server_utils.EpisodeState
keyboard_listener = remote_server_utils.keyboard_listener
send_msg = remote_server_utils.send_msg
recv_msg = remote_server_utils.recv_msg

# WidowX specific imports
try:
    from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs as _WidowXConfigs, show_video

    WIDOWX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: widowx_envs not found: {e}")
    print("Robot functionality will not work. Install with: pip install -e .[widowx]")
    WIDOWX_AVAILABLE = False
    _WidowXConfigs = None

# OpenPI geometry utilities for state processing
try:
    from openpi.shared.geometry import quat2mat, mat2euler
except ImportError:
    print("Warning: openpi geometry not found, using fallback")

    # Fallback implementations
    def quat2mat(quat):
        """Minimal quaternion to rotation matrix conversion"""
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def mat2euler(mat):
        """Minimal rotation matrix to euler angles conversion"""
        sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(mat[2, 1], mat[2, 2])
            y = np.arctan2(-mat[2, 0], sy)
            z = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            x = np.arctan2(-mat[1, 2], mat[1, 1])
            y = np.arctan2(-mat[2, 0], sy)
            z = 0
        return np.array([x, y, z])


# WidowX data collection frequency (from official example)
WIDOWX_CONTROL_FREQUENCY = 5  # num times per second: 5hz = 5 times per second


# Default configuration - used as fallback or can be overridden
class WidowXConfigsDefault:
    """Default configuration for WidowX robot"""

    DefaultEnvParams = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, 0.028, -1.57, 0],
            [0.45, 0.25, 0.25, 1.57, 0],
        ],
        "catch_environment_except": False,
        "start_state": [0.3, 0.0, 0.15, 0, 0, 0, 1],
        "skip_move_to_neutral": False,
        "return_full_image": False,
        "camera_topics": [
            {"name": "/blue/image_raw"},
        ],
    }


# Use imported config if available, otherwise use default
WidowXConfigs = WidowXConfigsDefault


def init_robot(robot_ip: str, robot_port: int, resolution: int = 224):
    """Initialize connection to WidowX robot."""
    if not WIDOWX_AVAILABLE:
        raise RuntimeError("widowx_envs not installed. Please install with: pip install -e .[widowx]")

    print(f"Connecting to WidowX controller @ {robot_ip}:{robot_port}...")

    widowx_client = WidowXClient(host=robot_ip, port=robot_port)
    widowx_client.init(WidowXConfigs.DefaultEnvParams, image_size=resolution)

    print("Successfully connected to WidowX.")
    print("Waiting for initial observation...")

    # Wait for valid observation
    obs = wait_for_observation(widowx_client)

    print("Initial observation received.")
    print("Resetting robot...")
    widowx_client.reset()
    obs = wait_for_observation(widowx_client)

    show_video(widowx_client, duration=2.5)
    print("Video shown. Robot ready")
    return widowx_client


def wait_for_observation(client: WidowXClient, timeout: int = 60) -> Dict:
    """Wait for and return a valid observation from the robot.

    Args:
        client: WidowX client
        timeout: Maximum time to wait
    """
    start_time = time.time()
    while True:
        obs = client.get_observation()
        if obs is not None:
            print("✓ Received observation")
            return obs

        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"No observation received from robot after {timeout}s")

        time.sleep(1)
        print(f"⏳ Waiting for robot observation... (elapsed: {elapsed:.1f}s)")


def format_observation_for_dsrl(raw_obs: Dict[str, Any], prompt: str, resolution: int = 224) -> Dict[str, Any]:
    """
    Format raw robot observation into the structure expected by DSRL.
    Matches the WidowX / bridge-style observation layout (state vector + RGB image) used with `REMOTE_ROBOT` training.

    Args:
        raw_obs: Raw observation from WidowX client
        prompt: Task instruction string
        resolution: Target image resolution

    Returns:
        Formatted observation dict with keys: 'state', 'image', 'instruction'
    """
    # Process state (end-effector pose) - convert quaternion to euler angles
    eef_pos = raw_obs["state"]
    print(eef_pos[:3])
    # state = preprocess_widowx_proprio(eef_pos)

    # Process image - WidowX returns 'full_image' key
    image = raw_obs.get("full_image")
    if image is None:
        raise ValueError(f"No 'full_image' in observation. Available keys: {raw_obs.keys()}")

    # Resize to target resolution if needed
    if image.shape[0] != resolution or image.shape[1] != resolution:
        image = cv2.resize(image, (resolution, resolution))

    formatted_obs = {
        "state": eef_pos.astype(np.float32),
        "observation.images.image_0": image,
        "prompt": prompt,
    }

    return formatted_obs


def handle_client(conn, addr, widowx_client, args, episode_state):
    """Handle a single client connection with keyboard input support."""
    print(f"Handling client from {addr}")

    def _interruptible_sleep(total_sleep_s: float) -> bool:
        """Sleep in small increments so operator success/failure can interrupt quickly.

        Returns True if we should abort (episode already done/truncated), else False.
        """
        end_t = time.time() + max(0.0, float(total_sleep_s))
        while time.time() < end_t:
            done_i, truncated_i, _, _ = episode_state.get_status()
            if done_i or truncated_i:
                return True
            time.sleep(max(0.01, end_t - time.time()))
        return False

    try:
        while True:
            cmd = recv_msg(conn)
            if cmd is None:
                print("Client disconnected")
                break

            if cmd["type"] == "RESET":
                print("\n" + "=" * 60)
                print("Resetting robot for new episode...")
                print("=" * 60)

                # Reset episode state
                episode_state.reset()

                widowx_client.reset()
                time.sleep(0.5)  # Allow time for reset

                # Get observation after reset
                raw_obs = wait_for_observation(widowx_client)

                # Format observation
                formatted_obs = format_observation_for_dsrl(raw_obs, args.prompt, args.resolution)

                print(f"\nTask: {args.prompt}")
                print(f"Max steps: {args.max_steps if args.max_steps else 'unlimited'}")

                # Wait for operator to press Enter before starting episode
                print("\n" + "─" * 60)
                print("🤖 Robot is ready. Set up the scene if needed.")
                print("─" * 60)
                input(">>> Press ENTER to start episode...")
                print("─" * 60)
                print("Episode starting! Press 's' for success, 'f' for failure")
                print("─" * 60 + "\n")

                # Advertise server capabilities
                info = {"supports_action_chunking": True}

                # Send back to client (episode starts now)
                send_msg(conn, {**formatted_obs, "info": info})

            elif cmd["type"] == "SUCCESS_CHECK":
                # This remote server doesn't support multi-stage tasks so just trust that the task is done
                send_msg(conn, {"done": True})

            elif cmd["type"] == "STEP":
                action = cmd["action"]
                need_obs = cmd.get("need_obs", True)
                need_obs_every_step = cmd.get("need_obs_every_step", False)

                # Convert to numpy and detect if this is a chunk
                action = np.array(action, dtype=np.float32)
                is_chunk = action.ndim == 2  # Single: (7,), Chunk: (N, 7)

                if is_chunk:
                    # Execute multiple actions (chunked execution)
                    actions = action

                    print(f"\n[CHUNK] Executing {len(actions)} actions...")

                    num_steps_executed = 0
                    done = False
                    truncated = False
                    success = False
                    reward = 0.0
                    status_info = {}
                    if need_obs_every_step:
                        formatted_obs_list = []
                        reward_list = []
                        done_list = []
                        truncated_list = []
                        success_list = []
                        status_info_list = []

                    try:
                        for i, single_action in enumerate(actions):
                            # Check status BEFORE executing action - stop immediately if marked
                            done, truncated, success, status_info = episode_state.get_status()
                            if done or truncated:
                                # Already marked success/failure, don't execute more actions
                                break

                            start_time = time.time()
                            episode_state.increment_step()
                            num_steps_executed += 1
                            current_step = episode_state.get_step_count()

                            # Discretize gripper
                            if single_action.shape[0] == 7:
                                single_action[6] = int(single_action[6] > 0.5)

                            # Re-check status right before sending command (closes a small timing window)
                            done, truncated, success, status_info = episode_state.get_status()
                            if done or truncated:
                                break

                            # Execute action
                            widowx_client.step_action(single_action)

                            # Check episode status again after action
                            done, truncated, success, status_info = episode_state.get_status()
                            # Print progress
                            is_last = i == len(actions) - 1
                            print(f"  Step {current_step} ({i + 1}/{len(actions)})", end="")

                            if done or truncated:
                                status_str = " - SUCCESS" if success else " - FAILURE"
                                print(status_str)
                                # Only break here if NOT need_obs_every_step
                                # Otherwise, continue to collect final observation below
                                if not need_obs_every_step:
                                    break
                            elif is_last:
                                print(" (chunk done)")
                            else:
                                print()

                            # match WidowX frequency
                            elapsed_time = time.time() - start_time
                            if elapsed_time < 1 / WIDOWX_CONTROL_FREQUENCY:
                                aborted = _interruptible_sleep(1 / WIDOWX_CONTROL_FREQUENCY - elapsed_time)
                                if aborted:
                                    # Episode was marked during the wait; don't execute more actions.
                                    done, truncated, success, status_info = episode_state.get_status()
                                    if not need_obs_every_step:
                                        break

                            # Determine reward
                            if success:
                                reward = 1.0
                            else:
                                reward = 0.0

                            if need_obs_every_step:
                                # Get observation for every step
                                raw_obs = wait_for_observation(widowx_client)
                                formatted_obs = format_observation_for_dsrl(raw_obs, args.prompt, args.resolution)

                                # Re-check status - user may have pressed s/f during wait_for_observation
                                done, truncated, success, status_info = episode_state.get_status()

                                # Update reward based on fresh status
                                if success:
                                    reward = 1.0
                                else:
                                    reward = 0.0

                                formatted_obs_list.append(formatted_obs)
                                reward_list.append(reward)
                                done_list.append(done)
                                truncated_list.append(truncated)
                                success_list.append(success)
                                status_info_list.append(status_info)

                                if success or done or truncated:
                                    break

                        # Handle timeout confirmation
                        if truncated and status_info.get("timeout", False):
                            print("\n" + "─" * 60)
                            print("⏱️  TIMEOUT reached! Was the task successful?")
                            print("─" * 60)
                            print(">>> Press 's' for SUCCESS or 'f' for FAILURE")
                            print("─" * 60)

                            while episode_state.success is None:
                                time.sleep(0.1)

                            _, _, success, status_info = episode_state.get_status()
                            print("─" * 60 + "\n")

                        # Refresh status one last time right before producing the response
                        done, truncated, success, status_info = episode_state.get_status()

                        # Determine reward based on latest status (used for non-list response)
                        reward = 1.0 if success else 0.0

                        # Get final observation
                        if not need_obs_every_step:
                            raw_obs = wait_for_observation(widowx_client)
                            formatted_obs = format_observation_for_dsrl(raw_obs, args.prompt, args.resolution)
                        else:
                            # Ensure list-mode response is never empty even if the episode ended
                            # before the first action in this chunk was executed.
                            if len(formatted_obs_list) == 0:
                                raw_obs = wait_for_observation(widowx_client)
                                formatted_obs = format_observation_for_dsrl(raw_obs, args.prompt, args.resolution)
                                formatted_obs_list.append(formatted_obs)
                                reward_list.append(float(reward))
                                done_list.append(bool(done))
                                truncated_list.append(bool(truncated))
                                success_list.append(bool(success))
                                status_info_list.append(status_info)
                            else:
                                # If the operator marks success/failure AFTER the last step's observation
                                # was already collected, ensure the *final returned transition* reflects it.
                                # This avoids "episode ended but trajectory says not done/truncated" bugs.
                                reward_list[-1] = float(reward)
                                done_list[-1] = bool(done)
                                truncated_list[-1] = bool(truncated)
                                success_list[-1] = bool(success)
                                status_info_list[-1] = status_info

                        # Send response
                        if need_obs_every_step:
                            send_msg(
                                conn,
                                {
                                    "state": [f_obs["state"] for f_obs in formatted_obs_list],
                                    "observation.images.image_0": [
                                        f_obs["observation.images.image_0"] for f_obs in formatted_obs_list
                                    ],
                                    "prompt": [f_obs["prompt"] for f_obs in formatted_obs_list],
                                    "reward": [float(r) for r in reward_list],
                                    "done": [bool(d) for d in done_list],
                                    "truncated": [bool(t) for t in truncated_list],
                                    "success": [bool(s) for s in success_list],
                                    # num_steps = number of env actions actually executed on the robot.
                                    # (May be 0 even if we return a single terminal observation.)
                                    "num_steps": num_steps_executed,
                                    # Optional extra debugging signal for list-mode consumers.
                                    "num_transitions": len(reward_list),
                                    "info": [s_info for s_info in status_info_list],
                                },
                            )
                        else:
                            send_msg(
                                conn,
                                {
                                    "state": formatted_obs["state"],
                                    "observation.images.image_0": formatted_obs["observation.images.image_0"],
                                    "prompt": formatted_obs["prompt"],
                                    "reward": float(reward),
                                    "done": bool(done),
                                    "truncated": bool(truncated),
                                    "success": bool(success),
                                    "num_steps": num_steps_executed,
                                    "info": status_info,
                                },
                            )

                    except Exception as e:
                        print(f"Error executing action chunk: {e}")
                        import traceback

                        traceback.print_exc()

                        if need_obs_every_step:
                            send_msg(
                                conn,
                                {
                                    "state": [np.zeros(7, dtype=np.float32)],
                                    "observation.images.image_0": [
                                        np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8)
                                    ],
                                    "prompt": [args.prompt],
                                    "reward": [0.0],
                                    "done": [True],
                                    "truncated": [True],
                                    "success": [False],
                                    "num_steps": num_steps_executed,
                                    "info": [{"error": str(e)}],
                                },
                            )
                        else:
                            send_msg(
                                conn,
                                {
                                    "state": np.zeros(7, dtype=np.float32),
                                    "observation.images.image_0": np.zeros(
                                        (args.resolution, args.resolution, 3), dtype=np.uint8
                                    ),
                                    "prompt": args.prompt,
                                    "reward": 0.0,
                                    "done": True,
                                    "truncated": True,
                                    "success": False,
                                    "num_steps": num_steps_executed,
                                    "info": {"error": str(e)},
                                },
                            )

                else:
                    # Execute single action
                    episode_state.increment_step()
                    current_step = episode_state.get_step_count()
                    num_steps_executed = 1

                    # Action is expected to be [x, y, z, rx, ry, rz, gripper]
                    # Convert gripper to discrete 0/1 if needed
                    if action.shape[0] == 7:
                        # Discretize gripper
                        action[6] = int(action[6] > 0.5)

                    # Execute action on robot
                    formatted_obs = None
                    try:
                        step_result = widowx_client.step_action(action)

                        # Check episode status FIRST to see if we need camera
                        done, truncated, success, status_info = episode_state.get_status()

                        # Determine if we need camera capture:
                        # - Client requested full observation (need_obs=True)
                        # - Episode is ending (done or truncated)
                        need_camera = need_obs or done or truncated

                        # Get observation (with or without camera)
                        raw_obs = wait_for_observation(widowx_client)

                        # Format observation
                        if need_camera:
                            formatted_obs = format_observation_for_dsrl(raw_obs, args.prompt, args.resolution)
                        else:
                            # Minimal observation - just state, placeholder image
                            # This is much faster since we skip camera capture
                            formatted_obs = {
                                "state": raw_obs["state"].astype(np.float32),
                                "observation.images.image_0": np.zeros(
                                    (args.resolution, args.resolution, 3), dtype=np.uint8
                                ),
                                "prompt": args.prompt,
                            }

                        # If timeout, wait for operator to press s/f using keyboard listener
                        if truncated and status_info.get("timeout", False):
                            print("\n" + "─" * 60)
                            print("⏱️  TIMEOUT reached! Was the task successful?")
                            print("─" * 60)
                            print(">>> Press 's' for SUCCESS or 'f' for FAILURE")
                            print("─" * 60)

                            # Wait for operator to mark success/failure via keyboard listener
                            while episode_state.success is None:
                                time.sleep(0.1)

                            # Get the updated status after operator input
                            _, _, success, status_info = episode_state.get_status()
                            print("─" * 60 + "\n")

                            # Now we need camera since episode ended
                            if not need_camera:
                                # Get fresh observation with camera
                                raw_obs = wait_for_observation(widowx_client)
                                formatted_obs = format_observation_for_dsrl(raw_obs, args.prompt, args.resolution)

                        # Determine reward based on success (works with truncated=True)
                        if success:
                            reward = 1.0  # Success
                        else:
                            reward = 0.0  # Ongoing or failure

                        # Merge info
                        info = status_info.copy()
                        if isinstance(step_result, tuple) and len(step_result) > 3:
                            info.update(step_result[3])

                        # Print step info
                        status_str = ""
                        if truncated or done:
                            status_str = f" - {'SUCCESS' if success else 'FAILURE'}"
                        print(f"Step {current_step}{status_str}")

                        # Send response
                        send_msg(
                            conn,
                            {
                                "state": formatted_obs["state"],
                                "observation.images.image_0": formatted_obs["observation.images.image_0"],
                                "prompt": formatted_obs["prompt"],
                                "reward": float(reward),
                                "done": bool(done),
                                "truncated": bool(truncated),
                                "success": bool(success),
                                "num_steps": num_steps_executed,
                                "info": info,
                            },
                        )

                    except Exception as e:
                        print(f"Error executing action: {e}")
                        import traceback

                        traceback.print_exc()

                        # Try to get current observation for error response
                        error_state = np.zeros(7, dtype=np.float32)
                        error_image = np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8)

                        if formatted_obs is not None:
                            error_state = formatted_obs["state"]
                            error_image = formatted_obs["observation.images.image_0"]
                        else:
                            # Try to get observation even after error
                            try:
                                raw_obs = widowx_client.get_observation()
                                if raw_obs is not None:
                                    formatted_obs = format_observation_for_dsrl(raw_obs, args.prompt, args.resolution)
                                    error_state = formatted_obs["state"]
                                    error_image = formatted_obs["image"]
                            except:
                                pass  # Use zeros if we can't get observation

                        # Send error response
                        send_msg(
                            conn,
                            {
                                "state": error_state,
                                "observation.images.image_0": error_image,
                                "prompt": args.prompt,
                                "reward": 0.0,
                                "done": True,
                                "truncated": True,
                                "success": False,
                                "num_steps": num_steps_executed,
                                "info": {"error": str(e)},
                            },
                        )

            elif cmd["type"] == "CLOSE":
                print("Closing connection...")
                break

    except Exception as e:
        print(f"Error handling client: {e}")
        import traceback

        traceback.print_exc()
    finally:
        conn.close()


def run_widowx_remote_server(args):
    """
    Run a remote server that exposes the WidowX robot via socket interface.
    Works with Pinggy and other tunneling services.
    """
    # Initialize robot
    widowx_client = init_robot(args.robot_ip, args.robot_port, args.resolution)

    # Create episode state manager
    episode_state = EpisodeState(max_steps=args.max_steps)

    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, args=(episode_state,), daemon=True)
    keyboard_thread.start()

    # Setup socket server
    host = "0.0.0.0"
    port = args.server_port

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)

    # Set a timeout so we can check for keyboard interrupt periodically
    server_socket.settimeout(1.0)

    print(f"Starting WidowX remote server on {host}:{port}...")
    print("Ready to accept connections (works with Pinggy tunnels)")

    try:
        while True:
            try:
                print("Waiting for connection...")
                conn, addr = server_socket.accept()
                print(f"Connection accepted from {addr}")

                # Handle client (blocking, one at a time for robot safety)
                handle_client(conn, addr, widowx_client, args, episode_state)
            except socket.timeout:
                # This allows KeyboardInterrupt to be caught
                continue

    except KeyboardInterrupt:
        print("\nServer stopping due to keyboard interrupt...")
    finally:
        server_socket.close()
        print("Server closed.")


def main():
    parser = argparse.ArgumentParser(description="Run WidowX robot as remote server for DSRL training")
    parser.add_argument("--robot-ip", type=str, default="localhost", help="IP address of WidowX robot controller")
    parser.add_argument("--robot-port", type=int, default=5556, help="Port of WidowX robot controller")
    parser.add_argument("--server-port", type=int, default=6000, help="Port for remote server to listen on")
    parser.add_argument("--prompt", type=str, required=True, help="Task instruction for the robot")
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution (will resize if needed)")
    parser.add_argument(
        "--max-steps", type=int, default=60, help="Maximum steps per episode before timeout (default: 60)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WidowX Remote Server for DSRL Training")
    print("=" * 60)
    print(f"Robot: {args.robot_ip}:{args.robot_port}")
    print(f"Server: 0.0.0.0:{args.server_port}")
    print(f"Task: {args.prompt}")
    print(f"Max steps: {args.max_steps if args.max_steps else 'unlimited'}")
    print("=" * 60)

    run_widowx_remote_server(args)


if __name__ == "__main__":
    main()
