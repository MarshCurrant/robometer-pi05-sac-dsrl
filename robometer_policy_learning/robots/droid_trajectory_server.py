#!/usr/bin/env python
"""
Remote server for DROID robot that sends scene data (images + metadata) to clients
and executes trajectories received from clients in a loop.

This server implements a trajectory-based control protocol:
1. Client sends GET_SCENE → Server responds with camera images + metadata JSON
2. Client sends EXECUTE_TRAJECTORY → Server executes trajectory, then sends next scene
3. Loop continues until episode ends or client disconnects

Based on:
- robots/droid_remote_server.py for robot initialization
- grab_scene.py for image extraction and metadata format

Prerequisites:
1. Install droid package: https://github.com/droid-dataset/droid
2. Install openpi_client: pip install openpi-client
3. Configure camera IDs in command line args

Example usage:
# Terminal 1: Start trajectory server
python robots/droid_trajectory_server.py \
    --left-camera-id "24259877" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13062452" \
    --server-port 6000

# Terminal 2: Client connects and sends commands
# GET_SCENE → receives images + metadata
# EXECUTE_TRAJECTORY → sends trajectory, receives next scene
# Loop continues...

Keyboard Controls (in Terminal 1):
  's' - Mark current episode as SUCCESS
  'f' - Mark current episode as FAILURE
  'q' - Quit server
"""

import argparse
import time
import numpy as np
import socket
import json
from typing import Dict, Any, Tuple, Optional
import threading
import os

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

# DROID specific imports
try:
    from droid.robot_env import RobotEnv
    from PIL import Image

    DROID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DROID environment not found: {e}")
    print("Robot functionality will not work. Install with:")
    print("  pip install droid")
    print("  pip install openpi-client")
    DROID_AVAILABLE = False

# DROID data collection frequency (from official example)
DROID_CONTROL_FREQUENCY = 15  # num times per second


def init_robot(left_camera_id: str, right_camera_id: str, wrist_camera_id: str, external_camera: str = "left"):
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
        raise RuntimeError("DROID not installed. Please install:\n  pip install droid\n  pip install openpi-client")

    print(f"Connecting to DROID robot...")
    print(f"  Left camera: {left_camera_id}")
    print(f"  Right camera: {right_camera_id}")
    print(f"  Wrist camera: {wrist_camera_id}")
    print(f"  Using {external_camera} camera for policy")

    # Initialize the Panda environment
    # Using joint velocity action space is very important (from official example)
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")

    print("✓ DROID robot connected")

    camera_config = {
        "left_camera_id": left_camera_id,
        "right_camera_id": right_camera_id,
        "wrist_camera_id": wrist_camera_id,
        "external_camera": external_camera,
    }

    return env, camera_config


def extract_scene_data(
    env_obs: Dict[str, Any],
    camera_config: Dict[str, str],
    target_width: int = 320,
    target_height: int = 192,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Extract scene data (images and metadata) from DROID RobotEnv.
    Similar to grab_scene.py but returns data instead of saving to disk.

    Args:
        env_obs: Raw observation from env.get_observation()
        camera_config: Camera configuration dict with left_camera_id, right_camera_id, wrist_camera_id
        target_width: Target image width (default: 320)
        target_height: Target image height (default: 192)

    Returns:
        Tuple of (images_dict, metadata_dict):
        - images_dict: {"left_image": np.ndarray, "right_image": np.ndarray, "wrist_image": np.ndarray}
        - metadata_dict: {"cartesian_position": list, "joint_positions": list, "gripper_position": float}
    """
    # Extract images
    image_observations = env_obs["image"]
    left_image, right_image, wrist_image = None, None, None

    # Find left, right, and wrist images based on camera IDs
    for key in image_observations:
        # Note: "left" below refers to the left camera in the stereo pair
        if camera_config["left_camera_id"] in key and "left" in key:
            left_image = image_observations[key]
        elif camera_config["right_camera_id"] in key and "left" in key:
            right_image = image_observations[key]
        elif camera_config["wrist_camera_id"] in key and "left" in key:
            wrist_image = image_observations[key]

    if left_image is None:
        raise ValueError(f"Could not find left camera image with ID {camera_config['left_camera_id']}")
    if right_image is None:
        raise ValueError(f"Could not find right camera image with ID {camera_config['right_camera_id']}")
    if wrist_image is None:
        raise ValueError(f"Could not find wrist camera image with ID {camera_config['wrist_camera_id']}")

    # Process images (from official example)
    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert BGR to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # Resize images to target size
    left_pil = Image.fromarray(left_image.astype(np.uint8))
    left_pil = left_pil.resize((target_width, target_height), Image.LANCZOS)
    left_image_resized = np.array(left_pil)

    right_pil = Image.fromarray(right_image.astype(np.uint8))
    right_pil = right_pil.resize((target_width, target_height), Image.LANCZOS)
    right_image_resized = np.array(right_pil)

    wrist_pil = Image.fromarray(wrist_image.astype(np.uint8))
    wrist_pil = wrist_pil.resize((target_width, target_height), Image.LANCZOS)
    wrist_image_resized = np.array(wrist_pil)

    # Extract robot state
    robot_state = env_obs["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"]).tolist()
    joint_positions = np.array(robot_state["joint_positions"]).tolist()
    gripper_position = float(robot_state["gripper_position"])

    # Create metadata dict (matching grab_scene.py format)
    metadata = {
        "cartesian_position": cartesian_position,
        "joint_positions": joint_positions,
        "gripper_position": gripper_position,
    }

    # Create images dict
    images = {
        "left_image": left_image_resized,
        "right_image": right_image_resized,
        "wrist_image": wrist_image_resized,
    }

    return images, metadata


def execute_trajectory(
    env: RobotEnv,
    trajectory: Dict[str, Any],
    episode_state: EpisodeState,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a trajectory received from client.

    Args:
        env: DROID RobotEnv instance
        trajectory: Trajectory dict with joint_vel, start_idx, end_idx, etc.
        episode_state: EpisodeState instance for tracking
        max_steps: Maximum steps per episode (for timeout checking)

    Returns:
        Dict with execution status: {"success": bool, "num_steps": int, "error": str or None}
    """
    try:
        # Validate trajectory format
        if "joint_vel" not in trajectory:
            return {"success": False, "num_steps": 0, "error": "Missing 'joint_vel' in trajectory"}

        joint_vel = trajectory["joint_vel"]
        if not isinstance(joint_vel, list) or len(joint_vel) == 0:
            return {"success": False, "num_steps": 0, "error": "Invalid 'joint_vel' format"}

        # Get trajectory bounds
        start_idx = trajectory.get("start_idx", 0)
        end_idx = trajectory.get("end_idx", len(joint_vel) - 1)

        # Validate bounds
        if start_idx < 0 or end_idx >= len(joint_vel) or start_idx > end_idx:
            return {
                "success": False,
                "num_steps": 0,
                "error": f"Invalid trajectory bounds: start_idx={start_idx}, end_idx={end_idx}, len={len(joint_vel)}",
            }

        # Extract trajectory segment
        trajectory_segment = joint_vel[start_idx : end_idx + 1]

        print(f"\n[EXECUTE_TRAJECTORY] Executing {len(trajectory_segment)} steps (indices {start_idx}-{end_idx})...")

        num_steps_executed = 0
        done = False
        truncated = False
        success = False

        # Get current gripper position from observation to maintain state
        try:
            current_obs = env.get_observation()
            robot_state = current_obs.get("robot_state", {})
            current_gripper = robot_state.get("gripper_position", 0.0)
            # Normalize gripper to [0, 1] range (assuming it's already in that range or needs conversion)
            # DROID gripper_position is typically already in [0, 1] range
            gripper_action = float(current_gripper)
        except Exception:
            # Fallback to default if we can't get current state
            gripper_action = 0.5  # Default: half-open

        # Execute each velocity command
        for i, joint_velocities in enumerate(trajectory_segment):
            start_time = time.time()

            # Increment step counter
            episode_state.increment_step()
            num_steps_executed += 1
            current_step = episode_state.get_step_count()

            # Convert to numpy
            joint_velocities = np.array(joint_velocities, dtype=np.float32)

            # Validate joint velocities (should be 7D)
            if joint_velocities.shape != (7,):
                return {
                    "success": False,
                    "num_steps": num_steps_executed,
                    "error": f"Invalid joint_vel shape at step {i}: {joint_velocities.shape}, expected (7,)",
                }

            # Use current gripper state (maintains gripper position during trajectory)
            # Client can send gripper commands separately if needed

            # Construct action: [joint_vel (7), gripper (1)] = 8D
            action = np.concatenate([joint_velocities, np.array([gripper_action])])

            # Binarize gripper (from droid_remote_server.py)
            if action[-1] > 0.5:
                action = np.concatenate([action[:-1], np.ones((1,))])
            else:
                action = np.concatenate([action[:-1], np.zeros((1,))])

            # Clip action to [-1, 1]
            action = np.clip(action, -1, 1)

            # Execute action
            try:
                env.step(action)
            except Exception as e:
                return {
                    "success": False,
                    "num_steps": num_steps_executed,
                    "error": f"Error executing action at step {i}: {str(e)}",
                }

            # Check episode status
            done, truncated, success, status_info = episode_state.get_status()

            # Check for timeout
            if max_steps and current_step >= max_steps:
                truncated = True
                status_info["timeout"] = True

            # Print progress
            is_last = i == len(trajectory_segment) - 1
            print(f"  Step {current_step} ({i + 1}/{len(trajectory_segment)})", end="")

            if done or truncated:
                status_str = " - SUCCESS" if success else " - FAILURE"
                print(status_str)
                break
            elif is_last:
                print(" (trajectory done)")
            else:
                print()

            # Match DROID control frequency
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

        # Handle timeout confirmation if needed
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

        return {
            "success": success,
            "num_steps": num_steps_executed,
            "done": done,
            "truncated": truncated,
            "error": None,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "num_steps": 0, "error": f"Exception in execute_trajectory: {str(e)}"}


def handle_client(conn, addr, env, camera_config, args, episode_state):
    """Handle a single client connection with trajectory execution loop."""
    print(f"Handling client from {addr}")
    try:
        while True:
            cmd = recv_msg(conn)
            if cmd is None:
                print("Client disconnected")
                break

            if cmd["type"] == "RESET":
                print("\n" + "=" * 60)
                print("Resetting DROID robot for new episode...")
                print("=" * 60)

                # Reset episode state
                episode_state.reset()

                # Reset robot
                env.reset_rewardfm()

                # Get observation after reset
                raw_obs = env.get_observation()

                # Extract scene data
                images, metadata = extract_scene_data(
                    raw_obs, camera_config, args.target_width, args.target_height
                )

                print(f"\nMax steps: {args.max_steps if args.max_steps else 'unlimited'}")
                print("Ready to send scene data and execute trajectories\n")

                # Send scene data to client
                response = {
                    "type": "SCENE_DATA",
                    "left_image": images["left_image"],
                    "right_image": images["right_image"],
                    "wrist_image": images["wrist_image"],
                    "metadata": metadata,
                }
                send_msg(conn, response)

            elif cmd["type"] == "GET_SCENE":
                # Get current observation
                raw_obs = env.get_observation()

                # Extract scene data
                images, metadata = extract_scene_data(
                    raw_obs, camera_config, args.target_width, args.target_height
                )

                # Send scene data to client
                response = {
                    "type": "SCENE_DATA",
                    "left_image": images["left_image"],
                    "right_image": images["right_image"],
                    "wrist_image": images["wrist_image"],
                    "metadata": metadata,
                }
                send_msg(conn, response)

            elif cmd["type"] == "EXECUTE_TRAJECTORY":
                # Receive trajectory from client
                trajectory = cmd.get("trajectory")
                if trajectory is None:
                    send_msg(
                        conn,
                        {
                            "type": "ERROR",
                            "message": "Missing 'trajectory' in EXECUTE_TRAJECTORY command",
                        },
                    )
                    continue

                # Execute trajectory
                execution_result = execute_trajectory(env, trajectory, episode_state, args.max_steps)

                # Check if we should continue or end
                if execution_result.get("error"):
                    # Send error response
                    send_msg(
                        conn,
                        {
                            "type": "TRAJECTORY_RESULT",
                            "success": False,
                            "error": execution_result["error"],
                            "num_steps": execution_result["num_steps"],
                        },
                    )
                    continue

                # After successful execution, automatically send next scene
                raw_obs = env.get_observation()
                images, metadata = extract_scene_data(
                    raw_obs, camera_config, args.target_width, args.target_height
                )

                # Send trajectory result + next scene
                response = {
                    "type": "TRAJECTORY_RESULT",
                    "success": execution_result["success"],
                    "num_steps": execution_result["num_steps"],
                    "done": execution_result.get("done", False),
                    "truncated": execution_result.get("truncated", False),
                    # Include next scene automatically
                    "next_scene": {
                        "left_image": images["left_image"],
                        "right_image": images["right_image"],
                        "wrist_image": images["wrist_image"],
                        "metadata": metadata,
                    },
                }
                send_msg(conn, response)

            elif cmd["type"] == "CLOSE":
                print("Closing connection...")
                break

    except Exception as e:
        print(f"Error handling client: {e}")
        import traceback

        traceback.print_exc()
    finally:
        conn.close()


def run_droid_trajectory_server(args):
    """
    Run a remote server that sends scene data and executes trajectories.
    Works with Pinggy and other tunneling services.
    """
    # Initialize robot
    env, camera_config = init_robot(
        args.left_camera_id, args.right_camera_id, args.wrist_camera_id, args.external_camera
    )

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

    print(f"Starting DROID trajectory server on {host}:{port}...")
    print("Ready to accept connections (works with Pinggy tunnels)")

    try:
        while True:
            try:
                print("Waiting for connection...")
                conn, addr = server_socket.accept()
                print(f"Connection accepted from {addr}")

                # Handle client (blocking, one at a time for robot safety)
                handle_client(conn, addr, env, camera_config, args, episode_state)
            except socket.timeout:
                # This allows KeyboardInterrupt to be caught
                continue

    except KeyboardInterrupt:
        print("\nServer stopping due to keyboard interrupt...")
    finally:
        server_socket.close()
        print("Server closed.")


def main():
    parser = argparse.ArgumentParser(description="Run DROID robot as trajectory execution server")
    # Hardware parameters
    parser.add_argument(
        "--left-camera-id", type=str, required=True, help="Left camera serial number (e.g., '24259877')"
    )
    parser.add_argument(
        "--right-camera-id", type=str, required=True, help="Right camera serial number (e.g., '24514023')"
    )
    parser.add_argument(
        "--wrist-camera-id", type=str, required=True, help="Wrist camera serial number (e.g., '13062452')"
    )
    parser.add_argument(
        "--external-camera",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Which external camera to use for policy",
    )

    # Server parameters
    parser.add_argument("--server-port", type=int, default=6000, help="Port for remote server to listen on")
    parser.add_argument(
        "--target-width", type=int, default=320, help="Target image width (default: 320 for pi0.5)"
    )
    parser.add_argument(
        "--target-height", type=int, default=192, help="Target image height (default: 192 for pi0.5)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum steps per episode before timeout (default: 600)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DROID Trajectory Execution Server")
    print("=" * 60)
    print(f"Left camera: {args.left_camera_id}")
    print(f"Right camera: {args.right_camera_id}")
    print(f"Wrist camera: {args.wrist_camera_id}")
    print(f"External camera: {args.external_camera}")
    print(f"Server: 0.0.0.0:{args.server_port}")
    print(f"Image size: {args.target_width}x{args.target_height}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60)

    run_droid_trajectory_server(args)


if __name__ == "__main__":
    main()
