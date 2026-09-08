#!/usr/bin/env python
"""
Remote server for real DROID robot that interfaces with DSRL training.
Based on Physical Intelligence's official DROID example:
https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py

Uses standard TCP sockets that work with Pinggy and other tunneling services.
Includes keyboard input for marking success/failure and episode timeouts.

This file shares structure with widowx_remote_server.py for consistency.

Prerequisites:
1. Install droid package: https://github.com/droid-dataset/droid
2. Install openpi_client: pip install openpi-client
3. Configure camera IDs in command line args

Example usage:
# Terminal 1: Start robot server
python robots/droid_remote_server.py \
    --left-camera-id "24259877" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13062452" \
    --external-camera left \
    --server-port 6000 \
    --prompt "pick up the red block" \
    --max-steps 40

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
  's' - Mark current episode as SUCCESS
  'f' - Mark current episode as FAILURE  
  'q' - Quit server
"""

import argparse
import time
import numpy as np
import socket
import pickle
import struct
from typing import Dict, Any
import threading
import sys
import select
import os
def resize_images(images: np.ndarray, downscale_factor: int = 2) -> np.ndarray:
    """Resize images by downscale factor"""
    B, H, W, C = images.shape
    if H == H // downscale_factor and W == W // downscale_factor:
        return images

    resized = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((W // downscale_factor, H // downscale_factor))
        resized.append(np.array(pil_img))
    return np.array(resized)
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


def extract_observation(
    env_obs: Dict[str, Any],
    camera_config: Dict[str, str],
    prompt: str,
    resolution: int = 224,
    save_to_disk: bool = False,
) -> Dict[str, Any]:
    """
    Extract observation from DROID RobotEnv in the format expected by DSRL.
    Based on official Pi0 DROID example.

    Args:
        env_obs: Raw observation from env.get_observation()
        camera_config: Camera configuration dict
        prompt: Task instruction
        resolution: Target image resolution
        save_to_disk: Whether to save combined image to disk

    Returns:
        Formatted observation dict with DROID keys
    """
    # Extract images
    image_observations = env_obs["image"]
    print(f"image_observations keys: {image_observations.keys()}")
    left_image, _, wrist_image = None, None, None

    for key in image_observations:
        # Note: "left" below refers to the left camera in the stereo pair
        # The model is only trained on left stereo cams

        if camera_config["left_camera_id"] in key and "left" in key:
            left_image = image_observations[key]

        # elif camera_config['right_camera_id'] in key and "left" in key:
        #     right_image = image_observations[key]
        elif camera_config["wrist_camera_id"] in key and "left" in key:
            wrist_image = image_observations[key]

    # Process images (from official example)
    # Drop the alpha dimension

    # print(f"right_image: {right_image.shape}")
    left_image = left_image[..., :3]
    # right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert BGR to RGB
    left_image = left_image[..., ::-1]
    # right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # Resize images to target resolution
    external_image = left_image if camera_config["external_camera"] == "left" else left_image
    external_image_resized = resize_images(external_image[None], downscale_factor=2)[0]
    wrist_image_resized = resize_images(wrist_image[None], downscale_factor=2)[0]

    #external_image_resized = image_tools.resize_with_pad(external_image, resolution, resolution)
    #wrist_image_resized = image_tools.resize_with_pad(wrist_image, resolution, resolution)

    #print(f"external_image_resized: {external_image_resized.shape}")
    #print(f"wrist_image_resized: {wrist_image_resized.shape}")

    # Extract proprioceptive state
    robot_state = env_obs["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save combined image for live viewing
    if save_to_disk:
        combined_image = np.concatenate([external_image_resized, wrist_image_resized], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")
        print("  Saved camera views to robot_camera_views.png")

    # Format in DROID observation format (matches droid_policy.py)
    formatted_obs = {
        "observation/exterior_image_1_left": external_image_resized.astype(np.uint8),
        "observation/wrist_image_left": wrist_image_resized.astype(np.uint8),
        "observation/joint_position": joint_position.astype(np.float32),
        "observation/gripper_position": gripper_position.astype(np.float32),
        "prompt": prompt,
    }

    return formatted_obs


def handle_client(conn, addr, env, camera_config, args, episode_state):
    """Handle a single client connection with keyboard input support."""
    print(f"Handling client from {addr}")

    def apply_operator_requested_reset() -> bool:
        """Apply keyboard-requested reset from this main loop thread."""
        if episode_state.consume_env_reset_request():
            print("\nApplying operator-requested reset...")
            env.reset()
            return True
        return False

    try:
        while True:
            cmd = recv_msg(conn)
            print(f"cmd: {cmd}")
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
                env.reset()
                # time.sleep(0.5)  # Allow robot to settle

                # Get observation after reset
                raw_obs = env.get_observation()

                # Format observation (save first frame to disk)
                formatted_obs = extract_observation(
                    raw_obs, camera_config, args.prompt, args.resolution, save_to_disk=True
                )

                if (formatted_obs["observation/exterior_image_1_left"] == 0).all():
                    print("Error: Exterior image is all zeros")
                    exit()

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
                print(f"prompt for send_msg {formatted_obs['prompt']}")
                send_msg(conn, {**formatted_obs, "info": info})
            
            elif cmd["type"] == "SUCCESS_CHECK":
                # This remote server doesn't support multi-stage tasks so just trust that the task is done
                send_msg(conn, {"done": True})
                env.reset()

            elif cmd["type"] == "STEP":
                action = cmd["action"]
                need_obs = cmd.get("need_obs", True)

                # Convert to numpy and detect if this is a chunk
                action = np.array(action, dtype=np.float32)
                is_chunk = action.ndim == 2  # Single: (8,), Chunk: (N, 8)

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

                            episode_state.increment_step()
                            num_steps_executed += 1
                            current_step = episode_state.get_step_count()

                            # Binarize gripper
                            if single_action[-1] > 0.5:
                                single_action = np.concatenate([single_action[:-1], np.ones((1,))])
                            else:
                                single_action = np.concatenate([single_action[:-1], np.zeros((1,))])

                            # Clip action
                            single_action = np.clip(single_action, -1, 1)

                            # Execute action
                            env.step(single_action)

                            # Check episode status
                            done, truncated, success, status_info = episode_state.get_status()

                            # Apply reset in main loop thread if requested from keyboard.
                            if apply_operator_requested_reset() and not (done or truncated):
                                status_info["manual_reset"] = True
                                break

                            # Print progress
                            is_last = i == len(actions) - 1
                            print(f"  Step {current_step} ({i + 1}/{len(actions)})", end="")

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

                        # Get final observation
                        raw_obs = env.get_observation()
                        formatted_obs = extract_observation(
                            raw_obs, camera_config, args.prompt, args.resolution, save_to_disk=False
                        )

                        # Determine reward
                        if success:
                            reward = 1.0
                        else:
                            reward = 0.0

                        # Send response
                        print(f"prompt for send_msg {formatted_obs['prompt']}")
                        send_msg(
                            conn,
                            {
                                "observation/exterior_image_1_left": formatted_obs["observation/exterior_image_1_left"],
                                "observation/wrist_image_left": formatted_obs["observation/wrist_image_left"],
                                "observation/joint_position": formatted_obs["observation/joint_position"],
                                "observation/gripper_position": formatted_obs["observation/gripper_position"],
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
                        print(f"prompt for send_msg {formatted_obs['prompt']}")

                        send_msg(
                            conn,
                            {
                                "observation/exterior_image_1_left": np.zeros(
                                    (args.resolution, args.resolution, 3), dtype=np.uint8
                                ),
                                "observation/wrist_image_left": np.zeros(
                                    (args.resolution, args.resolution, 3), dtype=np.uint8
                                ),
                                "observation/joint_position": np.zeros(7, dtype=np.float32),
                                "observation/gripper_position": np.zeros(1, dtype=np.float32),
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
                    start_time = time.time()

                    episode_state.increment_step()
                    current_step = episode_state.get_step_count()
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

                        # Check episode status from operator/timeout
                        done, truncated, success, status_info = episode_state.get_status()

                        # Apply reset in main loop thread if requested from keyboard.
                        if apply_operator_requested_reset() and not (done or truncated):
                            status_info["manual_reset"] = True

                        # Get observation
                        raw_obs = env.get_observation()

                        # Format observation
                        formatted_obs = extract_observation(
                            raw_obs, camera_config, args.prompt, args.resolution, save_to_disk=False
                        )

                        if (formatted_obs["observation/exterior_image_1_left"] == 0).all():
                            print("Error: Exterior image is all zeros")
                            status_info["error"] = True

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

                        # Determine reward
                        if success:
                            reward = 1.0
                        else:
                            reward = 0.0

                        # Print step info
                        status_str = ""
                        if truncated or done:
                            status_str = f" - {'SUCCESS' if success else 'FAILURE'}"
                        print(f"Step {current_step}{status_str}")

                        # Send response
                        print(f"prompt for send_msg {formatted_obs['prompt']}")
                        send_msg(
                            conn,
                            {
                                "observation/exterior_image_1_left": formatted_obs["observation/exterior_image_1_left"],
                                "observation/wrist_image_left": formatted_obs["observation/wrist_image_left"],
                                "observation/joint_position": formatted_obs["observation/joint_position"],
                                "observation/gripper_position": formatted_obs["observation/gripper_position"],
                                "prompt": formatted_obs["prompt"],
                                "reward": float(reward),
                                "done": bool(done),
                                "truncated": bool(truncated),
                                "success": bool(success),
                                "num_steps": num_steps_executed,
                                "info": status_info,
                            },
                        )

                        # Sleep to match DROID control frequency (from official example)
                        elapsed_time = time.time() - start_time
                        if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

                        if status_info.get("error", False):
                            print("status error goingn to exit")
                            exit()
                    except Exception as e:
                        print(f"Error executing action: {e}")
                        import traceback

                        traceback.print_exc()

                        # Send error response
                        print(f"prompt for send_msg {formatted_obs['prompt']}")
                        send_msg(
                            conn,
                            {
                                "observation/exterior_image_1_left": np.zeros(
                                    (args.resolution, args.resolution, 3), dtype=np.uint8
                                ),
                                "observation/wrist_image_left": np.zeros(
                                    (args.resolution, args.resolution, 3), dtype=np.uint8
                                ),
                                "observation/joint_position": np.zeros(7, dtype=np.float32),
                                "observation/gripper_position": np.zeros(1, dtype=np.float32),
                                "prompt": args.prompt,
                                "reward": 0.0,
                                "done": False,
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


def run_droid_remote_server(args):
    """
    Run a remote server that exposes the DROID robot via socket interface.
    Works with Pinggy and other tunneling services.
    """
    # Initialize robot
    env, camera_config = init_robot(
        args.left_camera_id, args.right_camera_id, args.wrist_camera_id, args.external_camera
    )

    # Create episode state manager
    episode_state = EpisodeState(max_steps=args.max_steps)

    # Start keyboard listener thread
    keyboard_thread = threading.Thread(
        target=keyboard_listener,
        args=(episode_state,),
        kwargs={"env": env, "defer_env_reset": True},
        daemon=True,
    )
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

    print(f"Starting DROID remote server on {host}:{port}...")
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
    parser = argparse.ArgumentParser(description="Run DROID robot as remote server for DSRL training")
    # Hardware parameters (from official example)
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
    parser.add_argument("--prompt", type=str, required=True, help="Task instruction for the robot")
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution (will resize if needed)")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum steps per episode before timeout (default: 600, same as official example)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DROID Robot Remote Server for DSRL Training")
    print("=" * 60)
    print(f"Left camera: {args.left_camera_id}")
    print(f"Right camera: {args.right_camera_id}")
    print(f"Wrist camera: {args.wrist_camera_id}")
    print(f"External camera: {args.external_camera}")
    print(f"Server: 0.0.0.0:{args.server_port}")
    print(f"Task: {args.prompt}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60)

    run_droid_remote_server(args)


if __name__ == "__main__":
    main()
