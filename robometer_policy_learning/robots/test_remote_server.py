#!/usr/bin/env python
"""
Test script for remote robot servers (DROID-format or WidowX).
Verifies that the server is running and responding correctly.

Usage:
    # Test DROID-format remote server (e.g. real DROID)
    python robots/test_remote_server.py --url tcp://localhost:6000

    # Test WidowX real server
    python robots/test_remote_server.py --url tcp://localhost:6000 --format simpler

    # Test with Pinggy tunnel
    python robots/test_remote_server.py --url tcp://xyz.a.free.pinggy.link:12345
"""

import argparse
import numpy as np
import sys
import os

# Repo root (parent of robometer_policy_learning/) so the package resolves without an editable install
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from robometer_policy_learning.envs.remote_env import RemoteEnv


def test_connection(server_url: str, obs_format: str = "droid", num_steps: int = 5):
    """Test connection to remote server."""
    print("=" * 60)
    print("Remote Server Connection Test")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    print(f"Observation format: {obs_format}")
    print(f"Test steps: {num_steps}")
    print("=" * 60 + "\n")

    try:
        # Create environment
        print("1. Creating remote environment...")
        env = RemoteEnv(server_url, obs_format=obs_format)
        print("   ✓ Environment created")

        # Reset
        print("\n2. Resetting environment...")
        obs, info = env.reset()
        print("   ✓ Environment reset")
        print(f"   - Observation keys: {obs.keys()}")
        if obs_format == "droid":
            print(f"   - Base image shape: {obs['observation/exterior_image_1_left'].shape}")
            print(f"   - Wrist image shape: {obs['observation/wrist_image_left'].shape}")
            print(f"   - Joint position shape: {obs['observation/joint_position'].shape}")
            print(f"   - Gripper position shape: {obs['observation/gripper_position'].shape}")
        else:
            print(f"   - State shape: {obs['state'].shape}")
            print(f"   - Image shape: {obs['image'].shape}")
        if "prompt" in info:
            print(f"   - Prompt: {info['prompt']}")

        # Execute test steps
        print(f"\n3. Executing {num_steps} test steps...")
        for step in range(num_steps):
            # Random action (small movements)
            action = np.random.randn(env.action_space.shape[0]) * 0.01

            obs, reward, done, truncated, info = env.step(action)

            success_str = "✓" if info.get("success", False) else ""
            print(f"   Step {step + 1}: reward={reward:.3f}, done={done}, truncated={truncated} {success_str}")

            if done or truncated:
                print(f"   Episode ended: success={info.get('success', False)}")
                break

        # Test action chunk (if we haven't ended)
        if not (done or truncated) and num_steps < 3:
            print("\n4. Testing action chunk execution...")
            chunk_actions = np.random.randn(3, env.action_space.shape[0]) * 0.01
            obs, reward, done, truncated, info = env.step_chunk(chunk_actions)
            print(f"   ✓ Executed chunk of {info.get('num_steps', 3)} actions")
            print(f"   reward={reward:.3f}, done={done}, truncated={truncated}")

        # Close
        print("\n5. Closing environment...")
        env.close()
        print("   ✓ Environment closed")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test remote robot server connection")
    parser.add_argument(
        "--url", type=str, default="tcp://localhost:6000", help="Server URL (e.g., tcp://localhost:6000)"
    )
    parser.add_argument("--format", type=str, default="droid", choices=["droid", "simpler"], help="Observation format")
    parser.add_argument("--steps", type=int, default=5, help="Number of test steps to execute")

    args = parser.parse_args()

    success = test_connection(args.url, args.format, args.steps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
