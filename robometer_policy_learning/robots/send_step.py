#!/usr/bin/env python
"""
Directly send STEP commands to DROID remote server.
This connects to the server and sends STEP commands without using RemoteEnv wrapper.

Usage:
    python robots/send_step.py --url tcp://localhost:6000
"""

import argparse
import socket
import pickle
import struct
import numpy as np
from urllib.parse import urlparse


def recvall(sock, n):
    """Helper to receive n bytes or return None if EOF is hit."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def send_msg(sock, msg):
    """Send a message with length prefix over socket."""
    msg_bytes = pickle.dumps(msg)
    msg_len = struct.pack(">I", len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)


def recv_msg(sock):
    """Receive a length-prefixed message from socket."""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    return pickle.loads(recvall(sock, msglen))


def main():
    parser = argparse.ArgumentParser(description="Send STEP commands directly to DROID remote server")
    parser.add_argument(
        "--url", type=str, default="tcp://localhost:6000", help="Server URL (e.g., tcp://localhost:6000)"
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of STEP commands to send")

    args = parser.parse_args()

    # Parse URL
    parsed = urlparse(args.url)
    host = parsed.hostname
    port = parsed.port

    if not host or not port:
        print(f"Error: Invalid server URL: {args.url}")
        print("Expected format: tcp://host:port")
        return

    print("=" * 60)
    print("Direct STEP Command Test")
    print("=" * 60)
    print(f"Connecting to: {host}:{port}")
    print(f"Will send {args.steps} STEP commands")
    print("=" * 60 + "\n")

    try:
        # Connect to server
        print("1. Connecting to server...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print("   ✓ Connected")

        # Send RESET
        print("\n2. Sending RESET command...")
        send_msg(sock, {"type": "RESET"})
        print("   ✓ RESET sent")

        # # Receive reset response
        print("   Waiting for reset response...")
        reset_response = recv_msg(sock)
        if reset_response:
            print("   ✓ Reset response received")
            print(f"   - Keys: {list(reset_response.keys())}")
            if "prompt" in reset_response:
                print(f"   - Prompt: {reset_response['prompt']}")
        else:
            print("   ✗ No response received")
            return

        # Send STEP commands
        print(f"\n3. Sending {args.steps} STEP commands...")
        for step in range(args.steps):
            # Create small action: [joint_velocity (7)] (no gripper component)
            action = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
            
            print(f"\n   Step {step+1}:")
            print(f"   - Action shape: {action.shape}")
            print(f"   - Action: {action}")

            # Send STEP command
            send_msg(sock, {"type": "STEP", "action": action.tolist(), "need_obs": True})
            print("   - STEP command sent")

            # Receive response
            print("   - Waiting for response...")
            response = recv_msg(sock)
            if response:
                print("   ✓ Response received")
                print(f"   - Keys: {list(response.keys())}")
                if "reward" in response:
                    print(f"   - Reward: {response['reward']}")
                if "done" in response:
                    print(f"   - Done: {response['done']}")
                if "truncated" in response:
                    print(f"   - Truncated: {response['truncated']}")
                if "success" in response:
                    print(f"   - Success: {response['success']}")
                if "observation/joint_position" in response:
                    print(f"   - Joint position: {response['observation/joint_position']}")
                if "info" in response:
                    print(f"   - Info: {response['info']}")
            else:
                print("   ✗ No response received")
                break

        # Close connection
        print("\n4. Closing connection...")
        sock.close()
        print("   ✓ Closed")

        print("\n" + "=" * 60)
        print("✓ STEP test completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
