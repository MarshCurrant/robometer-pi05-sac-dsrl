"""
Shared utilities for remote robot servers (WidowX, DROID real robot, etc.).
Used by both widowx_remote_server.py and droid_remote_server.py.
"""

import socket
import pickle
import struct
import threading
import time
import sys
import select


class EpisodeState:
    """Shared state for episode management with keyboard input."""

    def __init__(self, max_steps=None):
        self.step_count = 0
        self.max_steps = max_steps
        self.success = None  # None, True (success), or False (failure)
        self.reset_env_requested = False  # Keyboard thread can request main-loop reset
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.step_count = 0
            self.success = None
            self.reset_env_requested = False

    def request_env_reset(self):
        with self.lock:
            self.reset_env_requested = True

    def consume_env_reset_request(self):
        with self.lock:
            requested = self.reset_env_requested
            self.reset_env_requested = False
            return requested

    def increment_step(self):
        with self.lock:
            self.step_count += 1

    def get_step_count(self):
        with self.lock:
            return self.step_count

    def mark_success(self):
        with self.lock:
            self.success = True
            print("\n✓ Episode marked as SUCCESS by operator")

    def mark_failure(self):
        with self.lock:
            self.success = False
            print("\n✗ Episode marked as FAILURE by operator")

    def get_status(self):
        """Returns (done, truncated, success, info)

        MDP semantics:
        - done=True: Episode reached natural terminal state
        - truncated=True: Episode cut short (operator label, timeout, etc.)

        For real robot with manual labeling, we use truncated=True since
        the operator is intervening to end the episode, not a natural termination.
        """
        with self.lock:
            if self.success is not None:
                # Operator marked success/failure
                # If operator marked SUCCESS, treat as terminal success (done=True).
                # If operator marked FAILURE, treat as truncated cutoff (truncated=True).
                if self.success:
                    return True, False, True, {"is_success": True}
                return False, True, False, {"is_success": False}
            elif self.max_steps and self.step_count >= self.max_steps:
                # Timeout - episode cut short
                return False, True, False, {"timeout": True}
            else:
                # Continue - episode still running
                return False, False, False, {}


class MultiStageEpisodeState(EpisodeState):
    """Extended episode state for multi-stage prompts with transitions."""
    def __init__(self, max_steps=None):
        super().__init__(max_steps)
        self.prompt_transition_ready = False  # Flag for prompt transition confirmation
        self.prompt_step_count = 0  # Per-prompt step counter
    
    def reset(self):
        with self.lock:
            self.step_count = 0
            self.prompt_step_count = 0
            self.success = None
            self.prompt_transition_ready = False
    
    def increment_step(self):
        """Increment both episode and per-prompt step counters."""
        with self.lock:
            self.step_count += 1
            self.prompt_step_count += 1
    
    def increment_prompt_step(self):
        """Increment only the per-prompt step counter (for special cases)."""
        with self.lock:
            self.prompt_step_count += 1
    
    def get_prompt_step_count(self):
        with self.lock:
            return self.prompt_step_count
    
    def reset_prompt_step_count(self):
        with self.lock:
            self.prompt_step_count = 0
    
    def confirm_advance(self):
        """Confirm advance to next prompt stage."""
        with self.lock:
            self.prompt_transition_ready = True
    
    def clear_transition(self):
        """Clear the transition flag."""
        with self.lock:
            self.prompt_transition_ready = False
    
    def is_transition_ready(self):
        """Check if transition is confirmed."""
        with self.lock:
            return self.prompt_transition_ready
    
    def get_status(self, prompt_max_steps=None):
        """Returns (done, truncated, success, info)
        
        Extended version that checks per-prompt timeout.
        
        Args:
            prompt_max_steps: Max steps for current prompt (if None, uses episode max_steps)
        """
        with self.lock:
            if self.success is not None:
                # Operator marked success/failure
                # For multi-stage: success means current stage succeeded (not episode end)
                # Failure still means episode end
                if self.success:
                    # Success - current stage completed, but episode continues
                    return False, False, True, {'is_success': True, 'stage_complete': True}
                # Failure - episode ends
                return False, True, False, {'is_success': False}
            
            # Check per-prompt timeout
            max_steps_to_check = prompt_max_steps if prompt_max_steps is not None else self.max_steps
            if max_steps_to_check and self.prompt_step_count >= max_steps_to_check:
                # Timeout for current prompt
                return False, True, False, {'timeout': True, 'prompt_timeout': True}
            elif self.max_steps and self.step_count >= self.max_steps:
                # Overall episode timeout
                return False, True, False, {'timeout': True}
            else:
                # Continue - episode still running
                return False, False, False, {}


def keyboard_listener(episode_state, use_raw_mode=True, multi_stage=False, env=None, defer_env_reset=False):
    """
    Listen for keyboard input in a separate thread.

    Args:
        episode_state: EpisodeState instance to update
        use_raw_mode: If True, use raw terminal mode for single-key input (Unix only)
                      If False, require Enter after each key
        multi_stage: If True, enable multi-stage controls (c for confirm advance)
        env: Environment instance to use for manual reset
        defer_env_reset: If True, request reset for main loop thread to apply
    """
    print("\n" + "=" * 60)
    print("Keyboard Controls:")
    if multi_stage:
        print("  's' - Mark current stage as SUCCESS")
        print("  'f' - Mark episode as FAILURE (ends episode)")
        print("  'c' - Confirm advance to next stage (during transition)")
    else:
        print("  's' - Mark episode as SUCCESS")
        print("  'f' - Mark episode as FAILURE")
    if env is not None:
        print("  'r' - Perform a manual reset (in case robot is stuck) without starting a new episode. Only if supported by the server.")
    else:
        print("  'r' - Not supported for this server.")
    print("  'q' - Quit server")
    if not use_raw_mode:
        print("  (Press Enter after each key)")
    print("=" * 60 + "\n")

    # Debouncing: track last press time for 's' and 'f' keys
    last_key_press_time = {'s': 0, 'f': 0}
    debounce_cooldown = 2  # seconds - ignore repeated presses within 2s

    # Try to set up raw terminal mode for single-key input (Unix only)
    old_settings = None
    if use_raw_mode and sys.platform != "win32":
        try:
            import tty
            import termios

            old_settings = termios.tcgetattr(sys.stdin)
            # Set terminal to cbreak mode (single char input, no echo issues)
            tty.setcbreak(sys.stdin.fileno())
        except Exception as e:
            print(f"Warning: Could not enable raw terminal mode: {e}")
            print("Falling back to line mode (press Enter after each key)")
            old_settings = None

    try:
        while True:
            try:
                if sys.platform != "win32" and old_settings is not None:
                    # Unix with raw mode - single character read
                    i, o, e = select.select([sys.stdin], [], [], 0.1)
                    if i:
                        key = sys.stdin.read(1).lower()
                    else:
                        continue
                elif sys.platform != "win32":
                    # Unix without raw mode - need Enter
                    i, o, e = select.select([sys.stdin], [], [], 0.1)
                    if i:
                        key = sys.stdin.readline().strip().lower()
                        if not key:
                            continue
                    else:
                        continue
                else:
                    # Windows - use msvcrt for single key input
                    try:
                        import msvcrt

                        if msvcrt.kbhit():
                            key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                        else:
                            time.sleep(0.1)
                            continue
                    except ImportError:
                        # Fallback to blocking input
                        key = input().strip().lower()

                current_time = time.time()
                
                if key == "s":
                    # Check debounce cooldown
                    if current_time - last_key_press_time['s'] < debounce_cooldown:
                        continue  # Ignore double-click
                    last_key_press_time['s'] = current_time
                    episode_state.mark_success()
                    # For async runtimes, perform reset from main loop thread.
                    if env is not None and (multi_stage or defer_env_reset):
                        episode_state.request_env_reset()
                    elif env is not None:
                        env.reset()
                elif key == "f":
                    # Check debounce cooldown
                    if current_time - last_key_press_time['f'] < debounce_cooldown:
                        continue  # Ignore double-click
                    last_key_press_time['f'] = current_time
                    episode_state.mark_failure()
                    # For async runtimes, perform reset from main loop thread.
                    if env is not None and (multi_stage or defer_env_reset):
                        episode_state.request_env_reset()
                    elif env is not None:
                        env.reset()
                elif key == 'c' and multi_stage and hasattr(episode_state, 'confirm_advance'):
                    episode_state.confirm_advance()
                    print("\n✓ Advance to next stage confirmed")
                elif key == 'r' and env is not None:
                    print("\nPerforming manual reset...")
                    if multi_stage or defer_env_reset:
                        episode_state.request_env_reset()
                    else:
                        env.reset()
                elif key == 'q':
                    print("\nQuitting server...")
                    # Restore terminal before exit
                    if old_settings is not None:
                        import termios

                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    import os

                    os._exit(0)

            except Exception as e:
                print(f"Keyboard listener error: {e}")
                time.sleep(0.1)
    finally:
        # Restore terminal settings on exit
        if old_settings is not None:
            try:
                import termios

                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass


def send_msg(sock, msg):
    """Send a message with length prefix over socket."""
    msg_bytes = pickle.dumps(msg)
    msg_len = struct.pack(">I", len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)


def recv_msg(sock):
    """Receive a length-prefixed message from socket."""
    # Read message length
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    # Read the message data
    return pickle.loads(recvall(sock, msglen))


def recvall(sock, n):
    """Helper to receive n bytes or return None if EOF is hit."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)
