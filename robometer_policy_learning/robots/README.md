## Remote Robot Servers for DSRL Training

**Canonical doc** for TCP remote **DROID** and **WidowX** servers in this repo: protocol, CLI examples, observation formats, and troubleshooting. For WidowX bridge training, BRIDGE controller setup, and episode UX, see [REAL_ROBOT_README.md](../../REAL_ROBOT_README.md). High-level index: [README — Remote Robot Training](../../README.md#remote-robot-training).

The Python modules in this directory implement those servers for DSRL over TCP.

## Overview

The remote server architecture allows you to:
- **Train anywhere**: Run training on cloud/local machine, robot on dedicated hardware
- **Consistent protocol**: Use the same training client across supported robot backends
- **Remote access**: Tunnel connections through Pinggy for remote training
- **Manual labeling**: Keyboard controls for success/failure annotation

## Available Servers

We provide 2 remote servers:
1. **DROID Real Robot** - Physical Franka Panda with DROID setup
2. **WidowX Real Robot** - Physical WidowX 250 6DOF manipulator

Both servers use the same TCP socket protocol and share common utilities.

### 1. DROID Real Robot Server

**File**: `droid_remote_server.py`  
**Environment**: Physical DROID robot (Franka Panda)  
**Robot**: DROID manipulator in real world  
**Observation format**: DROID format (matches [droid_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py))  
**Based on**: [Official Pi0 DROID example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py)

```bash
# From repository root
uv run python robometer_policy_learning/robots/droid_remote_server.py \
    --left-camera-id "23804457" \
    --wrist-camera-id "13263313" \
    --external-camera left \
    --server-port 6000 \
    --prompt "pick up the red block" \
    --max-steps 600
```

**Key features**:
- Real-world DROID robot control
- Joint velocity action space (15 Hz)
- Stereo + wrist cameras
- Manual success labeling (keyboard)
- Based on official Pi0 implementation

**Setup**: Requires `droid` and `openpi-client` packages

### 2. WidowX Real Robot Server

**File**: `widowx_remote_server.py`  
**Environment**: Physical WidowX robot  
**Robot**: WidowX 250 6DOF manipulator  
**Observation format**: WidowX / bridge-style (state vector + single RGB image)

```bash
# From repository root
uv run python robometer_policy_learning/robots/widowx_remote_server.py \
    --robot-ip localhost \
    --robot-port 5556 \
    --server-port 6000 \
    --prompt "pick up the red block" \
    --max-steps 40
```

**Key features**:
- Real-world training
- Manual success labeling (keyboard)
- Safety monitoring
- Physical robot control

**Setup**: Requires `widowx_envs` package

## Architecture

### Communication Protocol

The remote servers use the same TCP socket protocol with length-prefixed pickle messages:

**Commands**:
1. **RESET** - Reset environment, get initial observation
2. **STEP** - Execute single action, get next observation
3. **STEP_CHUNK** - Execute multiple actions efficiently (single round-trip)
4. **CLOSE** - Close connection

**Response**:
```python
{
    'observation/...': ...,  # Observation data
    'reward': float,         # Reward signal
    'done': bool,            # Terminal state
    'truncated': bool,       # Episode cut short
    'success': bool,         # Task success
    'info': dict,            # Additional info
}
```

### Shared Code

**`remote_server_utils.py`** - Common utilities:
- `EpisodeState` - Episode state management
- `keyboard_listener` - Cross-platform keyboard input
- `send_msg/recv_msg` - Socket communication

**`../envs/remote_env.py`** - Client-side Gymnasium wrapper:
- `RemoteEnv` - Connects to any remote server
- Automatic reconnection
- Format conversion (DROID ↔ WidowX-style observations)

## Usage

### 1. Start Server

```bash
# From repository root (DROID example)
uv run python robometer_policy_learning/robots/droid_remote_server.py \
    --left-camera-id "23804457" \
    --wrist-camera-id "13263313" \
    --external-camera left \
    --server-port 6000 \
    --prompt "pick up the red block" \
    --max-steps 600
```

### 2. Optional Tunneling

```bash
# Terminal 2: Create Pinggy tunnel for remote access
ssh -p 443 -R0:localhost:6000 a.pinggy.io
# Note the generated URL: tcp://xyz.a.free.pinggy.link:12345
```

### 3. Run Training

```bash
# Terminal 3: Start DSRL training (from repository root)
uv run python scripts/train_dsrl.py \
    env_name="DROID_remote" \
    remote_env_url="tcp://localhost:6000" \
    pi0_checkpoint=gs://openpi-assets/checkpoints/pi0_droid/
```

### 4. Monitor & Control (Server Terminal)

Keyboard controls while server is running:
- **'s'** - Mark episode as SUCCESS
- **'f'** - Mark episode as FAILURE
- **'q'** - Quit server

On timeout, you'll be prompted to label the episode.

## Verify connectivity

Use the bundled client to confirm the server speaks the expected protocol:

```bash
# From repository root
uv run python robometer_policy_learning/robots/test_remote_server.py \
    --url tcp://localhost:6000 \
    --format droid \
    --steps 5
```

This will:
1. Connect to server
2. Reset environment
3. Execute test steps
4. Test action chunking
5. Close connection

## Observation Formats

### DROID Format (real DROID robot)

```python
{
    'observation/exterior_image_1_left': (224, 224, 3) uint8,  # Base camera
    'observation/wrist_image_left': (224, 224, 3) uint8,       # Wrist camera
    'observation/joint_position': (7,) float32,                # Joint angles
    'observation/gripper_position': (1,) float32,              # Gripper state
    'prompt': str,                                              # Task instruction
}
```

### WidowX / bridge observation format

```python
{
    'state': (7,) float32,              # End-effector pose [x, y, z, rx, ry, rz, gripper]
    'image': (224, 224, 3) uint8,       # Single camera view
    'prompt': str,                       # Task instruction
}
```

The `RemoteEnv` wrapper handles format conversion automatically.

## File Structure

```
robometer_policy_learning/robots/
├── README.md                      # This file
├── remote_server_utils.py         # Shared utilities (socket, keyboard, episode state)
├── droid_remote_server.py         # DROID real robot server (Franka Panda)
├── widowx_remote_server.py        # WidowX real robot server
├── test_remote_server.py          # Connection test script
├── DROID_MULTI_README.md          # Multi-task DROID (async relabel)
└── DROID_MULTI_STAGE_README.md    # Multi-stage DROID remote
```

## Comparison

| Feature | DROID Real | WidowX Real |
|---------|-----------|-------------|
| **Robot** | Franka Panda | WidowX 250 |
| **Speed** | ~15 Hz | ~5 Hz |
| **Setup** | Physical DROID | Physical WidowX |
| **Success Detection** | Manual (keyboard) | Manual (keyboard) |
| **Safety** | Requires monitoring | Requires monitoring |
| **Cameras** | Stereo + wrist (3) | Single camera |
| **Action Space** | Joint velocity | Joint position |
| **Observation** | DROID format | WidowX (state + image) |
| **Use Case** | Real-world DROID | Real-world WidowX |

**Both use**:
- Same socket protocol
- Same keyboard controls
- Same training code
- Same episode semantics (done/truncated/success)

## Adding New Servers

To add a new robot server:

1. **Import shared utilities**:
   ```python
   from remote_server_utils import EpisodeState, keyboard_listener, send_msg, recv_msg
   ```

2. **Implement server functions**:
   - `init_robot()` - Initialize robot hardware
   - `format_observation_for_dsrl()` - Format observations
   - `handle_client()` - Handle RESET/STEP/STEP_CHUNK/CLOSE

3. **Follow the pattern**:
   - Use `EpisodeState` for tracking
   - Start `keyboard_listener` thread
   - Handle timeout confirmations
   - Support both STEP and STEP_CHUNK

See `droid_remote_server.py` or `widowx_remote_server.py` as templates.

## Troubleshooting

### Connection Failed

- Check server is running: `netstat -an | grep 6000`
- Check firewall allows TCP on port
- For Pinggy: Verify tunnel URL is correct

### Slow Performance

- Check network latency between training client and robot server
- Use `STEP_CHUNK` for multiple actions (faster than multiple STEP calls)

### Success Detection Issues

- Use keyboard controls ('s'/'f') on the server to manually label episodes

### Image Issues

- Check resolution matches: `--resolution 224`
- Verify camera topics/devices are correct
- Check image format (uint8, HWC)

## References

- [OpenPI DROID Policy](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py) - Reference implementation
- [Pinggy](https://pinggy.io/) - Secure tunneling service
- Main README: [../../README.md](../../README.md) (includes [Remote Robot Training](../../README.md#remote-robot-training) and [Remote Reward Relabeling](../../README.md#remote-reward-relabeling))
- WidowX bridge + episode workflow: [REAL_ROBOT_README.md](../../REAL_ROBOT_README.md)
- Multi-task DROID: [DROID_MULTI_README.md](DROID_MULTI_README.md), [DROID_MULTI_STAGE_README.md](DROID_MULTI_STAGE_README.md)

