# DROID Multi-Prompt Remote Server

Remote server for real DROID robot that supports multiple prompts, advancing to the next prompt on each RESET. Each episode uses a single prompt, and prompts cycle sequentially across episodes.

Based on Physical Intelligence's official DROID example: https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py

## Prerequisites

1. Install DROID package: https://github.com/droid-dataset/droid
2. Install openpi-client: `pip install openpi-client`
3. Configure camera IDs (find your camera serial numbers)

## Setup Prompts File

Create a text file (e.g., `prompts.txt`) with one prompt per line:

```
put the coke can on the pan
put the ice cream cone on the pan
close the drawer
```

Empty lines are automatically skipped. The server will advance through prompts sequentially on each RESET.

## Running the Server

### With Multiple Prompts (Recommended)

```bash
python robots/droid_remote_server_multi.py \
    --left-camera-id "24259877" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13062452" \
    --external-camera left \
    --server-port 6000 \
    --prompt-file prompts.txt \
    --max-steps 600
```

### With Single Prompt

```bash
python robots/droid_remote_server_multi.py \
    --left-camera-id "24259877" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13062452" \
    --external-camera left \
    --server-port 6000 \
    --prompt "pick up the red block" \
    --max-steps 600
```

## Arguments

### Required Arguments

- `--left-camera-id` - Left camera serial number (e.g., `"24259877"`)
- `--right-camera-id` - Right camera serial number (e.g., `"24514023"`)
- `--wrist-camera-id` - Wrist camera serial number (e.g., `"13062452"`)

**Note:** Either `--prompt` OR `--prompt-file` must be provided (not both).

### Optional Arguments

- `--external-camera` - Which camera to use: `left` or `right` (default: `left`)
- `--server-port` - Port to listen on (default: `6000`)
- `--prompt-file` - Path to file with multiple prompts (one per line)
- `--prompt` - Single task instruction (for single-prompt mode)
- `--max-steps` - Maximum steps per episode before timeout (default: `600`)
- `--resolution` - Image resolution (default: `224`)
- `--reset-function` - Reset function to use:
  - `reset` (default) - Standard robot reset
  - `reset_rewardfm_partial` - Partial reset for rewardfm
  - `reset_rewardfm` - Full reset for rewardfm
  - `none` - No robot reset

## Keyboard Controls

While the server is running (in the server terminal):

- **ENTER** - Start episode (after robot reset)
- **'s'** - Mark current episode as SUCCESS
- **'f'** - Mark current episode as FAILURE
- **'q'** - Quit server

## How It Works

### Prompt Progression

1. **First Episode**: Uses the first prompt from the file (or the single `--prompt`)
2. **Subsequent Episodes**: On each RESET, automatically advances to the next prompt
3. **After Last Prompt**: Loops back to the first prompt

### Episode Flow

1. **RESET Received**: 
   - Checks if more prompts are available
   - If yes: Advances to next prompt
   - If no: Loops back to first prompt
   - Resets robot and episode state
   - Waits for operator to press ENTER

2. **Episode Execution**: 
   - Robot executes actions for current prompt
   - Operator can mark success ('s') or failure ('f')
   - Episode ends on success, failure, or timeout

3. **Next RESET**: 
   - Automatically uses next prompt in sequence
   - Process repeats

### Example Sequence

With prompts file containing:
```
put the coke can on the pan
put the ice cream cone on the pan
```

- **Episode 1**: RESET → Uses "put the coke can on the pan"
- **Episode 2**: RESET → Advances to "put the ice cream cone on the pan"
- **Episode 3**: RESET → Loops back to "put the coke can on the pan"
- **Episode 4**: RESET → Advances to "put the ice cream cone on the pan"
- And so on...

## Complete Example Usage

### Terminal 1: Start Server

```bash
python robots/droid_remote_server_multi.py \
    --left-camera-id "24259877" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13062452" \
    --external-camera left \
    --server-port 6000 \
    --prompt-file prompts.txt \
    --max-steps 600 \
    --reset-function reset
```

### Terminal 2: Optional Tunneling (for remote access)

```bash
ssh -p 443 -R0:localhost:6000 a.pinggy.io
# Note the generated URL: tcp://xyz.a.free.pinggy.link:12345
```

### Terminal 3: Start DSRL Training

```bash
python scripts/train_dsrl.py \
    env_name="DROID_remote" \
    remote_env_url="tcp://localhost:6000" \
    pi0_checkpoint=/path/to/pi0 \
    ...
```

## Differences from Other Servers

### vs. `droid_remote_server.py` (Single Prompt)
- **Multiple prompts**: Supports cycling through multiple tasks
- **Prompt file**: Can load prompts from file
- **Automatic progression**: Advances prompts on RESET

### vs. `droid_remote_server_multi_stage.py` (Multi-Stage)
- **Single-stage episodes**: Each episode uses one prompt (not multi-stage)
- **RESET-based progression**: Advances on RESET, not on success
- **Simpler flow**: No stage transitions or confirmations needed
- **Single step counter**: One counter per episode (not per prompt)

## Features

- ✅ Multiple prompts from file
- ✅ Automatic prompt cycling
- ✅ Single prompt mode (backward compatible)
- ✅ Action chunking support
- ✅ Keyboard controls for success/failure
- ✅ Episode timeout handling
- ✅ Configurable reset functions
- ✅ Works with Pinggy and other tunneling services

## Troubleshooting

### Camera Issues
- Ensure camera IDs are correct (check serial numbers)
- Verify cameras are connected and accessible
- Check camera permissions

### Connection Issues
- Verify server port is not in use: `lsof -i :6000`
- Check firewall settings
- For remote access, ensure tunneling is set up correctly

### Prompt Issues
- Ensure prompt file exists and is readable
- Check that prompts file has at least one non-empty line
- Verify file encoding is UTF-8

## See Also

- Multi-stage server: `droid_remote_server_multi_stage.py` (for sequential multi-stage tasks)
- Single prompt server: `droid_remote_server.py` (for single task)
- Main robots README: [README.md](README.md)
- Based on: [Official Pi0 DROID example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py)
