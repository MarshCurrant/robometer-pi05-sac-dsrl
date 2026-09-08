# DROID Multi-Stage Remote Server

Multi-stage remote server for real DROID robot that supports sequential prompts that advance on success, with per-prompt step counting.

## Prerequisites

1. Install DROID package: https://github.com/droid-dataset/droid
2. Install openpi-client: `pip install openpi-client`
3. Configure camera IDs (find your camera serial numbers)

## Setup Prompts File

Create a text file (e.g., `prompts.txt`) with one prompt per line:

```
pick up the red block
place the block on the shelf
close the drawer
```

Empty lines are automatically skipped. The server will advance through prompts sequentially as each stage is completed.

## Running the Server

```bash
python robots/droid_remote_server_multi_stage.py \
    --left-camera-id "23804457" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13263313" \
    --external-camera left \
    --server-port 6000 \
    --prompt-file prompts.txt \
    --max-steps 500
```

### Required Arguments

- `--left-camera-id` - Left camera serial number
- `--right-camera-id` - Right camera serial number  
- `--wrist-camera-id` - Wrist camera serial number

### Optional Arguments

- `--external-camera` - Which camera to use: `left` or `right` (default: `left`)
- `--server-port` - Port to listen on (default: `6000`)
- `--prompt-file` - Path to prompts file (required if `--prompt` not provided)
- `--prompt` - Single prompt (for backward compatibility)
- `--max-steps` - Max steps per prompt before timeout (default: `600`)
- `--resolution` - Image resolution (default: `224`)

## Keyboard Controls

While the server is running:

- **ENTER** - Start episode (after robot reset)
- **'s'** - Mark current stage as SUCCESS (advances to next prompt)
- **'f'** - Mark episode as FAILURE (ends episode)
- **'c'** - Confirm advance to next stage (during transition)
- **'q'** - Quit server

## How It Works

1. **Episode Start**: Robot resets, starts with first prompt from file
2. **Stage Execution**: Robot executes actions for current prompt
3. **Stage Success**: Press 's' when current stage is complete
4. **Stage Transition**: Press 'c' to confirm advancing to next prompt
5. **Episode End**: Episode ends when:
   - All stages completed (loops back to first stage)
   - Failure marked ('f')
   - Max steps reached per prompt

Each prompt has its own step counter, so `--max-steps` applies per stage, not per episode.

## Example Usage

### Terminal 1: Start Server
```bash
python robots/droid_remote_server_multi_stage.py \
    --left-camera-id "23804457" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13263313" \
    --external-camera left \
    --server-port 6000 \
    --prompt-file prompts.txt \
    --max-steps 400 \
    --reset-between-stages
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

## Differences from Single-Stage Server

- **Multi-stage prompts**: Sequential tasks that advance on success
- **Per-prompt step counting**: Each stage has its own step counter
- **Stage transitions**: Manual confirmation between stages
- **Prompt file**: Uses text file instead of single `--prompt` argument

## Testing

Test the connection:
```bash
python robots/test_remote_server.py \
    --url tcp://localhost:6000 \
    --format droid \
    --steps 5
```

## See Also

- Main robots README: [README.md](README.md)
- Single-stage server: `droid_remote_server.py`
- Based on: [Official Pi0 DROID example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py)

