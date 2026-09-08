import torch
import numpy as np
import os


def move_to_device(data, device):
    """Move data to the device."""
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def convert_to_tensor(data):
    """Convert data to a tensor."""
    if isinstance(data, dict):
        return {k: convert_to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_tensor(v) for v in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return data


def convert_to_numpy(data):
    """Recursively convert torch.Tensors to numpy arrays, and lists/dicts containing tensors."""
    if isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data


def is_fp16_supported() -> bool:
    if not torch.cuda.is_available():
        return True
    # See https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
    # FP16 on compute capability 6.x is deprecated
    allow_deprecated_fp16 = os.environ.get("DS_ALLOW_DEPRECATED_FP16", "0") == "1"
    major, _ = torch.cuda.get_device_capability()
    if major >= 7:
        return True
    elif major == 6 and allow_deprecated_fp16:
        return True
    else:
        return False
