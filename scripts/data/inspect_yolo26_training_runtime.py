"""Read-only resource/runtime check before an authorized local learning pilot."""
import json
import shutil
import sys

import psutil
import torch
import ultralytics

from scripts.train.train_yolo26_tennis import check_resource_guard

resources = {
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "ultralytics": ultralytics.__version__,
    "free_ram_bytes": psutil.virtual_memory().available,
    "free_disk_bytes": shutil.disk_usage(".").free,
    "free_vram_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(resources, indent=2))
check_resource_guard(resources, gpu_requested=True, role="ball")
