"""Record this receiving laptop and inspect the preserved starter checkpoints."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
import ultralytics

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
record = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'ultralytics': ultralytics.__version__, 'cuda_available': torch.cuda.is_available(), 'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True), 'starter_models': {}}
assert torch.cuda.is_available()
record['cuda_smoke'] = (torch.ones((128,128), device='cuda') @ torch.ones((128,128), device='cuda')).mean().item()
for path in (REPO / 'continuation_packages/starter_models').glob('*.pt'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema') or checkpoint.get('model')
    record['starter_models'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size, 'epoch': checkpoint.get('epoch'), 'version': checkpoint.get('version'), 'names': getattr(model, 'names', None), 'train_args': checkpoint.get('train_args')}
(OUT / 'environment_preflight.json').write_text(json.dumps(record, indent=2, default=str) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2, default=str))
