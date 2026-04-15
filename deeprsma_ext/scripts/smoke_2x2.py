"""Verify all 4 configurations of DeepRSMA_ext run one forward+backward.

Configs tested:
  (use_adapter=F, use_bias=F): pure baseline
  (use_adapter=T, use_bias=F): +adapter
  (use_adapter=F, use_bias=T): +bias
  (use_adapter=T, use_bias=T): full
"""
import os
import sys
import time
from pathlib import Path

try:
    import importlib.metadata as _md  # noqa
except ImportError:
    import importlib_metadata
    sys.modules['importlib.metadata'] = importlib_metadata

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEEPRSMA = ROOT / "DeepRSMA"
sys.path.insert(0, str(DEEPRSMA))
os.chdir(DEEPRSMA)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from data import RNA_dataset, Molecule_dataset, WordVocab   # noqa

from deeprsma_ext.data.ss_cache import SSCache
from deeprsma_ext.models.deeprsma_ext import DeepRSMA_ext


class CustomDualDataset(Dataset):
    def __init__(self, d1, d2):
        self.d1, self.d2 = d1, d2
    def __getitem__(self, i): return self.d1[i], self.d2[i]
    def __len__(self): return len(self.d1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("\nLoading SSCache + datasets...")
ss_cache = SSCache(truncate_to=511, verbose=False)
rna_ds = RNA_dataset('All_sf')
mole_ds = Molecule_dataset('All_sf')
small = CustomDualDataset(rna_ds[:16], mole_ds[:16])
loader = DataLoader(small, batch_size=16, num_workers=0, drop_last=False, shuffle=False)
print(f"  RNA {len(rna_ds)} / Mole {len(mole_ds)} / SSCache {len(ss_cache.edge_index)}")

loss_fn = nn.MSELoss()
configs = [
    ("baseline",       False, False),
    ("+adapter",       True,  False),
    ("+bias",          False, True),
    ("full",           True,  True),
]
print()
for label, ua, ub in configs:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model = DeepRSMA_ext(hidden_dim=128, ss_cache=ss_cache, use_adapter=ua, use_bias=ub).to(device)
    n = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=5e-4)

    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    for br, bm in loader:
        t0 = time.time()
        opt.zero_grad()
        pred = model(br.to(device), bm.to(device))
        loss = loss_fn(pred.squeeze(1), br.y)
        loss.backward()
        opt.step()
        dt = time.time() - t0
        break   # just one batch
    mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
    print(f"  {label:<15} ua={ua} ub={ub}  params={n/1e6:.2f}M  loss={loss.item():.3f}  dt={dt:.2f}s  peak_VRAM={mem:.2f}GB")

    del model, opt

print("\n=== 2x2 SMOKE TEST PASSED ===")
