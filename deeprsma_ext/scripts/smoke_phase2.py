"""Phase 2 integration smoke test.

Builds DeepRSMA_ext (with StructureAwareAdapter), runs one forward+backward
on a few batches from the All_sf dataset, reports params + peak VRAM + step time.

Run from:
  cd C:\\Users\\yanbin\\Desktop\\MM
  python deeprsma_ext/scripts/smoke_phase2.py
"""
import os
import sys
import time
from pathlib import Path

# seqfold importlib.metadata backport for Python 3.7 (only needed if fold_all is imported)
try:
    import importlib.metadata as _md  # noqa: F401
except ImportError:
    import importlib_metadata
    sys.modules['importlib.metadata'] = importlib_metadata

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Project root
ROOT = Path(__file__).resolve().parents[2]   # C:\Users\yanbin\Desktop\MM
sys.path.insert(0, str(ROOT))
DEEPRSMA = ROOT / "DeepRSMA"
sys.path.insert(0, str(DEEPRSMA))

# Switch CWD to DeepRSMA so its relative paths ('data/RSM_data/...') resolve
os.chdir(DEEPRSMA)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from data import RNA_dataset, Molecule_dataset, WordVocab

from deeprsma_ext.data.ss_cache import SSCache
from deeprsma_ext.models.deeprsma_ext import DeepRSMA_ext


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

BATCH_SIZE = 16
hidden_dim = 128
RNA_type = 'All_sf'

print("\n[1/4] Loading SSCache...")
t0 = time.time()
ss_cache = SSCache(truncate_to=511, verbose=True)
print(f"  SSCache load time: {time.time() - t0:.1f}s")

print("\n[2/4] Loading RNA + Mole datasets...")
t0 = time.time()
rna_ds = RNA_dataset(RNA_type)
mole_ds = Molecule_dataset(RNA_type)
print(f"  RNA dataset: {len(rna_ds)}, Mole dataset: {len(mole_ds)}")
print(f"  Dataset load time: {time.time() - t0:.1f}s")

# Check that all t_ids in the dataset are covered by the SS cache
miss = [d.t_id for d in rna_ds if not ss_cache.has(d.t_id)]
print(f"  RNAs missing from SS cache: {len(miss)}")
if miss:
    print(f"    first few: {miss[:5]}")


class CustomDualDataset(Dataset):
    def __init__(self, d1, d2):
        self.d1, self.d2 = d1, d2
    def __getitem__(self, i):
        return self.d1[i], self.d2[i]
    def __len__(self):
        return len(self.d1)


print("\n[3/4] Building DeepRSMA_ext...")
model = DeepRSMA_ext(
    hidden_dim=hidden_dim,
    ss_cache=ss_cache,
    llm_dim=640,
    adapter_layers=2,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_adapter = sum(p.numel() for p in model.rna_graph_model.adapter.parameters())
print(f"  Total params: {n_params/1e6:.2f}M  (adapter: {n_adapter/1e6:.2f}M)")

# Use first 32 samples for smoke test
N = 32
small = CustomDualDataset(rna_ds[:N], mole_ds[:N])
loader = DataLoader(small, batch_size=BATCH_SIZE, num_workers=0, drop_last=False, shuffle=False)

opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-7)
loss_fn = nn.MSELoss()

print(f"\n[4/4] Running 2 smoke-test forward+backward passes (batch_size={BATCH_SIZE})...")
torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
for step, (br, bm) in enumerate(loader):
    t0 = time.time()
    opt.zero_grad()
    pred = model(br.to(device), bm.to(device))
    loss = loss_fn(pred.squeeze(1), br.y)
    loss.backward()
    opt.step()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
    print(f"  step {step}: loss={loss.item():.4f}  time={dt:.2f}s  peak_VRAM={mem:.2f}GB")

print("\n=== PHASE 2 SMOKE TEST PASSED ===")
if torch.cuda.is_available():
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
