"""Smoke test for all Phase 4 ablation flags.

Verifies DeepRSMA_ext builds and runs forward+backward under:
  - adapter ablations: no_gcn, no_struct_emb, 3-layer
  - bias ablations: lambda_fixed, exposure_smooth, bias_direction variants
"""
import os, sys, time
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


class CD(Dataset):
    def __init__(self, d1, d2): self.d1, self.d2 = d1, d2
    def __getitem__(self, i): return self.d1[i], self.d2[i]
    def __len__(self): return len(self.d1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("\nLoading SSCache + datasets...")
ss_cache = SSCache(truncate_to=511, verbose=False)
ss_sm3 = SSCache(truncate_to=511, exposure_smooth=3, verbose=False)
rna_ds = RNA_dataset('All_sf')
mole_ds = Molecule_dataset('All_sf')
loader = DataLoader(CD(rna_ds[:16], mole_ds[:16]), batch_size=16, num_workers=0)
print(f"  RNA={len(rna_ds)} / Mole={len(mole_ds)}")

configs = [
    # (label, DeepRSMA_ext kwargs)
    ("adapter-default",           dict(use_adapter=True,  use_bias=False)),
    ("adapter-nogcn",             dict(use_adapter=True,  use_bias=False, adapter_use_gcn=False)),
    ("adapter-nostructemb",       dict(use_adapter=True,  use_bias=False, adapter_use_struct_emb=False)),
    ("adapter-L3",                dict(use_adapter=True,  use_bias=False, adapter_layers=3)),
    ("bias-default",              dict(use_adapter=False, use_bias=True)),
    ("bias-lambdaFixed",          dict(use_adapter=False, use_bias=True, lambda_trainable=False)),
    ("bias-mole_query",           dict(use_adapter=False, use_bias=True, bias_direction="mole_query")),
    ("bias-rna_query",            dict(use_adapter=False, use_bias=True, bias_direction="rna_query")),
    ("full-all-ablations",        dict(use_adapter=True,  use_bias=True,
                                        adapter_use_gcn=False, adapter_use_struct_emb=False,
                                        lambda_trainable=False, bias_direction="mole_query")),
]

loss_fn = nn.MSELoss()
print()
for lbl, cfg in configs:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    use_sm = "smooth3" in lbl
    cache = ss_sm3 if use_sm else ss_cache
    model = DeepRSMA_ext(hidden_dim=128, ss_cache=cache, **cfg).to(device)
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
        break
    mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
    n = sum(p.numel() for p in model.parameters())
    print(f"  {lbl:<25}  params={n/1e6:.2f}M  loss={loss.item():.3f}  dt={dt:.2f}s  VRAM={mem:.2f}GB")
    del model, opt

# Also test exposure smoothing (separate ss_cache)
print()
m = DeepRSMA_ext(hidden_dim=128, ss_cache=ss_sm3, use_adapter=True, use_bias=True).to(device)
opt = optim.Adam(m.parameters(), lr=5e-4)
for br, bm in loader:
    opt.zero_grad()
    pred = m(br.to(device), bm.to(device))
    loss = loss_fn(pred.squeeze(1), br.y)
    loss.backward()
    opt.step()
    break
print(f"  full+smooth3             params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  loss={loss.item():.3f}")

print("\n=== PHASE 4 ABLATION SMOKE TEST PASSED ===")
