"""
In-memory cache of secondary-structure info for all RNAs.

Loaded once at the start of training from cache/rnafold/*.json.
Provides per-RNA lookup of (edge_index, struct_types, exposure) without
interfering with DeepRSMA's PyG batching of the contact-map graph.

Supports Phase 4 ablation: exposure smoothing via sliding average window.
"""
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from deeprsma_ext.structure.ss_graph import (
    load_fold, build_ss_edge_index, get_struct_types, get_exposure,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "rnafold"


def smooth_exposure(exp: torch.Tensor, window: int) -> torch.Tensor:
    """Sliding-window mean smoothing on 1-D exposure tensor. window=0 or 1 = no-op."""
    if window <= 1:
        return exp
    L = exp.size(0)
    if L == 0:
        return exp
    # pad with edge values, then mean-pool with stride 1
    pad = window // 2
    x = F.pad(exp.unsqueeze(0).unsqueeze(0), (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, window) / window
    smoothed = F.conv1d(x, kernel).squeeze(0).squeeze(0)
    # If window is even, output length may differ by 1; crop to L
    return smoothed[:L]


class SSCache:
    """Load and serve SS graphs + struct types + exposure for all cached RNAs."""

    def __init__(self, truncate_to: int = 511, exposure_smooth: int = 0, verbose: bool = True):
        self.truncate_to = truncate_to
        self.exposure_smooth = exposure_smooth
        self.edge_index: Dict[str, torch.Tensor] = {}
        self.struct_types: Dict[str, torch.Tensor] = {}
        self.exposure: Dict[str, torch.Tensor] = {}
        self.L: Dict[str, int] = {}
        self._load(verbose)

    def _load(self, verbose: bool) -> None:
        files = list(CACHE_DIR.glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"No fold cache at {CACHE_DIR}. Run: python deeprsma_ext/structure/fold_all.py"
            )
        if verbose:
            print(f"[SSCache] loading {len(files)} folds from {CACHE_DIR}  "
                  f"(smooth_window={self.exposure_smooth})")
        for p in files:
            rid = p.stem
            rec = load_fold(rid)
            self.edge_index[rid] = build_ss_edge_index(rec, self.truncate_to)
            self.struct_types[rid] = get_struct_types(rec, self.truncate_to)
            exp = get_exposure(rec, self.truncate_to)
            if self.exposure_smooth > 1:
                exp = smooth_exposure(exp, self.exposure_smooth)
            self.exposure[rid] = exp
            self.L[rid] = min(rec["len"], self.truncate_to)
        if verbose:
            avg_len = sum(self.L.values()) / max(1, len(self.L))
            print(f"[SSCache] loaded. n_rnas={len(self.edge_index)} avg_len={avg_len:.1f}")

    def has(self, rna_id: str) -> bool:
        return rna_id in self.edge_index

    def get(self, rna_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.edge_index[rna_id], self.struct_types[rna_id], self.exposure[rna_id]
