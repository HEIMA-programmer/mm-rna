"""
In-memory cache of LLM embeddings per RNA.

For Phase 4 LLM swap (RNABERT / ERNIE-RNA / RiNALMo / etc.).
The default path uses the RNA-FM embeddings already bundled in
DeepRSMA/data/representations_cv/ — in that case, we don't need LLMCache
(the model uses `data.emb` loaded by the upstream dataset).

When --llm != rnafm, use LLMCache to override the per-sample embedding lookup
at forward time. The cache dir is cache/llm/{llm_name}/{Target_RNA_ID}.npy.
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


class LLMCache:
    """Per-RNA-ID lookup of a cached LLM embedding ndarray saved as .npy."""

    def __init__(self, llm_name: str, verbose: bool = True):
        self.llm_name = llm_name
        self.cache_dir = ROOT / "cache" / "llm" / llm_name
        self.emb: Dict[str, torch.Tensor] = {}
        self.dim: Optional[int] = None
        self._load(verbose)

    def _load(self, verbose: bool) -> None:
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"No LLM cache at {self.cache_dir}. "
                f"Run: python deeprsma_ext/llm/embed_all.py --llm {self.llm_name}"
            )
        files = list(self.cache_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in {self.cache_dir}")
        if verbose:
            print(f"[LLMCache:{self.llm_name}] loading {len(files)} embeddings from {self.cache_dir}")
        for p in files:
            rid = p.stem
            arr = np.load(p)
            t = torch.from_numpy(arr).float()
            self.emb[rid] = t
            if self.dim is None:
                self.dim = int(t.shape[-1])
        if verbose:
            print(f"[LLMCache:{self.llm_name}] loaded. n={len(self.emb)} dim={self.dim}")

    def has(self, rna_id: str) -> bool:
        return rna_id in self.emb

    def get(self, rna_id: str) -> torch.Tensor:
        return self.emb[rna_id]

    def get_dim(self) -> int:
        assert self.dim is not None, "LLMCache not loaded"
        return self.dim
