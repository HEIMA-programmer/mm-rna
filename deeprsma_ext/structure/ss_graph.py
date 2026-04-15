"""
Build secondary-structure graph for an RNA from its cached RNAfold record.

Edges are of two kinds:
  - backbone: (i, i+1) and (i+1, i) for each consecutive nucleotide pair
  - base-pair: (a, b) and (b, a) for each base pair in the dot-bracket

Output edge_index is a long tensor of shape [2, E], matching PyTorch Geometric's convention.
"""
import json
from pathlib import Path
from typing import Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "rnafold"


def load_fold(rna_id: str):
    """Load the cached RNAfold record for a given RNA ID."""
    path = CACHE / f"{rna_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cached fold not found for {rna_id}: {path}")
    with open(path, "r") as f:
        return json.load(f)


def build_ss_edge_index(record, truncate_to: int = None) -> torch.Tensor:
    """Build the SS graph edge_index (2, E) from a cached fold record.

    If truncate_to is given, drop any edge with an endpoint >= truncate_to.
    This matches DeepRSMA's sequence truncation (first 511 if len > 512).
    """
    L = record["len"]
    if truncate_to is not None:
        L = min(L, truncate_to)

    edges = []
    # Backbone edges (both directions)
    for i in range(L - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    # Base-pair edges (both directions). Skip pairs whose endpoint falls outside truncation.
    for a, b in record.get("pair_indices", []):
        if a >= L or b >= L:
            continue
        edges.append((a, b))
        edges.append((b, a))

    if not edges:
        # Fallback for L<=1: empty edge index with correct shape
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()   # [2, E]
    return edge_index


def get_struct_types(record, truncate_to: int = None) -> torch.Tensor:
    """Return per-position struct type ids as a LongTensor of shape [L]."""
    types = record["struct_types"]
    if truncate_to is not None:
        types = types[:truncate_to]
    return torch.tensor(types, dtype=torch.long)


def get_exposure(record, truncate_to: int = None) -> torch.Tensor:
    """Return per-position exposure score as a FloatTensor of shape [L], values in (0,1)."""
    exp = record["exposure"]
    if truncate_to is not None:
        exp = exp[:truncate_to]
    return torch.tensor(exp, dtype=torch.float)


def build_all(rna_id: str, truncate_to: int = 511) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot: load the cached fold and return (edge_index, struct_types, exposure).

    Default truncate_to=511 matches DeepRSMA's process_data_rna.py:53.
    """
    rec = load_fold(rna_id)
    return (
        build_ss_edge_index(rec, truncate_to),
        get_struct_types(rec, truncate_to),
        get_exposure(rec, truncate_to),
    )


if __name__ == "__main__":
    import sys
    rid = sys.argv[1] if len(sys.argv) > 1 else None
    if rid is None:
        # Pick first available
        files = sorted(CACHE.glob("*.json"))
        if not files:
            print("No cached folds yet. Run fold_all.py first.")
            sys.exit(1)
        rid = files[0].stem
    print(f"RNA ID: {rid}")
    ei, st, ex = build_all(rid)
    print(f"  edge_index shape: {tuple(ei.shape)}")
    print(f"  struct_types: {st.tolist()[:30]}{'...' if st.size(0) > 30 else ''}")
    print(f"  exposure    : {[round(v, 2) for v in ex.tolist()[:30]]}{'...' if ex.size(0) > 30 else ''}")
