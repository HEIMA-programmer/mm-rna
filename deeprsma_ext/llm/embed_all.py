"""
Precompute per-nucleotide RNA LLM embeddings for all R-SIM RNAs.

Supported LLMs (via the multimolecule HF hub package):
  rnabert   — multimolecule/rnabert      120-d
  rnafm     — multimolecule/rnafm        640-d  (same as DeepRSMA's bundled)
  ernierna  — multimolecule/ernierna     768-d
  rinalmo   — multimolecule/rinalmo      1280-d (large, fp16 recommended)

All embeddings are extracted with the LLM in eval() mode under torch.no_grad().
Output is saved to  cache/llm/{model}/{Target_RNA_ID}.npy

Usage:
  cd C:/Users/yanbin/Desktop/MM
  python deeprsma_ext/llm/embed_all.py --llm rnabert [--rna-types All_sf] [--fp16]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch

ROOT = Path(__file__).resolve().parents[2]
DEEPRSMA = ROOT / "DeepRSMA"
DATA = DEEPRSMA / "data"
CACHE_BASE = ROOT / "cache" / "llm"


LLM_SPECS = {
    "rnabert":  dict(hf_id="multimolecule/rnabert",  dim=120,  max_len=440,  bos_tok=True),
    "rnafm":    dict(hf_id="multimolecule/rnafm",    dim=640,  max_len=1024, bos_tok=True),
    "ernierna": dict(hf_id="multimolecule/ernierna", dim=768,  max_len=1024, bos_tok=True),
    "rinalmo":  dict(hf_id="multimolecule/rinalmo",  dim=1280, max_len=1024, bos_tok=True),
}


def collect_unique_rnas(rna_types, max_len):
    rnas = {}
    for rt in rna_types:
        csv = DATA / "RSM_data" / f"{rt}_dataset_v1.csv"
        if not csv.exists():
            print(f"[warn] missing CSV: {csv}")
            continue
        df = pd.read_csv(csv, delimiter="\t")
        for _, row in df.iterrows():
            rid = row["Target_RNA_ID"]
            seq = str(row["Target_RNA_sequence"]).strip().upper().replace("T", "U")
            if len(seq) > max_len:
                seq = seq[: max_len - 1]
            rnas.setdefault(rid, seq)
    return rnas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True, choices=list(LLM_SPECS.keys()))
    p.add_argument("--rna-types", type=str, default="All_sf")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--fp16", action="store_true", help="Use float16 inference (recommended for RiNALMo)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    spec = LLM_SPECS[args.llm]
    cache_dir = CACHE_BASE / args.llm
    cache_dir.mkdir(parents=True, exist_ok=True)

    rna_types = [s.strip() for s in args.rna_types.split(",")]
    rnas = collect_unique_rnas(rna_types, args.max_len)
    print(f"LLM={args.llm} | dim={spec['dim']} | max_len={spec['max_len']} | RNAs={len(rnas)}")

    # Filter pending
    pending = []
    n_skip = 0
    for rid, seq in rnas.items():
        if (cache_dir / f"{rid}.npy").exists() and not args.overwrite:
            n_skip += 1
            continue
        pending.append((rid, seq))
    print(f"Already cached: {n_skip}. To encode: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return 0

    # Load model
    try:
        from multimolecule import RnaTokenizer
        from transformers import AutoModel
    except ImportError:
        print("ERROR: pip install multimolecule")
        return 1

    tok = RnaTokenizer.from_pretrained(spec["hf_id"])
    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModel.from_pretrained(spec["hf_id"]).to(args.device).to(dtype).eval()
    print(f"Loaded {spec['hf_id']} onto {args.device} (dtype={dtype})")

    t_start = time.time()
    n_done = n_err = 0
    # Sort longest first (balances rolling ETA)
    pending.sort(key=lambda t: -len(t[1]))

    with torch.no_grad():
        for i, (rid, seq) in enumerate(pending):
            try:
                # Truncate to LLM-native max if needed (some are 440, 1024)
                seq_trunc = seq[: spec["max_len"] - 2]   # reserve 2 for BOS/EOS if any
                inputs = tok(seq_trunc, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                out = model(**inputs)
                h = out.last_hidden_state[0]   # [L_tokens, dim]
                # Strip special tokens: multimolecule tokenizers add [CLS] and [EOS]
                # Empirically the per-nucleotide part is [1:-1].
                h = h[1:-1]
                # Downcast to float32 for storage (smaller variance issue than fp16)
                arr = h.float().cpu().numpy()
                np.save(cache_dir / f"{rid}.npy", arr)
                n_done += 1
            except Exception as e:
                n_err += 1
                print(f"  [{i+1}/{len(pending)}] {rid} ERR: {e}", flush=True)
                continue

            if (i + 1) % 20 == 0 or (i + 1) == len(pending):
                dt = time.time() - t_start
                rate = (i + 1) / dt if dt > 0 else 0
                eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(pending)}] {rid} len={len(seq)} "
                      f"| done={n_done} err={n_err} | {rate:.1f}/s ETA={eta:.0f}s",
                      flush=True)

    dt = time.time() - t_start
    print(f"\nDone in {dt:.1f}s. encoded={n_done} skipped={n_skip} errors={n_err}")
    print(f"Cache dir: {cache_dir}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
