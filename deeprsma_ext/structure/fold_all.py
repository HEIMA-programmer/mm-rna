"""
Phase 2a: precompute secondary structure for all unique RNAs in DeepRSMA's R-SIM data.

Supports two folding engines:
  * rnafold (default)   — ViennaRNA RNAfold.exe via subprocess. Fast (C impl),
                          paper-compatible. Requires ViennaRNA installed.
  * seqfold             — pure-Python fallback, slow for long RNAs, Windows-friendly.

Outputs one JSON per RNA with:
  rna_id, seq, len, dot_bracket, dg, struct_types (int 0..4), exposure (float in (0,1)),
  pair_indices (list of [i, j] 0-indexed), engine (str)

Usage:
  cd C:/Users/yanbin/Desktop/MM
  python deeprsma_ext/structure/fold_all.py [--engine rnafold|seqfold]
                                            [--rnafold-path PATH]
                                            [--rna-types All_sf,Aptamers,...]
                                            [--max-len 512] [--workers N] [--overwrite]

Element type mapping (forgi labels):
  s -> 0 (stem)              exposure 0.20
  h -> 1 (hairpin loop)      exposure 0.80
  i -> 2 (internal/bulge)    exposure 0.85
  m -> 3 (multiloop)         exposure 0.90
  f, t -> 4 (dangling end)   exposure 0.70
"""
# seqfold uses importlib.metadata which is Python 3.8+ only; backport for 3.7
import sys
try:
    import importlib.metadata as _md  # noqa: F401
except ImportError:
    import importlib_metadata
    sys.modules['importlib.metadata'] = importlib_metadata

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEEPRSMA = ROOT / "DeepRSMA"
DATA = DEEPRSMA / "data"
CACHE = ROOT / "cache" / "rnafold"
CACHE.mkdir(parents=True, exist_ok=True)

import shutil as _shutil
# Cross-platform: on Windows try the default install path, on Linux rely on PATH.
_WIN_PATH = r"C:\Program Files (x86)\ViennaRNA Package\RNAfold.exe"
if os.name == "nt" and os.path.exists(_WIN_PATH):
    DEFAULT_RNAFOLD = _WIN_PATH
else:
    _found = _shutil.which("RNAfold")
    DEFAULT_RNAFOLD = _found if _found else "RNAfold"

ELEMENT_TO_TYPE = {"s": 0, "h": 1, "i": 2, "m": 3, "f": 4, "t": 4}
TYPE_TO_EXPOSURE = {0: 0.20, 1: 0.80, 2: 0.85, 3: 0.90, 4: 0.70}

# DeepRSMA's char_to_one_hot maps X -> 4, Y -> 5 (non-canonical nucleotides).
# seqfold only folds A/C/G/U; ViennaRNA accepts N but substitutes anyway. We canonicalize:
NON_CANONICAL_SUBST = {
    "Y": "C", "X": "A", "N": "A",
    "R": "A", "W": "A", "S": "C", "K": "G", "M": "A",
    "B": "C", "D": "A", "H": "A", "V": "A",
}


def canonicalize(seq: str) -> str:
    return "".join(NON_CANONICAL_SUBST.get(c, c) for c in seq)


def collect_unique_rnas(rna_types: List[str], max_len: int) -> Dict[str, str]:
    rna_id_to_seq: Dict[str, str] = {}
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
            if rid not in rna_id_to_seq:
                rna_id_to_seq[rid] = seq
            elif rna_id_to_seq[rid] != seq:
                print(f"[warn] RNA ID {rid} has differing sequences across rows; keeping first")
    return rna_id_to_seq


# -------------------- forgi parsing (shared by both engines) --------------------

def parse_dotbracket(db: str) -> Tuple[List[int], List[float], List[Tuple[int, int]]]:
    """Return per-position struct_types (0-indexed), exposure, and pair_indices (0-indexed)."""
    import forgi.graph.bulge_graph as fgb
    bg = fgb.BulgeGraph.from_dotbracket(db)
    L = len(db)
    struct_types: List[int] = []
    exposure: List[float] = []
    for i in range(1, L + 1):
        elem = bg.get_node_from_residue_num(i)
        t = ELEMENT_TO_TYPE.get(elem[0], 4)
        struct_types.append(t)
        exposure.append(TYPE_TO_EXPOSURE[t])

    pair_indices: List[Tuple[int, int]] = []
    stack: List[int] = []
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            pair_indices.append((j, i))
    return struct_types, exposure, pair_indices


# -------------------- ViennaRNA subprocess batch --------------------

# Regex to parse RNAfold output line: "<dotbracket> (<energy>)"
_RNAFOLD_RE = re.compile(r"^([\.\(\)\[\]\{\}\<\>]+)\s+\(\s*(-?\d+\.\d+)\s*\)\s*$")


def fold_batch_rnafold(rnafold_path: str, batch: List[Tuple[str, str]]) -> List[Dict]:
    """Run RNAfold.exe on a batch of (rna_id, seq) tuples via FASTA stdin.

    Returns a list of result dicts with same length/order as input.
    Each dict is either successful (has 'dot_bracket', 'dg', etc.) or has 'error'.
    """
    # Build FASTA input
    fasta_lines = []
    for rid, seq in batch:
        fasta_lines.append(f">{rid}")
        fasta_lines.append(canonicalize(seq))
    fasta_input = "\n".join(fasta_lines) + "\n"

    try:
        proc = subprocess.run(
            [rnafold_path, "--noPS"],
            input=fasta_input,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as e:
        return [{"rna_id": rid, "error": f"subprocess failed: {e}"} for rid, _ in batch]

    if proc.returncode != 0:
        return [{"rna_id": rid, "error": f"RNAfold returncode={proc.returncode}: {proc.stderr[:300]}"}
                for rid, _ in batch]

    # Parse output: per sequence, 3 lines (>id, seq, dotbracket energy)
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    results: List[Dict] = []
    seq_map = dict(batch)
    i = 0
    while i < len(out_lines):
        line = out_lines[i]
        if line.startswith(">"):
            rid = line[1:].strip()
            if i + 2 >= len(out_lines):
                results.append({"rna_id": rid, "error": "RNAfold output truncated"})
                break
            seq_line = out_lines[i + 1].strip()
            db_line = out_lines[i + 2].strip()
            m = _RNAFOLD_RE.match(db_line)
            if m is None:
                results.append({"rna_id": rid, "error": f"parse failed: {db_line!r}"})
                i += 3
                continue
            db, energy = m.group(1), float(m.group(2))
            orig_seq = seq_map.get(rid, seq_line)
            try:
                struct_types, exposure, pair_indices = parse_dotbracket(db)
            except Exception as e:
                results.append({"rna_id": rid, "error": f"forgi parse failed: {e}"})
                i += 3
                continue
            results.append({
                "rna_id": rid,
                "seq": orig_seq,
                "len": len(orig_seq),
                "dot_bracket": db,
                "dg": energy,
                "struct_types": struct_types,
                "exposure": exposure,
                "pair_indices": [list(p) for p in pair_indices],
                "engine": "rnafold",
            })
            i += 3
        else:
            i += 1

    # Ensure results in same order as input
    results_by_id = {r["rna_id"]: r for r in results}
    ordered = []
    for rid, _ in batch:
        if rid in results_by_id:
            ordered.append(results_by_id[rid])
        else:
            ordered.append({"rna_id": rid, "error": "missing in RNAfold output"})
    return ordered


# -------------------- seqfold worker (legacy/fallback) --------------------

def fold_and_parse_seqfold(args):
    """Worker function for seqfold engine."""
    rid, seq = args
    try:
        import importlib.metadata as _md   # noqa
    except ImportError:
        import importlib_metadata
        sys.modules['importlib.metadata'] = importlib_metadata
    from seqfold import fold, dot_bracket

    try:
        t0 = time.time()
        seq_canon = canonicalize(seq)
        structs = fold(seq_canon)
        db = dot_bracket(seq_canon, structs)
        dg = float(sum(s.e for s in structs)) if structs else 0.0
        struct_types, exposure, pair_indices = parse_dotbracket(db)
        return {
            "rna_id": rid,
            "seq": seq,
            "len": len(seq),
            "dot_bracket": db,
            "dg": dg,
            "struct_types": struct_types,
            "exposure": exposure,
            "pair_indices": [list(p) for p in pair_indices],
            "engine": "seqfold",
            "_runtime": time.time() - t0,
        }
    except Exception as e:
        return {"rna_id": rid, "error": f"{type(e).__name__}: {e}",
                "_traceback": traceback.format_exc()}


# -------------------- main --------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["rnafold", "seqfold"], default="rnafold")
    parser.add_argument("--rnafold-path", type=str, default=DEFAULT_RNAFOLD)
    parser.add_argument("--rna-types", type=str, default="All_sf")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="(seqfold only) number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="(rnafold only) FASTA batch size per subprocess invocation")
    args = parser.parse_args()

    rna_types = [s.strip() for s in args.rna_types.split(",")]
    rnas = collect_unique_rnas(rna_types, args.max_len)
    print(f"Unique RNAs to fold: {len(rnas)}")

    pending = []
    n_skip = 0
    for rid, seq in rnas.items():
        if (CACHE / f"{rid}.json").exists() and not args.overwrite:
            n_skip += 1
            continue
        if not seq:
            continue
        pending.append((rid, seq))
    print(f"Already cached: {n_skip}.  To fold: {len(pending)}.  Engine: {args.engine}")

    if not pending:
        print("Nothing to do.")
        return 0

    pending.sort(key=lambda t: -len(t[1]))   # longest first

    n_done = 0
    n_err = 0
    t_start = time.time()

    if args.engine == "rnafold":
        if not Path(args.rnafold_path).exists():
            print(f"ERROR: RNAfold not found at {args.rnafold_path}")
            print("Pass --rnafold-path to override, or use --engine seqfold")
            return 1
        print(f"Using RNAfold at: {args.rnafold_path}")
        # Process in batches via subprocess
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start:batch_start + args.batch_size]
            results = fold_batch_rnafold(args.rnafold_path, batch)
            for res in results:
                rid = res["rna_id"]
                if "error" in res:
                    n_err += 1
                    print(f"  ERR {rid}: {res['error']}", flush=True)
                    continue
                with open(CACHE / f"{rid}.json", "w") as f:
                    json.dump(res, f)
                n_done += 1
            elapsed = time.time() - t_start
            rate = (n_done + n_err) / elapsed if elapsed > 0 else 0
            remaining = len(pending) - (n_done + n_err)
            eta = remaining / rate if rate > 0 else 0
            print(f"  batch [{batch_start+len(batch)}/{len(pending)}] "
                  f"done={n_done} err={n_err} | {rate:.1f}/s ETA={eta:.0f}s",
                  flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(fold_and_parse_seqfold, pending, chunksize=1), start=1
            ):
                rid = result.get("rna_id", "?")
                if "error" in result:
                    n_err += 1
                    print(f"  [{i}/{len(pending)}] {rid} ERR: {result['error']}", flush=True)
                    continue
                result.pop("_runtime", None)
                with open(CACHE / f"{rid}.json", "w") as f:
                    json.dump(result, f)
                n_done += 1
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(pending) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(pending)}] {rid} len={result['len']} "
                      f"| done={n_done} err={n_err} | {rate:.1f}/s ETA={eta:.0f}s",
                      flush=True)

    elapsed = time.time() - t_start
    total_cached = n_done + n_skip
    print(f"\nDone in {elapsed:.1f}s. folded={n_done} skipped_preexisting={n_skip} errors={n_err}")
    print(f"Cache size: {total_cached}/{len(rnas)} RNAs have records in {CACHE}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
