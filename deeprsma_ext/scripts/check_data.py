"""
Phase 0 data manifest checker.
Verifies that the cloned DeepRSMA data is intact:
- counts CSV rows per RNA-type
- checks every Entry_ID in All_sf has a matching .prob_single contact file
- checks every unique Target_RNA_ID has a matching .npy LLM embedding
- writes logs/data_manifest.json with file counts and SHA-1 of CSVs

Run from the project root: C:\\Users\\yanbin\\Desktop\\MM
"""
import os
import sys
import json
import hashlib
from typing import Optional
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # C:\Users\yanbin\Desktop\MM
DEEPRSMA = ROOT / "DeepRSMA"
DATA = DEEPRSMA / "data"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

RNA_TYPES = ["All_sf", "Aptamers", "miRNA", "Repeats", "Ribosomal", "Riboswitch", "Viral_RNA"]
CONTACT_DIRS = [
    DATA / "RNA_contact" / "Aptamers_contact",
    DATA / "RNA_contact" / "miRNA_contact",
    DATA / "RNA_contact" / "Repeats_contact",
    DATA / "RNA_contact" / "Ribosomal_contact",
    DATA / "RNA_contact" / "Riboswitch_contact",
    DATA / "RNA_contact" / "Viral_RNA_contact",
]
EMB_DIR = DATA / "representations_cv"
EMB_INDEP_DIR = DATA / "representations_independent"


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_contact(entry_id) -> Optional[Path]:
    fname = f"{entry_id}.prob_single"
    for d in CONTACT_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


def main() -> int:
    manifest = {"deeprsma_dir": str(DEEPRSMA), "csvs": {}, "issues": [], "summary": {}}

    # CSVs
    for rt in RNA_TYPES:
        csv = DATA / "RSM_data" / f"{rt}_dataset_v1.csv"
        if not csv.exists():
            manifest["issues"].append(f"missing CSV: {csv}")
            continue
        df = pd.read_csv(csv, delimiter="\t")
        manifest["csvs"][rt] = {
            "path": str(csv.relative_to(ROOT)),
            "rows": int(len(df)),
            "sha1": sha1_file(csv),
            "unique_target_rna_ids": int(df["Target_RNA_ID"].nunique()),
            "unique_entry_ids": int(df["Entry_ID"].nunique()),
        }

    # All_sf is the primary dataset for Phase 1
    all_sf = pd.read_csv(DATA / "RSM_data" / "All_sf_dataset_v1.csv", delimiter="\t")

    # Check contact maps for every Entry_ID in All_sf
    missing_contacts = []
    for eid in all_sf["Entry_ID"].unique():
        if find_contact(eid) is None:
            missing_contacts.append(int(eid))
    manifest["summary"]["all_sf_total_entry_ids"] = int(all_sf["Entry_ID"].nunique())
    manifest["summary"]["all_sf_missing_contacts"] = len(missing_contacts)
    if missing_contacts:
        manifest["issues"].append(
            f"missing .prob_single for {len(missing_contacts)} Entry_IDs (first 5: {missing_contacts[:5]})"
        )

    # Check LLM embedding for every Target_RNA_ID in All_sf
    missing_embs = []
    for tid in all_sf["Target_RNA_ID"].unique():
        if not (EMB_DIR / f"{tid}.npy").exists():
            missing_embs.append(str(tid))
    manifest["summary"]["all_sf_total_target_ids"] = int(all_sf["Target_RNA_ID"].nunique())
    manifest["summary"]["all_sf_missing_embeddings"] = len(missing_embs)
    if missing_embs:
        manifest["issues"].append(
            f"missing .npy for {len(missing_embs)} Target_RNA_IDs (first 5: {missing_embs[:5]})"
        )

    # Counts
    manifest["summary"]["contact_files_per_dir"] = {
        d.name: len(list(d.glob("*.prob_single"))) for d in CONTACT_DIRS
    }
    manifest["summary"]["representations_cv_npy_count"] = len(list(EMB_DIR.glob("*.npy")))
    manifest["summary"]["representations_independent_npy_count"] = (
        len(list(EMB_INDEP_DIR.glob("*.npy"))) if EMB_INDEP_DIR.exists() else 0
    )

    # Sample one .npy to record shape
    sample_npys = list(EMB_DIR.glob("*.npy"))[:3]
    if sample_npys:
        import numpy as np
        manifest["summary"]["sample_npys"] = []
        for p in sample_npys:
            try:
                sample = np.load(p, allow_pickle=True)
                manifest["summary"]["sample_npys"].append({
                    "name": p.name,
                    "shape": list(sample.shape) if hasattr(sample, "shape") else "scalar",
                    "dtype": str(sample.dtype) if hasattr(sample, "dtype") else type(sample).__name__,
                })
            except Exception as e:
                manifest["summary"]["sample_npys"].append({"name": p.name, "error": str(e)})

    out_path = LOGS / "data_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"All_sf rows: {manifest['csvs']['All_sf']['rows']}")
    print(f"  unique Entry_IDs: {manifest['summary']['all_sf_total_entry_ids']}")
    print(f"  unique Target_RNA_IDs: {manifest['summary']['all_sf_total_target_ids']}")
    print(f"  missing contacts: {manifest['summary']['all_sf_missing_contacts']}")
    print(f"  missing embeddings: {manifest['summary']['all_sf_missing_embeddings']}")
    for s in manifest["summary"].get("sample_npys", []):
        print(f"  sample .npy {s.get('name')}: {s.get('shape', s.get('error'))}")
    return 0 if not manifest["issues"] else 1


if __name__ == "__main__":
    sys.exit(main())
