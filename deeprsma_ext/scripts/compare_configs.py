"""
Aggregate results from multiple training CSVs and produce:
  1. A tabular comparison (label × PCC/SCC/RMSE with mean ± std)
  2. Pairwise significance tests (Wilcoxon signed-rank on per-fold PCC)
  3. Markdown + LaTeX-ready table

Usage:
  python deeprsma_ext/scripts/compare_configs.py [--logs-dir logs/] [--rna-type All_sf] [--seed 1]

Auto-discovers files of form `{label}_{rna_type}_seed{sd}-{s}.csv` in logs/.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

ROOT = Path(__file__).resolve().parents[2]


def load_csv(path):
    """Return list of per-fold (pcc, scc, rmse) tuples and the 'mean' row if present."""
    per_fold = []
    mean_row = None
    std_row = None
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if row[0] == 'mean':
                mean_row = tuple(float(x) for x in row[1:4])
            elif row[0] == 'std':
                std_row = tuple(float(x) for x in row[1:4])
            else:
                try:
                    _ = int(row[0])
                except ValueError:
                    continue
                per_fold.append((float(row[1]), float(row[2]), float(row[3])))
    return per_fold, mean_row, std_row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs-dir', default=str(ROOT / 'logs'))
    p.add_argument('--rna-type', default='All_sf')
    p.add_argument('--seed-dataset', type=int, default=2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--pattern', default='',
                   help="optional glob pattern override, e.g. 'blind_rna_*'")
    p.add_argument('--no-sig', action='store_true', help="skip significance tests")
    args = p.parse_args()

    log_dir = Path(args.logs_dir)
    if args.pattern:
        files = sorted(log_dir.glob(f"{args.pattern}.csv"))
    else:
        suffix = f"_{args.rna_type}_seed{args.seed_dataset}-{args.seed}.csv"
        files = sorted(log_dir.glob(f"*{suffix}"))
    if not files:
        print(f"No CSVs found in {log_dir} matching pattern")
        return 1

    rows = []
    all_pccs = {}   # label -> list of per-fold PCCs
    for path in files:
        label = path.stem
        # Strip common suffix for nicer labels
        label_short = label.replace(f"_{args.rna_type}_seed{args.seed_dataset}-{args.seed}", "")
        per_fold, mean, std = load_csv(path)
        if not per_fold:
            continue
        pccs = [p_ for p_, _, _ in per_fold]
        sccs = [s_ for _, s_, _ in per_fold]
        rmses = [r_ for _, _, r_ in per_fold]
        row = dict(
            label=label_short,
            n_folds=len(per_fold),
            pcc_mean=np.mean(pccs), pcc_std=np.std(pccs),
            scc_mean=np.mean(sccs), scc_std=np.std(sccs),
            rmse_mean=np.mean(rmses), rmse_std=np.std(rmses),
        )
        rows.append(row)
        all_pccs[label_short] = pccs

    # Print markdown
    print("\n## Results\n")
    print(f"| Config | n | PCC (mean±std) | SCC (mean±std) | RMSE (mean±std) |")
    print(f"|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['n_folds']} | "
              f"{r['pcc_mean']:.4f} ± {r['pcc_std']:.4f} | "
              f"{r['scc_mean']:.4f} ± {r['scc_std']:.4f} | "
              f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f} |")

    # Significance tests
    if not args.no_sig and len(all_pccs) >= 2 and wilcoxon is not None:
        print("\n## Wilcoxon signed-rank (per-fold PCC)\n")
        labels = list(all_pccs.keys())
        print(f"| A vs B | Δmean PCC | W | p-value |")
        print(f"|---|---|---|---|")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_pccs = all_pccs[labels[i]]
                b_pccs = all_pccs[labels[j]]
                if len(a_pccs) != len(b_pccs):
                    continue
                try:
                    stat, p = wilcoxon(a_pccs, b_pccs)
                    delta = np.mean(a_pccs) - np.mean(b_pccs)
                    print(f"| {labels[i]} vs {labels[j]} | {delta:+.4f} | {stat:.1f} | {p:.4f} |")
                except Exception as e:
                    print(f"| {labels[i]} vs {labels[j]} | - | - | error: {e} |")

    # LaTeX table
    print("\n## LaTeX\n")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Config & PCC & SCC & RMSE \\")
    print(r"\midrule")
    for r in rows:
        print(f"{r['label']} & "
              f"${r['pcc_mean']:.3f}\\pm{r['pcc_std']:.3f}$ & "
              f"${r['scc_mean']:.3f}\\pm{r['scc_std']:.3f}$ & "
              f"${r['rmse_mean']:.3f}\\pm{r['rmse_std']:.3f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == '__main__':
    main()
