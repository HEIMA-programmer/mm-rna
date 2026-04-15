"""
Unified training driver for the full Phase 4 ablation matrix on N-fold stratified CV.

Mirrors DeepRSMA/main_cv.py — identical hyperparameters, seeds, stratification.
Configurable via flags:

  Core 2x2
    --use-adapter            enable StructureAwareAdapter (Improvement 1)
    --use-bias               enable exposure-bias cross-attention (Improvement 2)

  CV protocol
    --n-splits N             default 10; paper uses 10 but 5 supported
    --rna-type NAME          default All_sf
    --folds N                run only first N folds (default = n-splits)
    --epochs N               default 300
    --batch-size N           default 16
    --lr, --weight-decay, --hidden-dim    override hyperparams

  Seeds
    --seed N                 training seed (default 1)
    --seeds s1,s2,s3         multi-seed shortcut: runs the same config for each seed

  Adapter ablations
    --adapter-layers N       default 2
    --no-gcn                 drop GCN (adapter = proj + concat only)
    --no-struct-emb          drop struct-type embedding

  Bias ablations
    --lambda-fixed           freeze λ at 1.0 (not learned)
    --exposure-smooth N      sliding-window smoothing (default 0 = off; 3 or 5 recommended)
    --bias-direction X       both | mole_query | rna_query (default both)

  LLM swap
    --llm NAME               rnafm (default, uses data.emb) | rnabert | ernierna | rinalmo
                             non-rnafm requires precomputed cache via llm/embed_all.py
"""
import argparse
import csv
import os
import random
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error

from data import RNA_dataset, Molecule_dataset, WordVocab   # noqa: F401

from deeprsma_ext.data.ss_cache import SSCache
from deeprsma_ext.data.llm_cache import LLMCache
from deeprsma_ext.models.deeprsma_ext import DeepRSMA_ext


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CustomDualDataset(Dataset):
    def __init__(self, d1, d2):
        self.d1, self.d2 = d1, d2
        assert len(self.d1) == len(self.d2)
    def __getitem__(self, i):
        return self.d1[i], self.d2[i]
    def __len__(self):
        return len(self.d1)


class RegressorStratifiedCV:
    def __init__(self, n_splits=10, n_repeats=1, group_count=5, random_state=2, strategy='uniform'):
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.disc = KBinsDiscretizer(n_bins=group_count, encode='ordinal', strategy=strategy)
    def split(self, X, y):
        kgroups = self.disc.fit_transform(y[:, None])[:, 0]
        return self.cv.split(X, kgroups)


def make_label(args) -> str:
    """Build a descriptive run label from the flag combination."""
    parts = []
    if not args.use_adapter and not args.use_bias:
        parts.append("baseline")
    else:
        if args.use_adapter:
            parts.append("adapter")
            if not args.adapter_use_gcn:
                parts.append("nogcn")
            if not args.adapter_use_struct_emb:
                parts.append("nostruct")
            if args.adapter_layers != 2:
                parts.append(f"L{args.adapter_layers}")
        if args.use_bias:
            parts.append("bias")
            if args.lambda_fixed:
                parts.append("lamFix")
            if args.exposure_smooth > 1:
                parts.append(f"sm{args.exposure_smooth}")
            if args.bias_direction != "both":
                parts.append(args.bias_direction)
    if args.llm != "rnafm":
        parts.append(args.llm)
    if args.use_adapter and args.use_bias and len(parts) == 2:
        parts = ["full"]
    if args.n_splits != 10:
        parts.append(f"k{args.n_splits}")
    return "_".join(parts) if parts else "baseline"


def run_one_seed(args, seed, label, ss_cache, llm_cache, rna_ds, mole_ds, device):
    set_seed(seed)

    kf = RegressorStratifiedCV(
        n_splits=args.n_splits, n_repeats=1, group_count=5,
        random_state=args.seed_dataset, strategy='uniform',
    )
    splits = list(kf.split(np.arange(len(rna_ds)), np.array(rna_ds.y, dtype=float)))
    folds_to_run = args.folds or args.n_splits
    n_run = min(folds_to_run, len(splits))

    log_dir = ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    save_dir = ROOT / 'save_ext'
    save_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{label}_{args.rna_type}_seed{args.seed_dataset}-{seed}.csv"
    progress_path = log_dir / f"{label}_{args.rna_type}_seed{args.seed_dataset}-{seed}_progress.log"
    write_header = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['fold', 'pcc', 'scc', 'rmse'])

    p_list, s_list, r_list = [], [], []
    print(f"\n>>> Running seed={seed} | label={label} | {n_run}/{args.n_splits} folds × {args.epochs} epochs")
    seed_start = time.time()

    for fold_i, (train_id, test_id) in enumerate(splits[:n_run], start=1):
        print(f"\n=== seed={seed} Fold {fold_i}/{n_run} ===  train={len(train_id)} test={len(test_id)}")
        train_id = train_id.astype(np.int64)
        test_id = test_id.astype(np.int64)
        train_ds = CustomDualDataset(rna_ds[train_id], mole_ds[train_id])
        test_ds = CustomDualDataset(rna_ds[test_id], mole_ds[test_id])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0,
                                  drop_last=False, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0,
                                 drop_last=False, shuffle=False)

        model = DeepRSMA_ext(
            hidden_dim=args.hidden_dim,
            ss_cache=ss_cache,
            use_adapter=args.use_adapter,
            use_bias=args.use_bias,
            llm_dim=args.llm_dim,
            adapter_layers=args.adapter_layers,
            adapter_use_gcn=args.adapter_use_gcn,
            adapter_use_struct_emb=args.adapter_use_struct_emb,
            bias_direction=args.bias_direction,
            lambda_trainable=not args.lambda_fixed,
            llm_cache=llm_cache,
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()

        max_p, max_s, max_rmse = -1.0, -1.0, 0.0
        for epo in range(args.epochs):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            for br, bm in train_loader:
                opt.zero_grad()
                pred = model(br.to(device), bm.to(device))
                loss = loss_fn(pred.squeeze(1), br.y)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            model.eval()
            y_label, y_pred = [], []
            with torch.no_grad():
                for br, bm in test_loader:
                    label_v = Variable(torch.from_numpy(np.array(br.y))).float()
                    score = model(br.to(device), bm.to(device))
                    logits = torch.squeeze(score).detach().cpu().numpy()
                    y_label += label_v.cpu().numpy().flatten().tolist()
                    y_pred += logits.flatten().tolist()
            pcc = pearsonr(y_label, y_pred)[0]
            scc = spearmanr(y_label, y_pred)[0]
            rmse = float(np.sqrt(mean_squared_error(y_label, y_pred)))

            if pcc > max_p:
                max_p, max_s, max_rmse = pcc, scc, rmse
                torch.save(
                    model.state_dict(),
                    save_dir / f"{label}_{args.rna_type}_{args.seed_dataset}_{fold_i}_{seed}.pth",
                )

            line = (f"  epo {epo:4d}: train_loss={train_loss/len(train_loader):.4f} "
                    f"pcc={pcc:.4f} scc={scc:.4f} rmse={rmse:.4f}  "
                    f"best_pcc={max_p:.4f}  dt={time.time()-t0:.1f}s")
            print(line, flush=True)
            with open(progress_path, 'a') as pf:
                pf.write(f"fold{fold_i} " + line.strip() + "\n")

        p_list.append(max_p); s_list.append(max_s); r_list.append(max_rmse)
        print(f"  Fold {fold_i} best: pcc={max_p:.4f} scc={max_s:.4f} rmse={max_rmse:.4f}")
        print(f"  Running mean: pcc={np.mean(p_list):.4f}  scc={np.mean(s_list):.4f}  rmse={np.mean(r_list):.4f}")
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([fold_i, max_p, max_s, max_rmse])

    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mean', float(np.mean(p_list)), float(np.mean(s_list)), float(np.mean(r_list))])
        w.writerow(['std', float(np.std(p_list)), float(np.std(s_list)), float(np.std(r_list))])
    print(f"\n=== Done seed={seed} in {(time.time()-seed_start)/3600:.2f} h ===")
    print(f"Mean PCC = {np.mean(p_list):.4f} (std {np.std(p_list):.4f})")
    print(f"Mean SCC = {np.mean(s_list):.4f} (std {np.std(s_list):.4f})")
    print(f"Mean RMSE= {np.mean(r_list):.4f} (std {np.std(r_list):.4f})")
    return dict(pcc=np.mean(p_list), scc=np.mean(s_list), rmse=np.mean(r_list),
                pcc_std=np.std(p_list), log=str(log_path))


def main():
    p = argparse.ArgumentParser()
    # Core
    p.add_argument('--use-adapter', action='store_true')
    p.add_argument('--use-bias', action='store_true')
    # CV
    p.add_argument('--n-splits', type=int, default=10)
    p.add_argument('--folds', type=int, default=0, help="run first N folds; 0 = all n_splits")
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--rna-type', type=str, default='All_sf')
    p.add_argument('--batch-size', type=int, default=16)
    # Seeds
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--seeds', type=str, default='', help="comma-separated seeds (overrides --seed)")
    p.add_argument('--seed-dataset', type=int, default=2)
    # Hyperparams
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-7)
    p.add_argument('--hidden-dim', type=int, default=128)
    # Adapter ablations
    p.add_argument('--adapter-layers', type=int, default=2)
    p.add_argument('--no-gcn', dest='adapter_use_gcn', action='store_false')
    p.add_argument('--no-struct-emb', dest='adapter_use_struct_emb', action='store_false')
    # Bias ablations
    p.add_argument('--lambda-fixed', action='store_true')
    p.add_argument('--exposure-smooth', type=int, default=0)
    p.add_argument('--bias-direction', choices=['both', 'mole_query', 'rna_query'], default='both')
    # LLM swap
    p.add_argument('--llm', default='rnafm', choices=['rnafm', 'rnabert', 'ernierna', 'rinalmo'])
    p.add_argument('--llm-dim', type=int, default=0, help="override llm_dim; 0 = auto from cache")
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    label = make_label(args)
    print(f"Config: label={label}")
    print(f"  use_adapter={args.use_adapter}  use_bias={args.use_bias}  llm={args.llm}")
    print(f"  n_splits={args.n_splits}  epochs={args.epochs}  batch={args.batch_size}")
    print(f"  adapter: gcn={args.adapter_use_gcn} struct_emb={args.adapter_use_struct_emb} layers={args.adapter_layers}")
    print(f"  bias: direction={args.bias_direction} lambda_fixed={args.lambda_fixed} smooth={args.exposure_smooth}")

    # Caches — loaded once, reused across seeds
    ss_cache = None
    if args.use_adapter or args.use_bias:
        ss_cache = SSCache(
            truncate_to=511,
            exposure_smooth=args.exposure_smooth,
            verbose=True,
        )
    llm_cache = None
    if args.llm != "rnafm":
        llm_cache = LLMCache(args.llm, verbose=True)
        if args.llm_dim == 0:
            args.llm_dim = llm_cache.get_dim()
    else:
        if args.llm_dim == 0:
            args.llm_dim = 640

    # Datasets — loaded once
    print(f"\nLoading datasets for RNA_type={args.rna_type} ...")
    rna_ds = RNA_dataset(args.rna_type)
    mole_ds = Molecule_dataset(args.rna_type)
    print(f"  RNA={len(rna_ds)} Mole={len(mole_ds)}")

    # Determine seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    else:
        seeds = [args.seed]
    print(f"  Seeds to run: {seeds}")

    results = []
    for s in seeds:
        r = run_one_seed(args, s, label, ss_cache, llm_cache, rna_ds, mole_ds, device)
        results.append((s, r))

    if len(results) > 1:
        print("\n=== Multi-seed summary ===")
        pccs = [r['pcc'] for _, r in results]
        sccs = [r['scc'] for _, r in results]
        rmses = [r['rmse'] for _, r in results]
        print(f"PCC : {np.mean(pccs):.4f} ± {np.std(pccs):.4f}  (per-seed: {[f'{x:.4f}' for x in pccs]})")
        print(f"SCC : {np.mean(sccs):.4f} ± {np.std(sccs):.4f}")
        print(f"RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")


if __name__ == '__main__':
    main()
