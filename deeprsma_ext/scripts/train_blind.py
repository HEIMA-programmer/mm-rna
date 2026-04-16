"""
Blind test training: 5-fold manually-defined splits in DeepRSMA/data/blind_test/.

Three modes (matching DeepRSMA/main_blind.py):
  --cold rna    : blind-RNA (test RNAs never seen in training)
  --cold mole   : blind-Molecule (test small molecules never seen in training)
  --cold rm     : double-blind (both RNA and Molecule unseen)

Supports all the same model flags as train_cv.py.

Usage:
  python deeprsma_ext/scripts/train_blind.py --cold rna --use-adapter --use-bias --epochs 100
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
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from data import RNA_dataset, Molecule_dataset, WordVocab   # noqa

from deeprsma_ext.data.ss_cache import SSCache
from deeprsma_ext.data.llm_cache import LLMCache
from deeprsma_ext.models.deeprsma_ext import DeepRSMA_ext


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CustomDualDataset(Dataset):
    def __init__(self, d1, d2):
        self.d1, self.d2 = d1, d2
    def __getitem__(self, i): return self.d1[i], self.d2[i]
    def __len__(self): return len(self.d1)


def load_blind_folds(cold_type):
    """Return list of 5 DataFrames from data/blind_test/cold_{type}{1..5}.csv"""
    dfs = []
    for i in range(1, 6):
        path = Path(f'data/blind_test/cold_{cold_type}{i}.csv')
        dfs.append(pd.read_csv(path, delimiter=','))
    return dfs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cold', choices=['rna', 'mole', 'rm'], required=True,
                   help="blind test type: rna / mole / rm (double)")
    p.add_argument('--use-adapter', action='store_true')
    p.add_argument('--use-bias', action='store_true')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--rna-type', default='All_sf')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--seed-dataset', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-7)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--adapter-layers', type=int, default=2)
    p.add_argument('--no-gcn', dest='adapter_use_gcn', action='store_false')
    p.add_argument('--no-struct-emb', dest='adapter_use_struct_emb', action='store_false')
    p.add_argument('--lambda-fixed', action='store_true')
    p.add_argument('--lambda-init', type=float, default=0.1)
    p.add_argument('--exposure-smooth', type=int, default=0)
    p.add_argument('--bias-direction', choices=['both', 'mole_query', 'rna_query'], default='both')
    p.add_argument('--llm', default='rnafm', choices=['rnafm', 'rnabert', 'ernierna', 'rinalmo'])
    p.add_argument('--llm-dim', type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Build run label
    pieces = []
    if args.use_adapter: pieces.append('adapter')
    if args.use_bias: pieces.append('bias')
    if not pieces: pieces.append('baseline')
    if args.use_adapter and args.use_bias: pieces = ['full']
    if args.llm != 'rnafm': pieces.append(args.llm)
    label = f"blind_{args.cold}_" + "_".join(pieces)

    # Caches
    ss_cache = None
    if args.use_adapter or args.use_bias:
        ss_cache = SSCache(truncate_to=511, exposure_smooth=args.exposure_smooth, verbose=True)
    llm_cache = None
    if args.llm != 'rnafm':
        llm_cache = LLMCache(args.llm, verbose=True)
        if args.llm_dim == 0:
            args.llm_dim = llm_cache.get_dim()
    elif args.llm_dim == 0:
        args.llm_dim = 640

    # Datasets
    print(f"Loading datasets for RNA_type={args.rna_type} ...")
    rna_ds = RNA_dataset(args.rna_type)
    mole_ds = Molecule_dataset(args.rna_type)

    # All_sf CSV for mapping Entry_ID → dataset index
    all_df = pd.read_csv(f'data/RSM_data/{args.rna_type}_dataset_v1.csv', delimiter='\t')

    folds = load_blind_folds(args.cold)
    print(f"Blind test cold_{args.cold}: 5 folds, total test={sum(len(f) for f in folds)} entries")

    log_dir = ROOT / 'logs'; log_dir.mkdir(exist_ok=True)
    save_dir = ROOT / 'save_ext'; save_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{label}_seed{args.seed_dataset}-{args.seed}.csv"
    progress_path = log_dir / f"{label}_seed{args.seed_dataset}-{args.seed}_progress.log"
    write_header = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['fold', 'pcc', 'scc', 'rmse'])

    p_list, s_list, r_list = [], [], []
    run_start = time.time()
    for fold_i, df_test in enumerate(folds, start=1):
        print(f"\n=== Blind fold {fold_i}/5 ({args.cold}) ===")
        test_entry_ids = df_test['Entry_ID'].tolist()
        test_idx = all_df[all_df['Entry_ID'].isin(test_entry_ids)].index.tolist()

        train_entry_ids = []
        for j, df_j in enumerate(folds):
            if j != fold_i - 1:
                train_entry_ids.extend(df_j['Entry_ID'].tolist())
        train_idx = all_df[all_df['Entry_ID'].isin(train_entry_ids)].index.tolist()
        print(f"  train={len(train_idx)} test={len(test_idx)}")

        train_idx = np.array(train_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)
        train_ds = CustomDualDataset(rna_ds[train_idx], mole_ds[train_idx])
        test_ds = CustomDualDataset(rna_ds[test_idx], mole_ds[test_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=False)

        model = DeepRSMA_ext(
            hidden_dim=args.hidden_dim, ss_cache=ss_cache,
            use_adapter=args.use_adapter, use_bias=args.use_bias,
            llm_dim=args.llm_dim, adapter_layers=args.adapter_layers,
            adapter_use_gcn=args.adapter_use_gcn,
            adapter_use_struct_emb=args.adapter_use_struct_emb,
            bias_direction=args.bias_direction,
            lambda_trainable=not args.lambda_fixed, lambda_init=args.lambda_init,
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
                torch.save(model.state_dict(), save_dir / f"{label}_{fold_i}_{args.seed}.pth")

            line = (f"  epo {epo:4d}: loss={train_loss/len(train_loader):.4f} "
                    f"pcc={pcc:.4f} scc={scc:.4f} rmse={rmse:.4f}  best={max_p:.4f}  dt={time.time()-t0:.1f}s")
            print(line, flush=True)
            with open(progress_path, 'a') as pf:
                pf.write(f"fold{fold_i} " + line.strip() + "\n")

        p_list.append(max_p); s_list.append(max_s); r_list.append(max_rmse)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([fold_i, max_p, max_s, max_rmse])

    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mean', float(np.mean(p_list)), float(np.mean(s_list)), float(np.mean(r_list))])
        w.writerow(['std', float(np.std(p_list)), float(np.std(s_list)), float(np.std(r_list))])

    print(f"\n=== Done {label} in {(time.time()-run_start)/3600:.2f}h ===")
    print(f"Mean PCC  = {np.mean(p_list):.4f} ± {np.std(p_list):.4f}")
    print(f"Mean SCC  = {np.mean(s_list):.4f} ± {np.std(s_list):.4f}")
    print(f"Mean RMSE = {np.mean(r_list):.4f} ± {np.std(r_list):.4f}")


if __name__ == '__main__':
    main()
