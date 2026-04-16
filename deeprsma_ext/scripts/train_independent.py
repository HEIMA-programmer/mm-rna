"""
Independent test: train on Viral_RNA (R-SIM) → test on HIV-1 TAR (independent_data.csv).

Mirrors DeepRSMA/main_independent.py protocol (EPOCH=200, LR=6e-5, weight_decay=1e-5),
but uses the DeepRSMA_ext model so we can eval adapter / bias improvements on the
independent-test benchmark.

Note: DeepRSMA/main_independent.py uses hidden_dim=16, not 128. We keep that here
too for strict protocol match. Use --hidden-dim 128 to override.

Usage:
  python deeprsma_ext/scripts/train_independent.py --use-adapter --use-bias --epochs 200
"""
import argparse
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
from sklearn.metrics import mean_squared_error

from data import (RNA_dataset, Molecule_dataset,
                  RNA_dataset_independent, Molecule_dataset_independent,
                  WordVocab)  # noqa

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--use-adapter', action='store_true')
    p.add_argument('--use-bias', action='store_true')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=6e-5)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--hidden-dim', type=int, default=16)
    p.add_argument('--batch-train', type=int, default=8)
    p.add_argument('--batch-test', type=int, default=1)
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

    pieces = []
    if args.use_adapter: pieces.append('adapter')
    if args.use_bias: pieces.append('bias')
    if not pieces: pieces.append('baseline')
    if args.use_adapter and args.use_bias: pieces = ['full']
    if args.llm != 'rnafm': pieces.append(args.llm)
    label = "indep_" + "_".join(pieces)

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

    # Train on Viral_RNA_independent; test on HIV-1 TAR
    print("Loading training data (Viral_RNA_independent)...")
    train_rna = RNA_dataset('Viral_RNA_independent')
    train_mol = Molecule_dataset('Viral_RNA_independent')
    print(f"  train RNA={len(train_rna)} Mol={len(train_mol)}")

    print("Loading independent test (HIV-1 TAR)...")
    test_rna = RNA_dataset_independent()
    test_mol = Molecule_dataset_independent()
    print(f"  test RNA={len(test_rna)} Mol={len(test_mol)}")

    train_ds = CustomDualDataset(train_rna, train_mol)
    test_ds = CustomDualDataset(test_rna, test_mol)
    train_loader = DataLoader(train_ds, batch_size=args.batch_train, num_workers=0, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_test, num_workers=0, drop_last=False, shuffle=False)

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

    log_dir = ROOT / 'logs'; log_dir.mkdir(exist_ok=True)
    save_dir = ROOT / 'save_ext'; save_dir.mkdir(exist_ok=True)
    progress_path = log_dir / f"{label}_seed{args.seed}_progress.log"

    max_p, max_s, max_rmse = -1.0, -1.0, 0.0
    run_start = time.time()
    for epo in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for br, bm in train_loader:
            opt.zero_grad()
            pred = model(br.to(device), bm.to(device))
            loss = loss_fn(pred.squeeze(1), br.y.float())
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
                y_pred += logits.flatten().tolist() if logits.ndim else [float(logits)]
        pcc = pearsonr(y_label, y_pred)[0]
        scc = spearmanr(y_label, y_pred)[0]
        rmse = float(np.sqrt(mean_squared_error(y_label, y_pred)))

        if pcc > max_p:
            max_p, max_s, max_rmse = pcc, scc, rmse
            torch.save(model.state_dict(), save_dir / f"{label}_{args.seed}.pth")

        line = (f"epo {epo:4d}: loss={train_loss/len(train_loader):.4f} "
                f"pcc={pcc:.4f} scc={scc:.4f} rmse={rmse:.4f}  best={max_p:.4f}  dt={time.time()-t0:.1f}s")
        print(line, flush=True)
        with open(progress_path, 'a') as pf:
            pf.write(line + "\n")

    print(f"\n=== Done {label} in {(time.time()-run_start)/3600:.2f}h ===")
    print(f"Best PCC  = {max_p:.4f}")
    print(f"Best SCC  = {max_s:.4f}")
    print(f"Best RMSE = {max_rmse:.4f}")


if __name__ == '__main__':
    main()
