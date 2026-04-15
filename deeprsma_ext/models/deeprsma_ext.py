"""
Extended DeepRSMA model with Improvement 1 (adapter) and/or Improvement 2 (exposure bias).

Configurable via flags for the full Phase 4 ablation matrix:
  - use_adapter / use_bias          — core 2x2
  - adapter_use_gcn                 — adapter ablation
  - adapter_use_struct_emb          — adapter ablation
  - adapter_layers                  — adapter GCN depth
  - bias_direction                  — bias ablation (both/mole_query/rna_query)
  - lambda_trainable                — bias ablation (λ fixed vs learned)
  - llm_cache                       — Phase 4 LLM swap (None = RNA-FM default)
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

DEEPRSMA = Path(__file__).resolve().parents[2] / "DeepRSMA"
if str(DEEPRSMA) not in sys.path:
    sys.path.insert(0, str(DEEPRSMA))

from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention

from deeprsma_ext.models.rna_ext import RNA_feature_extraction_ext
from deeprsma_ext.models.cross_attention_ext import cross_attention_ext
from deeprsma_ext.data.ss_cache import SSCache
from deeprsma_ext.data.llm_cache import LLMCache


RNA_HIDDEN_LEN = 1024
RNA_SEQ_LEN = 512


class DeepRSMA_ext(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        ss_cache: SSCache = None,
        use_adapter: bool = True,
        use_bias: bool = True,
        llm_dim: int = 640,
        adapter_layers: int = 2,
        adapter_use_gcn: bool = True,
        adapter_use_struct_emb: bool = True,
        bias_direction: str = "both",
        lambda_trainable: bool = True,
        llm_cache: Optional[LLMCache] = None,
    ):
        super().__init__()
        if (use_adapter or use_bias) and ss_cache is None:
            raise ValueError("DeepRSMA_ext needs SSCache when use_adapter or use_bias is True")
        self.hidden_dim = hidden_dim
        self.use_adapter = use_adapter
        self.use_bias = use_bias
        self.ss_cache = ss_cache

        if use_adapter:
            self.rna_graph_model = RNA_feature_extraction_ext(
                hidden_size=hidden_dim,
                ss_cache=ss_cache,
                llm_dim=llm_dim,
                adapter_layers=adapter_layers,
                adapter_use_gcn=adapter_use_gcn,
                adapter_use_struct_emb=adapter_use_struct_emb,
                llm_cache=llm_cache,
            )
        else:
            self.rna_graph_model = RNA_feature_extraction(hidden_dim)

        self.mole_graph_model = GNN_molecule(hidden_dim)
        self.mole_seq_model = mole_seq_model(hidden_dim)

        if use_bias:
            self.cross_attention = cross_attention_ext(
                hidden_dim,
                lambda_trainable=lambda_trainable,
                bias_direction=bias_direction,
            )
        else:
            self.cross_attention = cross_attention(hidden_dim)

        self.line1 = nn.Linear(hidden_dim * 2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        self.rna1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.rna2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.ReLU()

    def _build_site_bias(self, rna_batch, device):
        B = len(rna_batch.rna_len)
        bias = torch.ones(B, RNA_HIDDEN_LEN, device=device)
        for j in range(B):
            rid = rna_batch.t_id[j] if hasattr(rna_batch, "t_id") else None
            if rid is None or not self.ss_cache.has(rid):
                continue
            _, _, exp = self.ss_cache.get(rid)
            L = min(int(rna_batch.rna_len[j]), exp.size(0), RNA_SEQ_LEN)
            bias[j, :L] = exp[:L].to(device)
        return bias

    def forward(self, rna_batch, mole_batch):
        device = next(self.parameters()).device
        hidden_dim = self.hidden_dim

        rna_out_seq, rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = \
            self.rna_graph_model(rna_batch, device)

        mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)
        mole_seq_final = (mole_seq_emb[-1] * (mole_mask_seq.to(device).unsqueeze(2))).mean(1).squeeze(1)

        flag = 0
        mole_out_graph = []
        mask = []
        for i in mole_batch.graph_len:
            count_i = i
            x = mole_graph_emb[flag:flag + count_i]
            temp = torch.zeros((128 - x.size(0)), hidden_dim).to(device)
            x = torch.cat((x, temp), 0)
            mole_out_graph.append(x)
            mask.append([] + count_i * [1] + (128 - count_i) * [0])
            flag += count_i
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)

        cross_kwargs = {}
        if self.use_bias:
            cross_kwargs["site_bias"] = self._build_site_bias(rna_batch, device)
        context_layer, _ = self.cross_attention(
            [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
            [rna_mask_seq.to(device), rna_mask_graph.to(device),
             mole_mask_seq.to(device), mole_mask_graph.to(device)],
            device,
            **cross_kwargs,
        )

        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]

        rna_cross_seq = (
            (out_rna[:, 0:512] * (rna_mask_seq.to(device).unsqueeze(2))).mean(1).squeeze(1)
            + rna_seq_final
        ) / 2
        rna_cross_stru = (
            (out_rna[:, 512:] * (rna_mask_graph.to(device).unsqueeze(2))).mean(1).squeeze(1)
            + rna_graph_final
        ) / 2
        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout(self.relu(self.rna1(rna_cross))))

        mole_cross_seq = (
            (out_mole[:, 0:128] * (mole_mask_seq.to(device).unsqueeze(2))).mean(1).squeeze(1)
            + mole_seq_final
        ) / 2
        mole_cross_stru = (
            (out_mole[:, 128:] * (mole_mask_graph.to(device).unsqueeze(2))).mean(1).squeeze(1)
            + mole_graph_final
        ) / 2
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout(self.relu(self.mole1(mole_cross))))

        out = torch.cat((rna_cross, mole_cross), 1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        return out
