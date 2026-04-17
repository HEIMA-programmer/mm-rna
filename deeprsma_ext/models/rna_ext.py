"""
Extended RNA feature extraction: replaces DeepRSMA's line 124
(`out_r = (out_r + out_seq) / 2`) with a StructureAwareAdapter output.

Implements user's Spec Step 4:
  E_adapted = Adapter(LLM, struct_types, ss_edge_index)
  E_combined = Linear(concat(nucleotide_emb, E_adapted))
  E_combined → feed to CNN (unchanged downstream)

Supports:
  - Phase 2 core: default (RNA-FM from data.emb)
  - Phase 4 LLM swap: pass llm_cache to use alternative LLM embeddings
  - Phase 4 adapter ablations: use_gcn, use_struct_emb via StructureAwareAdapter flags
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEEPRSMA = Path(__file__).resolve().parents[2] / "DeepRSMA"
if str(DEEPRSMA) not in sys.path:
    sys.path.insert(0, str(DEEPRSMA))

from model.gnn_model_rna import RNA_feature_extraction
from torch_geometric.nn import global_mean_pool

from deeprsma_ext.models.adapter import StructureAwareAdapter
from deeprsma_ext.data.ss_cache import SSCache
from deeprsma_ext.data.llm_cache import LLMCache


class RNA_feature_extraction_ext(RNA_feature_extraction):
    """Subclass that injects StructureAwareAdapter at the pre-CNN fusion point.

    Args:
        hidden_size: same as parent (typically 128).
        ss_cache: SSCache with fold data for all training RNAs.
        llm_dim: LLM embedding dim (640 RNA-FM, 120 RNABERT, 768 ERNIE-RNA, 1280 RiNALMo).
        adapter_layers: GCN layers in adapter.
        adapter_use_gcn: if False, adapter runs without GCN (ablation).
        adapter_use_struct_emb: if False, adapter skips struct-type embedding (ablation).
        llm_cache: if provided, overrides `data.emb` per-sample (Phase 4 LLM swap).
                   If None, uses `data.emb` as bundled by DeepRSMA's RNA_dataset
                   (RNA-FM 640-d, loaded from representations_cv/*.npy).
        fallback_to_mean: safety valve for uncached RNAs.
    """

    def __init__(
        self,
        hidden_size: int,
        ss_cache: SSCache,
        llm_dim: int = 640,
        adapter_layers: int = 2,
        adapter_use_gcn: bool = True,
        adapter_use_struct_emb: bool = True,
        fusion_type: str = "residual",
        llm_cache: Optional[LLMCache] = None,
        fallback_to_mean: bool = True,
    ):
        super().__init__(hidden_size)
        self.ss_cache = ss_cache
        self.llm_cache = llm_cache
        effective_llm_dim = llm_cache.get_dim() if llm_cache is not None else llm_dim
        self.adapter = StructureAwareAdapter(
            llm_dim=effective_llm_dim,
            hidden=hidden_size,
            n_layers=adapter_layers,
            use_gcn=adapter_use_gcn,
            use_struct_emb=adapter_use_struct_emb,
        )
        # Fusion strategies (configurable via fusion_type):
        #   'residual' (recommended): combined = x_r + α·adapted, α learnable scalar init 0.
        #       At t=0: exact baseline. No information bottleneck. Model learns α via gradient.
        #   'linear': combined = Linear(concat(x_r, adapted)), init to mean equivalent (Fix A).
        #       Matches user spec Step 4 literally but has 256→128 bottleneck.
        self.fusion_type = fusion_type
        if fusion_type == "residual":
            self.adapter_gate = nn.Parameter(torch.zeros(1))
        else:  # "linear"
            self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
            with torch.no_grad():
                half_I = 0.5 * torch.eye(hidden_size)
                self.fusion_proj.weight.copy_(torch.cat([half_I, half_I], dim=1))
                self.fusion_proj.bias.zero_()
        self.fallback_to_mean = fallback_to_mean

    def _get_llm_feat(self, rid: str, fallback: torch.Tensor, device) -> torch.Tensor:
        """Return per-nucleotide LLM embedding for one RNA sample.

        If an LLMCache is registered and has this RNA, return that tensor on device.
        Else return the `fallback` slice (RNA-FM from data.emb).
        """
        if self.llm_cache is not None and self.llm_cache.has(rid):
            return self.llm_cache.get(rid).to(device)
        return fallback

    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index
        emb_raw = data.emb   # [TotalNodes, 640] from DeepRSMA's .npy cache

        x_r = self.x_embedding(x[:, 0].int())
        x_g = self.x_embedding2(x[:, 0].int())

        # GAT on contact map (parent lines 78-82)
        x = F.relu(self.conv1(x_g, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        emb_graph = global_mean_pool(x, data.batch)

        # RNA-FM 640→128 projection (parent line 84)
        emb_proj = F.relu(self.line_emb(emb_raw))

        flag = 0
        node_len = data.rna_len
        out_graph = []
        out_seq = []
        out_r = []
        mask = []
        for j, i in enumerate(node_len):
            count_i = i
            mask.append([] + count_i * [1] + (512 - count_i) * [0])
            x1 = x[flag:flag + count_i]
            temp1 = torch.zeros((512 - x1.size(0), self.hidden_size), device=device)
            x1 = torch.cat((x1, temp1), 0)
            out_graph.append(x1)

            emb1 = emb_proj[flag:flag + count_i]
            temp2 = torch.zeros((512 - emb1.size(0)), 128).to(device)
            emb1 = torch.cat((emb1, temp2), 0)
            out_seq.append(emb1)

            x_r1 = x_r[flag:flag + count_i]
            temp3 = torch.zeros((512 - x_r1.size(0)), 128).to(device)
            x_r1 = torch.cat((x_r1, temp3), 0)
            out_r.append(x_r1)

            flag += count_i

        out_graph = torch.stack(out_graph).to(device)
        out_seq = torch.stack(out_seq).to(device)
        out_r_base = torch.stack(out_r).to(device)
        mask_graph = torch.tensor(mask, dtype=torch.float)
        mask_seq = torch.tensor(mask, dtype=torch.float)

        # ---- Adapter fusion: E_adapted + concat(x_r, E_adapted) → Linear ----
        flag = 0
        combined_list = []
        for j, i in enumerate(node_len):
            count_i = int(i)
            rid = data.t_id[j] if hasattr(data, "t_id") else None
            llm_fallback = emb_raw[flag:flag + count_i]
            # Route to LLM cache if provided
            llm_j = self._get_llm_feat(rid, llm_fallback, device) if rid else llm_fallback
            # If LLMCache returned a larger tensor than count_i (unlikely), truncate
            llm_j = llm_j[:count_i]

            use_adapter = rid is not None and self.ss_cache.has(rid)
            if use_adapter:
                ei, st, _ = self.ss_cache.get(rid)
                L = min(count_i, st.size(0), llm_j.size(0))
                ei = ei.to(device)
                st = st.to(device)
                adapted = self.adapter(llm_j[:L], st[:L], ei)        # [L, hidden]
                if adapted.size(0) < 512:
                    pad = torch.zeros(512 - adapted.size(0), self.hidden_size, device=device)
                    adapted = torch.cat([adapted, pad], dim=0)       # [512, hidden]
                if self.fusion_type == "residual":
                    combined = out_r_base[j] + self.adapter_gate * adapted   # [512, h]
                else:
                    combined = torch.cat([out_r_base[j], adapted], dim=-1)  # [512, 2h]
                    combined = self.fusion_proj(combined)                   # [512, h]
            else:
                if not self.fallback_to_mean:
                    raise KeyError(f"RNA {rid} not in SSCache and fallback_to_mean=False")
                combined = (out_r_base[j] + out_seq[j]) / 2
            combined_list.append(combined)
            flag += count_i
        out_r = torch.stack(combined_list).to(device)

        out_seq_cnn = self.CNN(out_r)
        emb_seq = (out_seq_cnn * (mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)

        return out_seq_cnn, out_graph, mask_seq, mask_graph, emb_seq, emb_graph
