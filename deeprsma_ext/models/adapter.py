"""
StructureAwareAdapter: GCN over the RNA secondary-structure graph.

Takes frozen per-nucleotide LLM embeddings + per-nucleotide structure-type IDs,
runs n_layers of GCN message passing on the SS graph (backbone ∪ base-pair edges),
returns a per-nucleotide hidden representation of dimension `hidden`.

Supports Phase 4 ablations:
  - use_gcn=False        → identity (no graph message passing)
  - use_struct_emb=False → drop struct-type embedding, project LLM to full hidden
  - n_layers             → configurable GCN depth (typical: 1, 2, 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class StructureAwareAdapter(nn.Module):
    """GCN adapter that fuses LLM sequence semantics with RNA secondary structure.

    Args:
        llm_dim: dim of frozen LLM embedding (640 RNA-FM, 120 RNABERT, 768 ERNIE-RNA, 1280 RiNALMo).
        hidden: output dim (matches DeepRSMA's hidden_dim, typically 128).
        n_layers: number of GCN layers.
        n_struct_types: number of structural element classes.
            5 classes: 0=stem, 1=hairpin, 2=internal/bulge, 3=multiloop, 4=dangling.
        struct_dim: dim reserved for struct-type emb inside `hidden`.
        dropout: after each GCN layer.
        use_gcn: if False, skip GCN message passing (ablation).
        use_struct_emb: if False, drop struct-type embedding (ablation).
    """

    def __init__(
        self,
        llm_dim: int = 640,
        hidden: int = 128,
        n_layers: int = 2,
        n_struct_types: int = 5,
        struct_dim: int = 16,
        dropout: float = 0.2,
        use_gcn: bool = True,
        use_struct_emb: bool = True,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden = hidden
        self.use_gcn = use_gcn
        self.use_struct_emb = use_struct_emb

        if use_struct_emb:
            assert struct_dim < hidden, "struct_dim must be less than hidden when enabled"
            self.llm_proj = nn.Linear(llm_dim, hidden - struct_dim)
            self.struct_emb = nn.Embedding(n_struct_types, struct_dim)
        else:
            self.llm_proj = nn.Linear(llm_dim, hidden)
            self.struct_emb = None

        if use_gcn:
            self.gcn = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        else:
            self.gcn = nn.ModuleList([])  # empty

        self.ln = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        llm_feat: torch.Tensor,
        struct_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_llm = self.llm_proj(llm_feat)
        if self.use_struct_emb:
            x_struct = self.struct_emb(struct_type_ids)
            x = torch.cat([x_llm, x_struct], dim=-1)
        else:
            x = x_llm

        for conv in self.gcn:
            x_new = F.relu(conv(x, edge_index))
            x_new = self.dropout(x_new)
            x = x + x_new   # residual

        return self.ln(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    L = 30
    for cfg in [
        dict(use_gcn=True, use_struct_emb=True, n_layers=2),
        dict(use_gcn=False, use_struct_emb=True, n_layers=2),
        dict(use_gcn=True, use_struct_emb=False, n_layers=2),
        dict(use_gcn=True, use_struct_emb=True, n_layers=3),
    ]:
        m = StructureAwareAdapter(llm_dim=640, hidden=128, **cfg)
        llm = torch.randn(L, 640)
        st = torch.randint(0, 5, (L,))
        ei = torch.tensor(
            [list(range(L - 1)) + list(range(1, L)),
             list(range(1, L)) + list(range(L - 1))], dtype=torch.long,
        )
        out = m(llm, st, ei)
        print(cfg, "→ out", tuple(out.shape), "params", sum(p.numel() for p in m.parameters()))
