"""
Exposure-bias cross-attention (Improvement 2).

Subclasses DeepRSMA's cross_attention stack to accept an optional
`site_bias` tensor of shape [B, 1024] (concat of rna_seq[512] + rna_stru[512])
and add a log-additive bias to the RNA-side attention logits.

Math:
  bias_log = λ_ℓ · log(s + ε)                           ε = 1e-8
  attn_scores_mole += bias_log[:, None, None, :]        # bias on RNA key axis
  attn_scores_rna  += bias_log[:, None, :, None]        # bias on RNA query axis

λ_ℓ is a learnable scalar per encoder layer (shared between the two directions).
If `lambda_trainable=False`, λ is registered as a buffer (not a parameter),
for the "λ fixed at 1.0" ablation.

Padding RNA hidden's structural half with 1.0 → log(1)=0 → no bias on GAT nodes.
"""
import sys
from pathlib import Path
import math

import torch
from torch import nn

DEEPRSMA = Path(__file__).resolve().parents[2] / "DeepRSMA"
if str(DEEPRSMA) not in sys.path:
    sys.path.insert(0, str(DEEPRSMA))

from model.cross_attention import (
    cross_attention,
    CrossFusion,
    Attention,
    Encoder,
    Encoder_1d,
)


BIAS_EPS = 1e-8


class CrossFusionWithBias(CrossFusion):
    """CrossFusion that adds log-exposure bias to RNA-side attention logits."""

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob,
                 lam=None, bias_direction: str = "both"):
        """
        Args:
            lam: an nn.Parameter (or buffer tensor); shared across directions.
            bias_direction: 'both' | 'mole_query' | 'rna_query'
                'mole_query' → bias only on mole→RNA direction (RNA key axis).
                'rna_query'  → bias only on RNA→mole direction (RNA query axis).
                'both'       → bias both directions (default).
        """
        super().__init__(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        if lam is None:
            self.lam = nn.Parameter(torch.ones(1))
        else:
            self.lam = lam
        assert bias_direction in ("both", "mole_query", "rna_query")
        self.bias_direction = bias_direction

    def forward(self, hidden_states, attention_mask, site_bias=None, device_override=None):
        dev = device_override if device_override is not None else hidden_states[0].device

        rna_hidden, rna_mask = hidden_states[0], attention_mask[0]
        mole_hidden, mole_mask = hidden_states[1], attention_mask[1]

        rna_mask = rna_mask.unsqueeze(1).unsqueeze(2)
        rna_mask = ((1.0 - rna_mask) * -10000.0).to(dev)
        mole_mask = mole_mask.unsqueeze(1).unsqueeze(2)
        mole_mask = ((1.0 - mole_mask) * -10000.0).to(dev)

        mixed_q_rna = self.query_rna(rna_hidden)
        mixed_k_rna = self.key_rna(rna_hidden)
        mixed_v_rna = self.value_rna(rna_hidden)
        q_rna = self.transpose_for_scores(mixed_q_rna)
        k_rna = self.transpose_for_scores(mixed_k_rna)
        v_rna = self.transpose_for_scores(mixed_v_rna)

        mixed_q_mole = self.query_mole(mole_hidden)
        mixed_k_mole = self.key_mole(mole_hidden)
        mixed_v_mole = self.value_mole(mole_hidden)
        q_mole = self.transpose_for_scores(mixed_q_mole)
        k_mole = self.transpose_for_scores(mixed_k_mole)
        v_mole = self.transpose_for_scores(mixed_v_mole)

        bias_log = None
        if site_bias is not None:
            bias_log = self.lam * torch.log(site_bias + BIAS_EPS)   # [B, L_rna]

        # mole as query, RNA as key/value
        attn_mole = torch.matmul(q_mole, k_rna.transpose(-1, -2))
        attn_mole = attn_mole / math.sqrt(self.attention_head_size)
        if bias_log is not None and self.bias_direction in ("both", "mole_query"):
            attn_mole = attn_mole + bias_log.unsqueeze(1).unsqueeze(2).to(dev)
        attn_mole = attn_mole + rna_mask
        probs_mole = self.dropout(nn.Softmax(dim=-1)(attn_mole))
        ctx_mole = torch.matmul(probs_mole, v_rna)
        ctx_mole = ctx_mole.permute(0, 2, 1, 3).contiguous()
        ctx_mole = ctx_mole.view(*ctx_mole.size()[:-2], self.all_head_size)

        # RNA as query, mole as key/value
        attn_rna = torch.matmul(q_rna, k_mole.transpose(-1, -2))
        attn_rna = attn_rna / math.sqrt(self.attention_head_size)
        if bias_log is not None and self.bias_direction in ("both", "rna_query"):
            attn_rna = attn_rna + bias_log.unsqueeze(1).unsqueeze(-1).to(dev)
        attn_rna = attn_rna + mole_mask
        probs_rna = self.dropout(nn.Softmax(dim=-1)(attn_rna))
        ctx_rna = torch.matmul(probs_rna, v_mole)
        ctx_rna = ctx_rna.permute(0, 2, 1, 3).contiguous()
        ctx_rna = ctx_rna.view(*ctx_rna.size()[:-2], self.all_head_size)

        return [ctx_rna, ctx_mole], [probs_rna, probs_mole]


class AttentionWithBias(Attention):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob, lam=None, bias_direction="both"):
        super().__init__(hidden_size, num_attention_heads, attention_probs_dropout_prob,
                         hidden_dropout_prob)
        self.self = CrossFusionWithBias(
            hidden_size, num_attention_heads, attention_probs_dropout_prob,
            lam=lam, bias_direction=bias_direction,
        )

    def forward(self, input_tensor, attention_mask, site_bias=None, device_override=None):
        self_output, attention_scores = self.self(
            input_tensor, attention_mask, site_bias=site_bias, device_override=device_override
        )
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_scores


class EncoderWithBias(Encoder):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads,
                 attention_probs_dropout_prob, hidden_dropout_prob, lam=None,
                 bias_direction="both"):
        super().__init__(hidden_size, intermediate_size, num_attention_heads,
                         attention_probs_dropout_prob, hidden_dropout_prob)
        self.attention = AttentionWithBias(
            hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob,
            lam=lam, bias_direction=bias_direction,
        )

    def forward(self, hidden_states, attention_mask, site_bias=None, device_override=None):
        attention_output, attention_scores = self.attention(
            hidden_states, attention_mask, site_bias=site_bias, device_override=device_override
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_scores


class Encoder_1dWithBias(Encoder_1d):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob,
                 lambda_trainable: bool = True, bias_direction: str = "both",
                 lambda_init: float = 0.1):
        super().__init__(n_layer, hidden_size, intermediate_size,
                         num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.lambda_trainable = lambda_trainable
        self.bias_direction = bias_direction

        # FIX B (Phase 4 round 2): default lambda_init=0.1 (was 1.0).
        # At t=0, bias_log = 0.1·log(exposure+eps); for exposure=0.2 (stems), bias≈-0.16,
        # for exposure=0.85 (loops), bias≈-0.016. This is a GENTLE prior that lets model
        # learn its own Q/K/V representations first, then gradually scale λ up if the
        # structure prior is truly useful. Initial λ=1.0 was too disruptive early in training.
        self.lams = nn.ParameterList([
            nn.Parameter(torch.full((1,), float(lambda_init)), requires_grad=lambda_trainable)
            for _ in range(n_layer)
        ])
        self.layer = nn.ModuleList([
            EncoderWithBias(
                hidden_size, intermediate_size, num_attention_heads,
                attention_probs_dropout_prob, hidden_dropout_prob,
                lam=self.lams[i], bias_direction=bias_direction,
            )
            for i in range(n_layer)
        ])

    def forward(self, hidden_states, attention_mask, site_bias=None,
                device_override=None, output_all_encoded_layers=True):
        dev = device_override if device_override is not None else hidden_states[0].device

        seq_rna_emb1 = torch.tensor([0], device=dev).expand(
            hidden_states[0].size(0), hidden_states[0].size(1))
        seq_rna_emb1 = self.mod(seq_rna_emb1)
        hidden_states[0] = hidden_states[0] + seq_rna_emb1

        seq_mole_emb1 = torch.tensor([0], device=dev).expand(
            hidden_states[2].size(0), hidden_states[2].size(1))
        seq_mole_emb1 = self.mod(seq_mole_emb1)
        hidden_states[2] = hidden_states[2] + seq_mole_emb1

        stru_rna_emb1 = torch.tensor([1], device=dev).expand(
            hidden_states[1].size(0), hidden_states[1].size(1))
        stru_rna_emb1 = self.mod(stru_rna_emb1)
        hidden_states[1] = hidden_states[1] + stru_rna_emb1

        stru_mole_emb1 = torch.tensor([1], device=dev).expand(
            hidden_states[3].size(0), hidden_states[3].size(1))
        stru_mole_emb1 = self.mod(stru_mole_emb1)
        hidden_states[3] = hidden_states[3] + stru_mole_emb1

        rna_hidden = torch.cat((hidden_states[0], hidden_states[1]), dim=1)
        mole_hidden = torch.cat((hidden_states[2], hidden_states[3]), dim=1)
        rna_mask = torch.cat((attention_mask[0], attention_mask[1]), dim=1)
        mole_mask = torch.cat((attention_mask[2], attention_mask[3]), dim=1)

        hidden_states = [rna_hidden, mole_hidden]
        attention_mask = [rna_mask, mole_mask]

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(
                hidden_states, attention_mask,
                site_bias=site_bias, device_override=dev,
            )
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attention_scores


class cross_attention_ext(cross_attention):
    """Drop-in replacement for DeepRSMA's cross_attention with optional site_bias."""

    def __init__(self, hidden_dim, lambda_trainable: bool = True, bias_direction: str = "both",
                 lambda_init: float = 0.1):
        super().__init__(hidden_dim)
        self.encoder = Encoder_1dWithBias(
            n_layer=4, hidden_size=hidden_dim, intermediate_size=hidden_dim,
            num_attention_heads=4,
            attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1,
            lambda_trainable=lambda_trainable, bias_direction=bias_direction,
            lambda_init=lambda_init,
        )

    def forward(self, emb, ex_e_mask, device1, site_bias=None):
        encoded_layers, attention_scores = self.encoder(
            emb, ex_e_mask, site_bias=site_bias, device_override=device1,
        )
        return encoded_layers, attention_scores
