import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class HypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, incidence: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.register_buffer("H", incidence, persistent=False)

    @staticmethod
    def _build_hypergraph_adjacency(H: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        W = torch.diag(edge_weight)
        dv = H @ W @ torch.ones(H.shape[1], dtype=H.dtype, device=H.device)
        de = H.sum(dim=0)

        dv_inv_sqrt = torch.diag(torch.pow(dv + eps, -0.5))
        de_inv = torch.diag(torch.pow(de + eps, -1.0))

        A = dv_inv_sqrt @ H @ W @ de_inv @ H.t() @ dv_inv_sqrt
        return A

    def forward(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        A = self._build_hypergraph_adjacency(self.H, edge_weight)
        x = self.linear(x)
        out = torch.einsum("ij,bjd->bid", A, x)
        return out


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights if return_attention else None


class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TemporalSelfAttentionBlock(d_model, nhead, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_list: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn = layer(x, return_attention=return_attention)
            if attn is not None:
                attn_list.append(attn)
        return x, attn_list


class HGNNTemporalTransformer(nn.Module):
    def __init__(
        self,
        node_input_dims: List[int],
        incidence: torch.Tensor,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        co_col_idx: int = 2,
        co_history_steps: int = 60,
    ):
        super().__init__()
        self.node_projectors = nn.ModuleList([nn.Linear(d, d_model) for d in node_input_dims])
        self.temporal_input_proj = nn.Linear(d_model + 1, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model)
        self.node_attention = nn.Linear(d_model, 1)
        self.time_weight = nn.Linear(d_model, 1)
        self.co_col_idx = int(co_col_idx)
        self.co_history_steps = int(max(1, co_history_steps))

        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        self.hgc1 = HypergraphConv(d_model, d_model, incidence)
        self.hgc2 = HypergraphConv(d_model, d_model, incidence)
        self.hyperedge_weight = nn.Parameter(torch.ones(incidence.shape[1]))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.last_attention_info: Dict[str, torch.Tensor] = {}

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        node_inputs: List[torch.Tensor],
        raw_sequence: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor | List[torch.Tensor]]]:
        if raw_sequence is None:
            raise ValueError("raw_sequence must be provided to build CO_history input.")

        projected = []
        for i, node_seq in enumerate(node_inputs):
            x = self.node_projectors[i](node_seq)
            projected.append(x)

        # x: [B, T, N, D]
        x = torch.stack(projected, dim=2)
        bsz, tlen, n_nodes, d_model = x.shape

        # Hypergraph first: run graph propagation independently at each timestep.
        x_graph = x.reshape(bsz * tlen, n_nodes, d_model)
        edge_weight = torch.clamp(self.hyperedge_weight, min=1e-6)
        h1 = self.act(self.hgc1(x_graph, edge_weight))
        h1 = self.dropout(h1)
        h2 = self.hgc2(h1, edge_weight)
        h = self.act(x_graph + h2)
        h = h.reshape(bsz, tlen, n_nodes, d_model)

        # Node-level attention for per-timestep node importance.
        node_attn = torch.softmax(self.node_attention(h), dim=2)
        global_seq = (h * node_attn).sum(dim=2)

        co_hist = raw_sequence[:, :, self.co_col_idx].unsqueeze(-1)
        win = min(self.co_history_steps, global_seq.size(1), co_hist.size(1))
        global_seq = global_seq[:, -win:, :]
        co_hist = co_hist[:, -win:, :]
        temporal_src = torch.cat([global_seq, co_hist], dim=-1)

        # Temporal second: model dynamics over time after graph reasoning.
        temporal_in = self.temporal_input_proj(temporal_src)
        temporal_in = self.pos_encoding(temporal_in)
        temporal_h, temporal_attn = self.temporal_encoder(
            temporal_in,
            return_attention=return_attention,
        )

        # Temporal pooling instead of last-step readout.
        time_attn = torch.softmax(self.time_weight(temporal_h), dim=1)
        z = (temporal_h * time_attn).sum(dim=1)
        pred = self.reg_head(z).squeeze(-1)

        self.last_attention_info = {
            "node_attention": node_attn.detach(),
            "time_attention": time_attn.detach(),
            "hyperedge_weight": edge_weight.detach(),
        }
        if return_attention:
            attention_info: Dict[str, torch.Tensor | List[torch.Tensor]] = {
                "node_attention": node_attn,
                "time_attention": time_attn,
                "temporal_attention": temporal_attn,
                "hyperedge_weight": edge_weight,
            }
            return pred, attention_info

        return pred


def build_incidence(node_order: List[str], hyperedges: List[List[str]]) -> torch.Tensor:
    n_nodes = len(node_order)
    n_edges = len(hyperedges)
    node_to_idx = {name: idx for idx, name in enumerate(node_order)}

    H = torch.zeros((n_nodes, n_edges), dtype=torch.float32)
    for e_idx, edge_nodes in enumerate(hyperedges):
        for name in edge_nodes:
            H[node_to_idx[name], e_idx] = 1.0
    return H
