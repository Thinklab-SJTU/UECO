import math
import torch
from torch import nn, Tensor
from typing import Sequence, Union
import torch.nn.functional as F
from model.out_layer.utils import GroupNorm32
from torch_scatter import scatter_softmax, scatter_add
from torch_sparse import SparseTensor
from torch_sparse import (
    sum as sparse_sum, mean as sparse_mean, max as sparse_max
)
from model.embedder.utils import (
    ScalarEmbeddingSine1D, ScalarEmbeddingSine2D, 
    ScalarEmbeddingSine3D, PositionEmbeddingSine
)
from model.embedder.time import timestep_embedding


####################################
#          Small utilities         #
####################################

def reshape_by_heads(qkv: Tensor, num_heads: int) -> Tensor:
    B, N, _ = qkv.size()
    q_reshaped = qkv.view(B, N, num_heads, -1)
    return q_reshaped.transpose(1, 2)  # (B, H, N, D)

def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        x = input1 + input2                      # (B, N, d)
        x = x.transpose(1, 2)                    # (B, d, N)
        x = self.norm(x)
        x = x.transpose(1, 2)                    # (B, N, d)
        return x

class AddAndLayerNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        x = input1 + input2                      # (N, d)
        x = self.norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.W2(F.relu(self.W1(x)))


####################################
#       Outer GCN Components       #
####################################

class GNNDenseLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        aggregation: str = "sum",
        norm: str = "layer",
        learn_norm: bool = True,
        track_norm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.D = nn.Linear(2, hidden_dim, bias=True)

        if norm == "batch":
            self.norm_x = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
            self.norm_e = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        else:
            self.norm_x = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
            self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)

    def aggregate(self, Vx: Tensor, gates: Tensor, graph: Tensor) -> Tensor:
        # Vx: (B,V,H) or (B,V,V,H) depending on caller
        Vx = Vx * gates
        if self.aggregation == "mean":
            return torch.sum(Vx, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vx))
        elif self.aggregation == "max":
            return torch.max(Vx, dim=2)[0]
        else:
            return torch.sum(Vx, dim=2)

    def forward(self, x: Tensor, e: Tensor, graph: Tensor):
        """
          x: (B,V,H)
          e: (B,V,V,H)
          edge: (B,V,V)
          graph: (B,V,V) (adjacency)
        """
        B, V, H = x.shape
        # ef = torch.stack([edges_feature, edges_feature.transpose(1, 2)], dim=3)  # (B,V,V,2)
        # De = self.D(ef)

        Ux = self.U(x)                                      # (B,V,H)

        Vx = self.V(x)                                      # (B,V,H)
        Vx = Vx.unsqueeze(1).expand(-1, V, -1, -1)          # (B,V,V,H)

        Ax = self.A(x)                                      # (B,V,H)
        Bx = self.B(x)                                      # (B,V,H)

        Ce = self.C(e)                                      # (B,V,V,H)
        e_new = Ax.unsqueeze(1) + Bx.unsqueeze(2) + Ce # (B,V,V,H)

        gates = torch.sigmoid(e_new)                        # (B,V,V,H)

        x_msg = self.aggregate(Vx, gates, graph)            # (B,V,H)
        x_out = Ux + x_msg

        x_out = x_out.view(B * V, H)
        x_out = F.relu(self.norm_x(x_out)).view(B, V, H)

        e_out = e_new.view(B * V * V, H)
        e_out = F.relu(self.norm_e(e_out)).view(B, V, V, H)

        return x_out, e_out

class GNNDenseBlock(nn.Module):
    def __init__(
        self,
        **model_params
    ):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        num_gcn_layers = model_params["num_gcn_layers"]
        self.dual = model_params["dual"]

        self.gnn_layers_row = nn.ModuleList(
            [
                GNNDenseLayer(
                    hidden_dim=embedding_dim,
                    aggregation="sum",
                    norm="layer",
                    learn_norm=True,
                    track_norm=False
                ) for _ in range(num_gcn_layers)
            ]
        )

        self.out_layers_row = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embedding_dim, elementwise_affine=True),
                nn.SiLU(),
                zero_module(nn.Linear(embedding_dim, embedding_dim)),
            ) for _ in range(num_gcn_layers)
        ])

        if self.dual:
            self.gnn_layers_col = nn.ModuleList(
                [
                    GNNDenseLayer(
                        hidden_dim=embedding_dim,
                        aggregation="sum",
                        norm="layer",
                        learn_norm=True,
                        track_norm=False,
                    ) for _ in range(num_gcn_layers)
                ]
            )

            self.out_layers_col = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(embedding_dim, elementwise_affine=True),
                    nn.SiLU(),
                    zero_module(nn.Linear(embedding_dim, embedding_dim)),
                ) for _ in range(num_gcn_layers)
            ])
        
            self.merge_edge = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

    def forward(
        self,
        x_row: Tensor,
        x_col: Tensor,
        edge_emb: Tensor,
        edge_index: Tensor = None
    ) -> Sequence[Tensor]:
        # GCN4CO-style edge 更新
        B, N, _ = x_row.shape
        graph = torch.ones(B, N, N, device=x_row.device)
        edge_emb_T = edge_emb.transpose(1, 2)  # (B,N,N,d)
        for gnn_layer, out_layer in zip(self.gnn_layers_row, self.out_layers_row):
            x_in, e_in = x_row, edge_emb
            x_row, edge_emb = gnn_layer(x_row, edge_emb, graph)
            x_row = x_row + x_in
            edge_emb = e_in + out_layer(edge_emb)


        if self.dual:
            for gnn_layer, out_layer in zip(self.gnn_layers_col, self.out_layers_col):
                x_in, e_in = x_col, edge_emb_T
                x_col, edge_emb_T = gnn_layer(x_col, edge_emb_T, graph)
                x_col = x_col + x_in
                edge_emb_T = e_in + out_layer(edge_emb_T)

            edge_emb_to_merge = torch.cat([edge_emb, edge_emb_T], dim=-1)
            edge_out = self.merge_edge(edge_emb_to_merge)
            return x_row, x_col, edge_out
        
        else: return x_row, edge_emb


class GNNSparseLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        aggregation: str = "sum",
        norm: str = "layer",
        learn_norm: bool = True,
        track_norm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        # 与原版 GCN4CO sparse 对齐：U, V, A, B, C
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 节点 / 边各自的归一化
        if norm == "batch":
            self.norm_x = nn.BatchNorm1d(
                hidden_dim, affine=learn_norm, track_running_stats=track_norm
            )
            self.norm_e = nn.BatchNorm1d(
                hidden_dim, affine=learn_norm, track_running_stats=track_norm
            )
        else:
            self.norm_x = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
            self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)

    def aggregate(
        self,
        Vx: Tensor,          # (E, H)
        gates: Tensor,       # (E, H)
        edge_index: Tensor,  # (2, E)
        nodes_num: int
    ) -> Tensor:
        """
        GCN4CO 的稀疏聚合逻辑：
          - 用 SparseTensor 存 Vx * gates，行 = src，列 = tgt
          - 按 dim=1 聚合得到每个节点的聚合消息
        """
        value = Vx * gates  # (E, H)

        sparseVh = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=value,
            sparse_sizes=(nodes_num, nodes_num),
        )

        if self.aggregation == "mean":
            return sparse_mean(sparseVh, dim=1)
        elif self.aggregation == "max":
            return sparse_max(sparseVh, dim=1)
        else:  # "sum"
            return sparse_sum(sparseVh, dim=1)

    def forward(
        self,
        x: Tensor,          # (V, H)  节点特征
        e: Tensor,          # (E, H)  边特征
        edge_index: Tensor  # (2, E), [src, tgt]
    ) -> Sequence[Tensor]:
        """
        x_out = ReLU( Norm_x( Ux + Agg( Vx, gates, edge_index ) ) )
        e_out = ReLU( Norm_e( Ax[src] + Bx[tgt] + C e ) )
        """
        src, tgt = edge_index
        nodes_num = x.size(0)
        # ----- 节点线性变换 Ux -----
        Ux = self.U(x)                 # (V, H)
        # ----- Vx：用于聚合的邻居表征 -----
        Vx = self.V(x[tgt])            # (E, H)
        # ----- 从节点到边的 message，用于更新 e -----
        Ax = self.A(x)                 # (V, H), source 侧
        Bx = self.B(x)                 # (V, H), target 侧
        Ce = self.C(e)                 # (E, H)
        e_new = Ax[src] + Bx[tgt] + Ce # (E, H)
        # ----- gates：边级别门控 -----
        gates = torch.sigmoid(e_new)   # (E, H)
        # ----- 聚合到节点：GCN4CO 逻辑 -----
        x_msg = self.aggregate(Vx, gates, edge_index, nodes_num)  # (V, H)
        x_new = Ux + x_msg                                       # (V, H)
        # ----- 归一化 + 激活 -----
        x_out = F.relu(self.norm_x(x_new))   # (V, H)
        e_out = F.relu(self.norm_e(e_new))   # (E, H)

        return x_out, e_out


class GNNSparseBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        num_gcn_layers = model_params["num_gcn_layers"]
        self.dual = model_params["dual"]

        self.gnn_layers_row = nn.ModuleList([
            GNNSparseLayer(
                hidden_dim=embedding_dim,
                aggregation="sum",
                norm="layer",
                learn_norm=True,
                track_norm=False,
            ) for _ in range(num_gcn_layers)
        ])

        self.out_layers_row = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.SiLU(),
                zero_module(nn.Linear(embedding_dim, embedding_dim)),
            ) for _ in range(num_gcn_layers)
        ])

        if self.dual:
            self.gnn_layers_col = nn.ModuleList([
                GNNSparseLayer(
                    hidden_dim=embedding_dim,
                    aggregation="sum",
                    norm="layer",
                    learn_norm=True,
                    track_norm=False,
                ) for _ in range(num_gcn_layers)
            ])

            self.out_layers_col = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.SiLU(),
                    zero_module(nn.Linear(embedding_dim, embedding_dim)),
                ) for _ in range(num_gcn_layers)
            ])

            self.merge_edge = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

    def forward(
        self,
        x_row: Tensor,     # (V,d)
        x_col: Tensor,     # (V,d)
        edge_emb: Tensor,  # (E,d)
        edge_index: Tensor
    ) -> Sequence[Tensor]:

        src, tgt = edge_index

        # ========= row view =========
        x_r = x_row
        e_r = edge_emb

        for gnn, out in zip(self.gnn_layers_row, self.out_layers_row):
            x_in, e_in = x_r, e_r
            x_r, e_tmp = gnn(x_r, e_r, edge_index)    # (V,d), (E,d)
            x_r = x_in + x_r                           # node residual
            e_r = e_in + out(e_tmp)                    # edge residual

        if self.dual:
            # ========= col view (reverse graph) =========
            edge_index_T = torch.stack([tgt, src], dim=0)

            x_c = x_col
            e_c = edge_emb

            for gnn, out in zip(self.gnn_layers_col, self.out_layers_col):
                x_in, e_in = x_c, e_c
                x_c, e_tmp = gnn(x_c, e_c, edge_index_T)   # (V,d), (E,d)
                x_c = x_in + x_c                           # node residual
                e_c = e_in + out(e_tmp)                    # edge residual

            # ========= merge two edge views =========
            edge_out = self.merge_edge(torch.cat([e_r, e_c], dim=-1))

            return x_r, x_c, edge_out
        else:
            return x_r, e_r
        

################################################
#   Edge-Aware Mix-Score Attention Components  #
################################################

class DenseMixedScoreEdge_MultiHeadAttention(nn.Module):
    """
    (Dense version)
    Mixed attention with content score (QK) and edge-aware score.
    This is for dense version only.

    Inputs:
        q: (B, H, R, D)
        k: (B, H, C, D)
        v: (B, H, C, D)
        edge_emb: (B, R, C, embedding_dim)

    Output:
        out: (B, R, H*D)
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        num_heads = model_params["num_heads"]
        ms_hidden_dim = model_params["ms_hidden_dim"]
        embedding_dim = model_params["embedding_dim"]
        qkv_dim = self.model_params["qkv_dim"]
        self.scale = math.sqrt(qkv_dim)

        # edge -> per-head scalar score
        self.edge_proj = nn.Linear(embedding_dim, num_heads, bias=True)

        # per-head MLP: [dot_score, edge_score] -> mixed_score
        self.mix_fc1 = nn.Linear(2, ms_hidden_dim, bias=True)
        self.mix_fc2 = nn.Linear(ms_hidden_dim, 1, bias=True)

    def forward(
        self,
        q: Tensor,          # (B, H, R, D) R=C=N, H: num_heads, D: qkv_dim
        k: Tensor,          # (B, H, C, D)
        v: Tensor,          # (B, H, C, D)
        edge_emb: Tensor    # (B, R, C, embedding_dim)
    ) -> Tensor:
        B, H, R, D = q.shape
        C = k.size(2)

        # ---- 1. content score ----
        # (B, H, R, C)
        dot_score = torch.matmul(q, k.transpose(-1, -2)) / self.scale

        # ---- 2. edge-aware score ----
        # (B, R, C, H)
        edge_score = self.edge_proj(edge_emb)
        # -> (B, H, R, C)
        edge_score = edge_score.permute(0, 3, 1, 2)

        # ---- 3. learned fusion (per head) ----
        # stack last dim = 2
        # (B, H, R, C, 2)
        score_pair = torch.stack([dot_score, edge_score], dim=-1)

        # apply MLP
        mixed = self.mix_fc1(score_pair)
        mixed = F.relu(mixed)
        mixed = self.mix_fc2(mixed).squeeze(-1)  # (B, H, R, C)

        # ---- 4. attention ----
        attn = torch.softmax(mixed, dim=-1)

        # ---- 5. aggregate ----
        # (B, H, R, D)
        out = torch.matmul(attn, v)

        # -> (B, R, H*D)
        out = out.transpose(1, 2).contiguous().view(B, R, H * D)
        return out


# class SparseMixedScoreEdge_MultiHeadAttention(nn.Module):
#     def __init__(self, **model_params):
#         super().__init__()
#         self.model_params = model_params

#         num_heads = model_params["num_heads"]
#         ms_hidden_dim = model_params["ms_hidden_dim"]
#         embedding_dim = model_params["embedding_dim"]
#         qkv_dim = self.model_params["qkv_dim"]
#         self.scale = math.sqrt(qkv_dim)
#         self.qkv_dim = qkv_dim
#         self.num_heads = num_heads

#         # edge -> per-head scalar score
#         self.edge_proj = nn.Linear(embedding_dim, num_heads, bias=True)

#         # per-head MLP: [dot_score, edge_score] -> mixed_score
#         self.mix_fc1 = nn.Linear(2, ms_hidden_dim, bias=True)
#         self.mix_fc2 = nn.Linear(ms_hidden_dim, 1, bias=True)

#     def forward(
#         self,
#         q: Tensor,          # (H,V,D)
#         k: Tensor,          # (H,V,D)
#         v: Tensor,          # (H,V,D)
#         edge_emb: Tensor,   # (E,d)
#         edge_index: Tensor  # (2,E)
#     ) -> Tensor:
#         src, tgt = edge_index   # (E,)
#         V = q.shape[1]
        
#         q_e = q[:, src]   # (H,E,D)
#         k_e = k[:, tgt]   # (H,E,D)
#         v_e = v[:, tgt]   # (H,E,D)

#         # ======================================================
#         # 2. dot-product score
#         # ======================================================
#         dot_score = (q_e * k_e).sum(dim=-1) / self.scale   # (H,E)

#         # ======================================================
#         # 3. edge-aware score
#         # ======================================================
#         edge_score = self.edge_proj(edge_emb)   # (E,H)
#         edge_score = edge_score.transpose(0, 1) # (H,E)

#         # ======================================================
#         # 4. mixed score: MLP([dot, edge])
#         # ======================================================
#         score_pair = torch.stack([dot_score, edge_score], dim=-1)  # (H,E,2)
#         mixed = self.mix_fc1(score_pair)
#         mixed = F.relu(mixed)
#         mixed = self.mix_fc2(mixed).squeeze(-1)                    # (H,E)

#         # ======================================================
#         # 5. attention / gating
#         # ======================================================
#         attn = torch.zeros_like(mixed)
#         for h in range(self.num_heads):
#             attn[h] = scatter_softmax(mixed[h], src)

#         # ======================================================
#         # 6. aggregation
#         # ======================================================
#         msg = attn.unsqueeze(-1) * v_e    # (H,E,D)
#         out = torch.zeros((self.num_heads, V, self.qkv_dim), device=msg.device)

#         for h in range(self.num_heads):
#             out[h].index_add_(0, src, msg[h])

#         out = out.transpose(0, 1).contiguous().view(V, self.num_heads * self.qkv_dim)
#         return out

class SparseMixedScoreEdge_MultiHeadAttention(nn.Module):
    """
    Sparse version mixed-attn:
      - q,k,v: (H,V,D)
      - edge_emb: (E,d_model)
      - edge_index: (2,E), src->tgt
    Output:
      - out_concat: (V, H*D)
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.num_heads = model_params["num_heads"]
        self.qkv_dim = model_params["qkv_dim"]
        self.scale = math.sqrt(self.qkv_dim)

        ms_hidden_dim = model_params["ms_hidden_dim"]
        d_model = model_params["embedding_dim"]

        # edge -> per-head scalar score (E,d_model) -> (E,H)
        self.edge_proj = nn.Linear(d_model, self.num_heads, bias=True)

        # shared MLP applied head-wise on pair [dot, edge]
        self.mix_fc1 = nn.Linear(2, ms_hidden_dim, bias=True)
        self.mix_fc2 = nn.Linear(ms_hidden_dim, 1, bias=True)

    def forward(
        self,
        q: Tensor,          # (H,V,D)
        k: Tensor,          # (H,V,D)
        v: Tensor,          # (H,V,D)
        edge_emb: Tensor,   # (E,d_model)
        edge_index: Tensor  # (2,E)
    ) -> Tensor:
        src, tgt = edge_index  # (E,), (E,)
        H, V, D = q.shape
        E = src.numel()

        # ---- gather to edges ----
        # q_e,k_e,v_e: (H,E,D)
        q_e = q.index_select(1, src)
        k_e = k.index_select(1, tgt)
        v_e = v.index_select(1, tgt)

        # ---- dot score: (H,E) ----
        dot_score = (q_e * k_e).sum(dim=-1) / self.scale

        # ---- edge score: (H,E) ----
        edge_score = self.edge_proj(edge_emb).transpose(0, 1)  # (H,E)

        # ---- mixed score via MLP: (H,E) ----
        score_pair = torch.stack([dot_score, edge_score], dim=-1)  # (H,E,2)
        mixed = self.mix_fc2(F.relu(self.mix_fc1(score_pair))).squeeze(-1)  # (H,E)

        # =========================================================
        # Parallel scatter_softmax over (head, src) groups (NO LOOP)
        # =========================================================
        # group_id: (H,E) with unique groups per head
        # group_id(h,e) = src(e) + h*V
        head_offset = (torch.arange(H, device=src.device).view(H, 1) * V)  # (H,1)
        group_id = src.view(1, E) + head_offset                            # (H,E)

        # flatten for scatter ops
        mixed_f = mixed.reshape(-1)           # (H*E,)
        group_f = group_id.reshape(-1)        # (H*E,)
        attn_f = scatter_softmax(mixed_f, group_f, dim=0)  # (H*E,)
        attn = attn_f.view(H, E)             # (H,E)

        # =========================================================
        # Parallel aggregation (NO LOOP)
        # =========================================================
        # msg: (H,E,D)
        msg = attn.unsqueeze(-1) * v_e

        # scatter_add expects same shape; we aggregate along "src" per head:
        # out: (H,V,D) by scatter_add over dim=1 using src broadcasted
        src_broadcast = src.view(1, E, 1).expand(H, E, D)  # (H,E,D)
        out = scatter_add(msg, src_broadcast, dim=1, dim_size=V)  # (H,V,D)

        # concat heads: (V, H*D)
        out_concat = out.transpose(0, 1).contiguous().view(V, H * D)
        return out_concat

class DenseEdgeAwareCrossAttnBlock(nn.Module):
    """
    MatNet EncodingBlock 的变体：
      - Q,K,V 和原版一致
      - MixedScore 使用 edge_emb 作为外部信息
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        num_heads = model_params["num_heads"]
        qkv_dim = model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)

        self.mixed_score_MHA = DenseMixedScoreEdge_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, embedding_dim)

        self.add_norm_1 = AddAndInstanceNormalization(**model_params)
        self.ffn = FeedForward(**model_params)
        self.add_norm_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb: Tensor, col_emb: Tensor, edge_emb: Tensor) -> Tensor:
        """
        row_emb: (B, R, d)
        col_emb: (B, C, d)
        edge_emb: (B, R, C, e_dim)
        """
        num_heads = self.model_params["num_heads"]

        q = reshape_by_heads(self.Wq(row_emb), num_heads)   # (B,H,R,D)
        k = reshape_by_heads(self.Wk(col_emb), num_heads)   # (B,H,C,D)
        v = reshape_by_heads(self.Wv(col_emb), num_heads)   # (B,H,C,D)

        out_concat = self.mixed_score_MHA(q, k, v, edge_emb)
        out = self.multi_head_combine(out_concat)          # (B,R,d)

        out1 = self.add_norm_1(row_emb, out)
        out2 = self.ffn(out1)
        out3 = self.add_norm_2(out1, out2)
        return out3  # (B,R,d)

class SparseEdgeAwareCrossAttnBlock(nn.Module):
    """
    Sparse version of MatNetEncodingBlockWithEdge.

    Semantics are strictly equivalent to the dense version:
      - MixedScore attention = f(dot(Q,K), edge_emb)
      - ATSP: scatter-softmax over outgoing edges per source
      - TSP: sigmoid gating
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        d_model = model_params["embedding_dim"]
        num_heads = model_params["num_heads"]
        qkv_dim = model_params["qkv_dim"]
        ms_hidden_dim = model_params["ms_hidden_dim"]

        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.scale = math.sqrt(qkv_dim)

        # ---- projections ----
        self.Wq = nn.Linear(d_model, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(d_model, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(d_model, num_heads * qkv_dim, bias=False)

        self.mixed_score_MHA = SparseMixedScoreEdge_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, d_model)

        # sparse: LayerNorm is safer
        self.add_norm_1 = AddAndLayerNormalization(**model_params)
        self.ffn = FeedForward(**model_params)
        self.add_norm_2 = AddAndLayerNormalization(**model_params)

    def _proj(self, W: nn.Linear, x: Tensor) -> Tensor:
        """
        (V,d) → (H,V,D)
        """
        V = x.size(0)
        x = W(x).view(V, self.num_heads, self.qkv_dim)
        return x.transpose(0, 1)

    def forward(
        self,
        task: str,
        row_emb: Tensor,     # (V,d)
        col_emb: Tensor,     # (V,d)
        edge_emb: Tensor,    # (E,d)
        edge_index: Tensor,  # (2,E)
    ):
        # ======================================================
        # 1. Q K V (node-level), then index to edges
        # ======================================================
        q = self._proj(self.Wq, row_emb)   # (H,V,D)
        k = self._proj(self.Wk, col_emb)   # (H,V,D)
        v = self._proj(self.Wv, col_emb)   # (H,V,D)

        out_concat = self.mixed_score_MHA.forward(q, k, v, edge_emb, edge_index)
        out = self.multi_head_combine(out_concat)

        # ======================================================
        # 7. residual + FFN
        # ======================================================
        out1 = self.add_norm_1(row_emb, out)
        out2 = self.ffn(out1)
        out3 = self.add_norm_2(out1, out2)

        return out3


class DenseEdgeAwareEncoderLayer(nn.Module):
    """
    一层：
      1) Transormer-based row/col 更新（外部信息来自 edge_emb）
      2) GCN4CO-style edge_emb 更新（只动 edge，不再改 node）
    """
    def __init__(self, task: str, **model_params):
        super().__init__()
        self.task = task
        self.row_block = DenseEdgeAwareCrossAttnBlock(**model_params)
        self.col_block = DenseEdgeAwareCrossAttnBlock(**model_params)
        
    def forward(
        self,
        row_emb: Tensor,      # (B,N,d)
        col_emb: Tensor,      # (B,N,d)
        edge_emb: Tensor,     # (B,N,N,d)
        # raw_edge_feat: Tensor # (B,N,N) 原始距离（用于 ATSP asym）
        edge_index: Tensor=None
    ):
        # MatNet-style 双向 cross-attn（对称 ATSP/TSP）
        row_out = self.row_block(row_emb, col_emb, edge_emb)              # (B,N,d)
        edge_emb_T = edge_emb.transpose(1, 2)                              # (B,N,N,d)
        col_out = self.col_block(col_emb, row_emb, edge_emb_T)            # (B,N,d)

        return row_out, col_out
    

class SparseEdgeAwareEncoderLayer(nn.Module):
    def __init__(self, task: str, **model_params):
        super().__init__()
        self.task = task
        self.row_block = SparseEdgeAwareCrossAttnBlock(**model_params)
        self.dual = model_params["dual"]
        if self.dual:
            self.col_block = SparseEdgeAwareCrossAttnBlock(**model_params)

    def forward(
        self,
        row_emb: Tensor,      # (V,d)
        col_emb: Tensor,      # (V,d)
        edge_emb: Tensor,     # (E,d)
        edge_index: Tensor
    ):
        row_out = self.row_block(
            self.task, row_emb, col_emb, edge_emb, edge_index
        )
        if self.dual:
            col_out = self.col_block(
                self.task, col_emb, row_emb, edge_emb,
                edge_index[[1, 0]]
            )
            return row_out, col_out
        else:
            return row_out


####################################
#            UECOEncoder            #
####################################

class UECOEncoder(nn.Module):
    def __init__(
        self,
        task: str,
        paradigm: str,
        sparse: bool = False,
        sparse_factor: int = -1,
        **model_params
    ):
        super().__init__()

        num_att_layers = model_params["num_att_layers"]
        hidden_dim = model_params["embedding_dim"]
        self.hidden_dim = hidden_dim
        self.task = task
        self.paradigm = paradigm
        self.sparse = sparse
        self.sparse_factor = sparse_factor
        self.dual = model_params.get("dual", task in ["TSP", "ATSP", "CVRP"])
        if "dual" not in model_params:
            model_params["dual"] = self.dual

        # time embedders (for diffusion)
        self.time_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_time_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_att_layers)
        ])
        self.edge_time_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_att_layers)
        ])

        # mask embedder (for expander)
        self.mask_embed = nn.Embedding(2, hidden_dim)
        nn.init.constant_(self.mask_embed.weight, 0)

        if not self.sparse:
            # ==== node embedders ====
            if task in ["TSP", "CVRP"]:
                # coords: (B,N,2)
                self.node_embedder_row = nn.Sequential(
                    PositionEmbeddingSine(hidden_dim // 2),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.node_embedder_col = nn.Sequential(
                    PositionEmbeddingSine(hidden_dim // 2),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:  # ATSP 或其它
                self.node_embedder_row = nn.Sequential(
                    ScalarEmbeddingSine2D(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.node_embedder_col = nn.Sequential(
                    ScalarEmbeddingSine2D(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            # ==== edge embedder ====
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine3D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # ==== encoder layers ====
            self.layers = nn.ModuleList(
                [DenseEdgeAwareEncoderLayer(task, **model_params) for _ in range(num_att_layers)]
            )
            self.gnn_block = GNNDenseBlock(**model_params)

            # ==== edge classifier: (B,N,N,d) -> (B,2,N,N) ====
            if paradigm in ["GP", "AE"]:
                if self.task in ["TSP", "ATSP", "CVRP"]:
                    self.edge_classifier = nn.Sequential(
                        GroupNorm32(32, hidden_dim),
                        nn.Conv2d(hidden_dim, out_channels=2, kernel_size=1, bias=True),
                    )
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

        else:  # sparse
            if self.task in ["TSP", "CVRP"]:
                self.node_embedder_row = nn.Sequential(
                    PositionEmbeddingSine(hidden_dim // 2),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                if self.dual:
                    self.node_embedder_col = nn.Sequential(
                        PositionEmbeddingSine(hidden_dim // 2),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
            else: # node-decision tasks
                self.node_embedder_row = nn.Sequential(
                    ScalarEmbeddingSine1D(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                if self.dual:
                    self.node_embedder_col = nn.Sequential(
                        ScalarEmbeddingSine1D(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine1D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers = nn.ModuleList(
                [SparseEdgeAwareEncoderLayer(task, **model_params) for _ in range(num_att_layers)]
            )
            self.gnn_block = GNNSparseBlock(**model_params)

            # ==== edge classifier: (B,N,N,d) -> (B,2,N,N) ====
            if paradigm in ["GP", "AE"]:
                if task in ["TSP", "ATSP", "CVRP"]:
                    self.edge_classifier = nn.Sequential(
                        GroupNorm32(32, hidden_dim),
                        nn.Conv2d(hidden_dim, out_channels=2, kernel_size=1, bias=True),
                    )
                else:
                    if self.dual:
                        self.merge_node = nn.Sequential(
                            nn.Linear(2 * hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        )
                    self.node_classifier = nn.Sequential(
                        GroupNorm32(32, hidden_dim),
                        nn.Conv2d(hidden_dim, out_channels=2, kernel_size=1, bias=True),
                    )
            else:
                raise NotImplementedError()

    def forward(
        self, task: str, focus_on_node: bool, focus_on_edge: bool, 
        node_feat: Tensor, x: Tensor, edge_feat: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor, edge_index: Tensor
    ) -> Tensor:
        # ==== initial node embeddings ====
        if self.task not in ["TSP", "CVRP"]:
            if not self.sparse: # ATSP
                assert self.task == "ATSP", "Node-oriented tasks support sparse modeling only."
                B, N, _ = edge_feat.shape
                row_idx = torch.randperm(N, device=edge_feat.device).unsqueeze(0).expand(B, -1) # (B,N)
                col_idx = torch.randperm(N, device=edge_feat.device).unsqueeze(0).expand(B, -1) # (B,N)
            else: # MIS, MCl, etc.
                row_idx = x # decision variables as node identifier
                if self.dual: col_idx = x
            row_emb: Tensor = self.node_embedder_row(row_idx)
            if self.dual: col_emb: Tensor = self.node_embedder_col(col_idx)
        else: # TSP, node_feat is not None
            row_emb: Tensor = self.node_embedder_row(node_feat)  # (B,N,d)
            if self.dual: col_emb: Tensor = self.node_embedder_col(node_feat)  # 对称任务：row/col 共享坐标

        # ==== initial edge embeddings ====
        if e is not None:
            edge_feat = edge_feat + e
        edge_emb: Tensor = self.edge_embed(edge_feat)       # (B,N,N,d)

        # ==== initial time/mask embeddings ====
        # e = self.edge_embed(e) # (E, H)
        mask = self.mask_embed(mask.long()) # (E, H)
        if self.task in ["TSP", "ATSP", "CVRP"]:
            edge_emb = edge_emb + mask # (V, H)
        elif self.task in ["MIS", "MCl", "MVC", "MCut"]:
            row_emb = row_emb + mask
            if self.dual: col_emb = col_emb + mask
        t = self.time_embedder(timestep_embedding(t, self.hidden_dim)) # (H,)

        # ==== stacked encoder layers ====
        for att_layer, node_time_layer, edge_time_layer in zip(self.layers, self.node_time_layers, self.edge_time_layers):
            if self.task in ["TSP", "CVRP"]:
                if not self.dual:
                    _, edge_emb = self.gnn_block(row_emb, None, edge_emb, edge_index)
                else:
                    _, _, edge_emb = self.gnn_block(row_emb, col_emb, edge_emb, edge_index)
            elif self.task in ["ATSP"]:
                row_emb, col_emb, edge_emb = self.gnn_block(row_emb, col_emb, edge_emb, edge_index)
            elif self.task in ["MIS", "MCl", "MVC", "MCut"]:
                if not self.dual:
                    row_emb, edge_emb = self.gnn_block(row_emb, None, edge_emb, edge_index)
                else:
                    row_emb, col_emb, edge_emb = self.gnn_block(row_emb, col_emb, edge_emb, edge_index)

            if self.dual:
                row_emb, col_emb = att_layer(row_emb, col_emb, edge_emb, edge_index)
            else:
                row_emb = att_layer(row_emb, row_emb, edge_emb, edge_index)

            if focus_on_node and t is not None:
                time_embed = node_time_layer(t)
                row_emb = row_emb + time_embed
                if self.dual:
                   col_emb = col_emb + time_embed
            if focus_on_edge and t is not None:
                time_embed = edge_time_layer(t)
                edge_emb = edge_emb + edge_time_layer(t)

        # ==== classifier ====
        if self.paradigm in ["GP", "AE"]:
            if self.task in ["TSP", "ATSP", "CVRP"]:
                if not self.sparse:
                    logits = self.edge_classifier(edge_emb.permute(0, 3, 1, 2))   # edge_emb: (B,N,N,d) -> (B,2,N,N)
                else:
                    N, d = row_emb.shape
                    E = edge_emb.shape[0]
                    edge_emb = edge_emb.reshape(1, N, -1, d).permute((0, 3, 1, 2))
                    edge_out: Tensor = self.edge_classifier(edge_emb)
                    logits = edge_out.reshape(-1, E).permute((1, 0))
            elif self.task in ["MIS", "MCl", "MVC", "MCut"]:
                assert self.sparse, "Node-oriented tasks support sparse modeling only."
                if self.dual:
                    node_embed = self.merge_node(torch.cat([row_emb, col_emb], dim=-1))
                else:
                    node_embed = row_emb
                N, d = node_embed.shape
                node_embed = node_embed.reshape(1, N, -1, d).permute((0, 3, 1, 2))
                node_out: Tensor = self.node_classifier(node_embed)
                logits = node_out.reshape(-1, N).permute((1, 0))
        else:
            raise NotImplementedError()
        
        return logits