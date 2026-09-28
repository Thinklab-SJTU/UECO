import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from model import UECOEncoder
from CVRPEnv import *


class UECOLCModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        # Unified encoder backbone (plug-and-play across paradigms/objectives)
        self.encoder: UECOEncoder = UECOEncoder(
            task="CVRP",
            paradigm="LC",
            sparse=False,
            **model_params
        )
        self.task = self.encoder.task

        # Decoder follows the original CVRP implementation
        self.decoder = RL4COCVRPDecoder(**model_params)

        self.encoded_nodes = None  # (batch, problem+1, embedding)
        self.encoded_row = None
        self.encoded_col = None

    def pre_forward(self, reset_state: Reset_State) -> None:
        depots = reset_state.depots
        # depots.shape: (batch, 1, 2)
        points = reset_state.points
        # points.shape: (batch, problem, 2)
        demands = reset_state.demands
        # demands.shape: (batch, problem)

        batch_size = depots.size(0)
        device = depots.device
        dtype = depots.dtype

        # Node features: [x, y, demand], with depot demand = 0
        depot_demand = torch.zeros(size=(batch_size, 1, 1), device=device, dtype=dtype)
        depot_feat = torch.cat((depots, depot_demand), dim=2)
        # shape: (batch, 1, 3)
        # is_depot = torch.zeros(size=(batch_size, points.shape[1] + 1, 1), device=device, dtype=torch.long)
        # is_depot[:, 0, :] = 1

        node_feat = torch.cat((points, demands[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        node_feat = torch.cat((depot_feat, node_feat), dim=1)
        # shape: (batch, problem+1, 3)
        # node_feat = torch.cat((node_feat, is_depot), dim=2)
        # shape: (batch, problem+1, 4)

        # Edge features: pairwise distances on coordinates (including depot)
        coords = torch.cat((depots, points), dim=1)
        # shape: (batch, problem+1, 2)
        edge_feat = torch.cdist(coords, coords, p=2)
        # shape: (batch, problem+1, problem+1)

        self.encoded_row, self.encoded_col = self.encoder.forward(
            node_feat=node_feat,
            edge_feat=edge_feat,
            edge_index=None
        ) # encoded_nodes.shape: (batch, node, embedding)
        self.decoder.set_kv(self.encoded_col)

    def forward(
        self,
        state: Step_State,
    ):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, always depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
        elif state.selected_count == 1:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_row = _get_encoding(self.encoded_row, selected) # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_row)
        else:
            # RL / sampling
            encoded_last_node = _get_encoding(self.encoded_row, state.current_node)
            # shape: (batch, pomo, embedding)

            probs = self.decoder.forward(encoded_last_node, state.capacity, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params.get("eval_type", "argmax") == "softmax":
                # selection by softmax sampling
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob

########################################
# DECODER
########################################

class RL4COCVRPDecoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        num_heads = self.model_params["num_heads"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq_last = nn.Linear(embedding_dim + 1, num_heads * qkv_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes: Tensor):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        num_heads = self.model_params["num_heads"]
        self.k = reshape_by_heads(self.Wk(encoded_nodes), num_heads=num_heads)
        # shape: (batch, num_heads, problem+1, qkv_dim)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), num_heads=num_heads)
        # shape: (batch, num_heads, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1: Tensor) -> None:
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        num_heads = self.model_params["num_heads"]
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), num_heads=num_heads) # shape: (batch, num_heads, n, qkv_dim)

    def forward(self, encoded_last_node, capacity, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # capacity.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        num_heads = self.model_params["num_heads"]
        embedding_dim = self.model_params["embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, capacity[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), num_heads=num_heads)
        # shape: (batch, num_heads, pomo, qkv_dim)

        q = q_last + self.q1
        # shape: (batch, num_heads, pomo, qkv_dim)

        out_concat = self._multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, num_heads*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem+1)
        sqrt_embedding_dim = math.sqrt(embedding_dim)
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem+1)

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        # shape: (batch, pomo, problem+1)
        score_masked = score_clipped + ninf_mask
        # shape: (batch, pomo, problem+1)
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem+1)

        return probs

    def _multi_head_attention(
            self,
            q: Tensor, 
            k: Tensor,
            v: Tensor,
            rank2_ninf_mask: Union[Tensor, None]=None, 
            rank3_ninf_mask: Union[Tensor, None]=None
        ) -> Tensor:
        # q shape: (batch, num_heads, n, qkv_dim)
        # k,v shape: (batch, num_heads, problem, qkv_dim)
        # rank2_ninf_mask.shape: (batch, problem)
        # rank3_ninf_mask.shape: (batch, n, problem)

        batch_s = q.size(0)
        num_heads = q.size(1)
        n = q.size(2)
        key_dim = q.size(3)

        input_s = k.size(2)

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, num_heads, n, problem)

        score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, num_heads, n, input_s)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, num_heads, n, input_s)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, num_heads, n, problem)

        out = torch.matmul(weights, v)
        # shape: (batch, num_heads, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, num_heads, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, num_heads * key_dim)
        # shape: (batch, n, num_heads*key_dim)

        return out_concat

########################################
# Attention Utils
########################################

def reshape_by_heads(qkv, num_heads):
    # q.shape: (batch, n, num_heads*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, num_heads, -1)
    # shape: (batch, n, num_heads, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, num_heads, n, key_dim)

    return q_transposed

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes
