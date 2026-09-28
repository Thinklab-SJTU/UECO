
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from model import UECOEncoder
from TSPEnv import *

class UECOLCModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder: UECOEncoder = UECOEncoder(
            task="TSP",
            paradigm="LC",
            sparse=False,
            **model_params
        )
        self.task = self.encoder.task
        self.decoder = RL4COTSPDecoder(**model_params)
        self.encoded_row = None # shape: (batch, node, embedding)
        self.encoded_col = None # shape: (batch, node, embedding)

    def pre_forward(self, reset_state: Reset_State) -> None:
        edge_feat = reset_state.problems # problems.shape: (batch, node, node)
        node_feat = reset_state.coordinates # shape: (batch, node, 2)
        self.encoded_row, self.encoded_col = self.encoder.forward(
            node_feat=node_feat, 
            edge_feat=edge_feat,
            edge_index=None
        ) # encoded_nodes.shape: (batch, node, embedding)
        self.decoder.set_kv(self.encoded_col)

    def forward(self, state: Step_State) -> Tuple[Tensor, Tensor]:
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_row = _get_encoding(self.encoded_row, selected) # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_row)
        else:
            encoded_current_row = _get_encoding(self.encoded_row, state.current_node) # shape: (batch, pomo, embedding)
            all_job_probs = self.decoder.forward(encoded_current_row, ninf_mask=state.ninf_mask) # shape: (batch, pomo, job)
            if self.training or self.model_params["eval_type"] == "softmax":
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = all_job_probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size) # shape: (batch, pomo)
                    prob = all_job_probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size) # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = all_job_probs.argmax(dim=2) # shape: (batch, pomo)
                prob = None
        return selected, prob

########################################
# Task-specific Decoder
########################################

class RL4COTSPDecoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        num_heads = self.model_params["num_heads"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq_0 = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_jobs: Tensor) -> None:
        # encoded_jobs.shape: (batch, job, embedding)
        num_heads = self.model_params["num_heads"]
        self.k = reshape_by_heads(self.Wk(encoded_jobs), num_heads=num_heads)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), num_heads=num_heads) # shape: (batch, num_heads, job, qkv_dim)
        self.single_head_key = encoded_jobs.transpose(1, 2) # shape: (batch, embedding, job)

    def set_q1(self, encoded_q1: Tensor) -> None:
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        num_heads = self.model_params["num_heads"]
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), num_heads=num_heads) # shape: (batch, num_heads, n, qkv_dim)

    def forward(self, encoded_q0: Tensor, ninf_mask: Tensor) -> Tensor:
        # encoded_q4.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job)
        num_heads = self.model_params["num_heads"]
        embedding_dim = self.model_params["embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]

        #  Multi-Head Attention
        #######################################################
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), num_heads=num_heads) # shape: (batch, num_heads, pomo, qkv_dim)
        q = self.q1 + q0 # shape: (batch, num_heads, pomo, qkv_dim)
        out_concat = self._multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask) # shape: (batch, pomo, num_heads*qkv_dim)
        mh_atten_out = self.multi_head_combine(out_concat) # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key) # shape: (batch, pomo, job)
        sqrt_embedding_dim = math.sqrt(embedding_dim)
        score_scaled = score / sqrt_embedding_dim # shape: (batch, pomo, job)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2) # shape: (batch, pomo, job)

        return probs

    def _multi_head_attention(
            self,
            q: Tensor, 
            k: Tensor,
            v: Tensor,
            rank2_ninf_mask: Union[Tensor, None]=None, 
            rank3_ninf_mask: Union[Tensor, None]=None
        ) -> Tensor:
        # q shape: (batch, num_heads, n, key_dim)   : n can be either 1 or pomo
        # k,v shape: (batch, num_heads, node, key_dim)
        # rank2_ninf_mask.shape: (batch, node)
        # rank3_ninf_mask.shape: (batch, group, node)
        batch_s = q.size(0)
        n = q.size(2)
        node_cnt = k.size(2)
        num_heads = self.model_params["num_heads"]
        qkv_dim = self.model_params["qkv_dim"]
        sqrt_qkv_dim = math.sqrt(qkv_dim)

        score = torch.matmul(q, k.transpose(2, 3)) # shape: (batch, num_heads, n, node)
        score_scaled = score / sqrt_qkv_dim
        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, num_heads, n, node_cnt)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, num_heads, n, node_cnt)
        weights = nn.Softmax(dim=3)(score_scaled) # shape: (batch, num_heads, n, node)
        out = torch.matmul(weights, v) # shape: (batch, num_heads, n, key_dim)
        out_transposed = out.transpose(1, 2) # shape: (batch, n, num_heads, key_dim)
        out_concat = out_transposed.reshape(batch_s, n, num_heads * qkv_dim) # shape: (batch, n, num_heads*key_dim)

        return out_concat


########################################
# NN SUB FUNCTIONS
########################################

def reshape_by_heads(qkv: Tensor, num_heads: int) -> Tensor:
    # q.shape: (batch, n, num_heads*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, num_heads, -1) # shape: (batch, n, num_heads, key_dim)
    q_transposed = q_reshaped.transpose(1, 2) # shape: (batch, num_heads, n, key_dim)
    return q_transposed

def _get_encoding(encoded_nodes: Tensor, node_index_to_pick: Tensor) -> Tensor:
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim) # shape: (batch, pomo, embedding)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index) # shape: (batch, pomo, embedding)
    return picked_nodes
