import torch
import torch.nn.functional as F
from torch_scatter import scatter_softmax
from typing import Any
from torch import Tensor, nn
from ml4co_kit import BaseModel
from typing import Union, Tuple, List
from env.env import GNN4COEnv
from model.decoder.base import GNN4CODecoder
from model.encoder.ueco_encoder import UECOEncoder


class UECOModel(BaseModel):
    def __init__(
        self,
        env: GNN4COEnv,
        encoder: UECOEncoder,
        decoder: GNN4CODecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 5e-5,
        weight_path: str = None,
        load_strict: bool = True
    ):
        super(UECOModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: GNN4COEnv
        self.model: UECOEncoder
        self.decoder: GNN4CODecoder = decoder
        
        # load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=load_strict)
        self.to(self.env.device)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # set mode
        self.env.mode = phase
        
        # get real data
        """
        task: ATSP or CVRP or MCl or MCut or MIS or MVC or TSP
        if sparse:
            [0] task
            [1] x: (V, C) or (V,) , nodes feature
            [2] e: (E, D) or (E,) , edges feature
            [3] edge_index: (2, E)
            [4] graph_list: graph data
            [5] ground_truth: (E,) or (V,)
            [6] nodes_num_list
            [7] edges_num_list
        else:
            [0] task
            [1] x: (B, V, C) or (B, V), nodes_feature
            [2] graph: (B, V, V)
            [3] ground_truth: (B, V, V) or (B, V)
            [4] nodes_num_list
        """
        if phase == "train":
            # get real train batch data
            batch_size = len(batch)
            batch_data = self.env.generate_train_data(batch_size)
            task = batch_data[0]
            
            # deal with different task
            if task in ["TSP", "ATSP", "CVRP"]:
                if self.env.sparse:
                    loss = self.train_edge_sparse_process(*batch_data)
                else:
                    loss = self.train_edge_dense_process(*batch_data)
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                if self.env.sparse:
                    loss = self.train_node_sparse_process(*batch_data)
                else:
                    loss = self.train_node_dense_process(*batch_data)
            else:
                raise NotImplementedError()
            
        elif phase == "val":
            # get val data
            batch_data = self.env.generate_val_data(batch_idx)
            task = batch_data[0]
            
            # deal with different task
            if task in ["TSP", "ATSP", "CVRP"]:
                if self.env.sparse:
                    loss, heatmap = self.inference_edge_sparse_process(*batch_data)
                else:
                    loss, heatmap = self.inference_edge_dense_process(*batch_data)
                    
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                if self.env.sparse:
                    loss, heatmap = self.inference_node_sparse_process(*batch_data)
                else:
                    loss, heatmap = self.inference_node_dense_process(*batch_data)
            else:
                raise NotImplementedError()
            
            # decoding
            if self.env.sparse:
                costs_avg = self.decoder.sparse_decode(heatmap, *batch_data, return_cost=True)
            else:
                costs_avg = self.decoder.dense_decode(heatmap, *batch_data, return_cost=True)
        else:
            raise NotImplementedError()
     
        # log
        metrics = {f"{phase}/loss": loss}
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg})
        for k, v in metrics.items():
            formatted_v = f"{v:.8f}"
            self.log(k, float(formatted_v), prog_bar=True, on_epoch=True, sync_dist=True)
        
        # return
        return loss if phase == "train" else metrics   
        
    def train_edge_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Tensor:
        # forward
        e_pred = self.model.forward(
            node_feat=x,
            edge_feat=e,
            edge_index=edge_index
        )  # (E,)

        loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
        return loss
   
    def train_edge_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Tensor:
        e_pred = self.model.forward(
            # task=task, 
            node_feat=x,
            edge_feat=graph,
            edge_index=None
        )
        # loss = edge_dense_bce_loss(
        #     task=task, 
        #     logits=e_pred, 
        #     target=ground_truth,
        #     mask_diagonal=False
        # )
        loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
        return loss
    
    def train_node_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Tensor:
        # raise NotImplementedError()
        x_pred: Tensor = self.model.forward(
            # task=task, 
            node_feat=x, 
            edge_feat=e, 
            edge_index=edge_index
        )
        
        # return
        loss = nn.CrossEntropyLoss()(x_pred, ground_truth)
        return loss

    def train_node_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Tensor:
        raise NotImplementedError("Node-oriented tasks must be trained with sparse graphs.")

    def inference_edge_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        # ===== forward =====
        e_pred = self.model.forward(
            node_feat=x,
            edge_feat=e,
            edge_index=edge_index
        )  # (E,)

        # ===== heatmap =====
        e_pred_softmax = e_pred.softmax(dim=-1)
        e_heatmap = e_pred_softmax[:, 1]

        # ===== validation loss =====
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_edge_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        e_pred: Tensor = self.model.forward(
            # task=task, 
            node_feat=x, 
            edge_feat=graph, 
            edge_index=None
        )
        # heatmap
        # B, N, _, _ = e_pred.shape
        # e_heatmap = e_pred
        # e_heatmap[:, torch.arange(N), torch.arange(N)] = 0
        # e_heatmap = e_pred.softmax(dim=2)

        # e_heatmap = e_pred.clone()
        # if task == "ATSP":
        #     e_heatmap[:, torch.arange(N), torch.arange(N)] = -1e9
        #     e_heatmap = e_heatmap.softmax(dim=2)
        # elif task == "TSP":
        #     e_heatmap = e_heatmap.sigmoid()
        #     e_heatmap[:, torch.arange(N), torch.arange(N)] = 0

        e_pred_softmax = e_pred.softmax(dim=1)
        e_heatmap = e_pred_softmax[:, 1, :, :]

        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, ground_truth)

            # loss = edge_dense_bce_loss(task, e_pred, ground_truth, mask_diagonal=False)

            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_node_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred: Tensor = self.model.forward(
            # task=task, 
            node_feat=x, 
            edge_feat=e, 
            edge_index=edge_index
        )
        
        # heatmap
        x_pred_softmax = x_pred.softmax(-1)
        x_heatmap = x_pred_softmax[:, 1]

        # import pdb; pdb.set_trace()
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(x_pred, ground_truth)
            return loss, x_heatmap
        elif self.env.mode == "solve":
            return x_heatmap
        else:
            raise ValueError()
        
    def inference_node_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError()


def edge_dense_bce_loss(
    task: str,
    logits: torch.Tensor,      # (B, N, N), raw logits
    target: torch.Tensor,      # (B, N, N), {0,1}
    mask_diagonal: bool = True
):
    """
    BCE loss on edges, ignoring masked entries.
    """
    # element-wise BCE (no reduction)
    # shape: (B, N, N)
    
    B, N, _ = logits.shape
    if task == "TSP":
        w_pos = torch.tensor(N / 2 - 1)
    else:
        w_pos = torch.tensor(N - 1)

    loss = F.binary_cross_entropy_with_logits(
        logits, target.float(), reduction='none',
        pos_weight=w_pos
    )
    if mask_diagonal:
        diag_mask = ~torch.eye(N, device=logits.device, dtype=torch.bool)
        # shape: (N, N), True on off-diagonal
        diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1)
    else:
        diag_mask = torch.ones_like(logits)

    loss = loss * diag_mask.float()
    denom = diag_mask.float().sum()

    # normalized loss
    return loss.sum() / denom


def edge_sparse_bce_loss(
    task: str,
    logits: torch.Tensor,      # (E,)
    target: torch.Tensor,      # (E,) \in {0,1}^|E|
):
    """
    Sparse BCE loss on edges.

    logits: (E,)
    target: (E,)
    nodes_num_list: 每个 graph 的节点数
    edges_num_list: 每个 graph 的边数（k-NN 后）
    """

    assert logits.dim() == 1
    assert logits.shape == target.shape

    with torch.no_grad():
        pos = target.sum()
        neg = target.numel() - pos
        pos_weight = neg / (pos + 1e-8)

    pos_weight = pos_weight.to(logits.device)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        reduction="mean",
        pos_weight=pos_weight
    )

    return loss


