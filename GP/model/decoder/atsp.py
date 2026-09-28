import numpy as np
from torch import Tensor
from ml4co_kit import to_numpy, atsp_greedy_decoder, np_sparse_to_dense
from model.decoder.base import GNN4CODecoder


class ATSPDecoder(GNN4CODecoder):
    def __init__(
        self,
        decoding_type: str = "greedy", 
        local_search_type: str = None
    ) -> None:
        super(ATSPDecoder, self).__init__()
        self.decoding_type = decoding_type
        self.local_search_type = local_search_type
        
    def _edge_dense_decode(
        self, heatmap: Tensor, x: Tensor, graph: Tensor
    ) -> np.ndarray:
        # tensor -> numpy array
        heatmap = to_numpy(heatmap)
        
        # decoding
        if self.decoding_type == "greedy":
            sol = atsp_greedy_decoder(-heatmap)
        else:
            raise NotImplementedError()
        
        if self.local_search_type is not None:
            raise NotImplementedError()
        
        # return
        return sol
    
    def _edge_sparse_decode(
        self, heatmap: Tensor, x: Tensor, graph: Tensor,
        edge_index: Tensor, nodes_num: int
    ) -> np.ndarray:
        # tensor -> numpy array
        heatmap = to_numpy(heatmap)
        edge_index = to_numpy(edge_index)

        # heatmap: sparse -> dense
        heatmap = np_sparse_to_dense(
            nodes_num=nodes_num, edge_index=edge_index, edge_attr=heatmap
        )
        
        # decoding
        if self.decoding_type == "greedy":
            sol = atsp_greedy_decoder(-heatmap)
        else:
            raise NotImplementedError()
        
        if self.local_search_type is not None:
            raise NotImplementedError()
        
        # return
        return sol
