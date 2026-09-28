import os
import sys
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
sys.path.append(root_folder)
from env import GNN4COEnv
from model import UECOModel, UECOEncoder, TSPDecoder
from solver import GNN4COTSPSolver
from tabulate import tabulate

###############################
#          Settings           #
###############################

# Problem Settings
TASK = "TSP"
NODES_NUM = 50
SPARSE_FACTOR = 50
TEST_TYPE = (TASK, "uniform", NODES_NUM)

# Solving Settings
DEVICE = "cuda:0"
SOLVING_SETTINGS = ("greedy", "2opt") # Choices: [("greedy", "2opt"), ("greedy", "mcts"), ("random_mcts", None), ("mcts", None)]

# Local Search Settings
MCTS_TIME_LIMIT = 0.05 # 0.05 for TSP50 and TSP100; 1 for TSP500 and TSP1K; 5 for TSP10K
MCTS_TYPE_FLAG = 2 # 2 for TSP50 and TSP100; 1 for TSP500, TSP1K and TSP10K, 1 for random_mcts

# Model Settings
encoder_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 3,
    "num_att_layers": 5,
    "num_heads": 16,
    "qkv_dim": 16,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "dual": True
}

###############################
#            Utils            #
###############################

def check_and_ckpt_to_pt(ckpt_path: str, save_path: str):
    if ckpt_path.split(".")[-1] == "pt":
        return ckpt_path
    import torch
    weight = torch.load(ckpt_path, map_location=DEVICE)["state_dict"]
    torch.save(weight, save_path)
    print(f"[TEST INFO] Saving '{ckpt_path}' to '{save_path}' done!")
    return save_path

def print_results(obj: float, ref_obj: float, gap: float, gap_std: float, avg_time: float):
    table = [[obj, ref_obj, f"{gap:.5f}%", gap_std, avg_time]]
    headers = ["obj", "ref_obj", "gap", "gap_std", "time (s/inst.)"]
    print("[TEST INFO] Test result:")
    print(tabulate(table, headers=headers, floatfmt=".5f"))

# Test File & Pretrained Weights
DATA_ROOT = os.environ.get("UECO_DATA_ROOT", os.path.join(os.path.dirname(root_folder), "data"))
TEST_FILE_DICT = {
    ("TSP", "uniform", 50): os.path.join(DATA_ROOT, "TSP", "tsp50_concorde_5.68759.txt"),
    ("TSP", "uniform", 100): os.path.join(DATA_ROOT, "TSP", "tsp100_concorde_7.75585.txt"),
    ("TSP", "uniform", 500): os.path.join(DATA_ROOT, "TSP", "tsp500_concorde_16.54581.txt"),
}

WEIGHT_PATH_DICT = {
    50: os.path.join(repo_folder, "weights", "TSP50_GP_UECO.pt"),
    100: os.path.join(repo_folder, "weights", "TSP100_GP_UECO.pt"),
    500: os.path.join(repo_folder, "weights", "TSP500_GP_UECO.pt"),
}


################################
#             Main             #
################################

if __name__ == "__main__":
    solver = GNN4COTSPSolver(
        model=UECOModel(
            env=GNN4COEnv(
                task=TASK, sparse_factor=SPARSE_FACTOR, device=DEVICE
            ),
            encoder=UECOEncoder(
                task=TASK,
                paradigm="GP",
                sparse=SPARSE_FACTOR>0,
                **encoder_params
            ),
            decoder=TSPDecoder(
                decoding_type=SOLVING_SETTINGS[0],
                local_search_type=SOLVING_SETTINGS[1],
                mcts_time_limit=MCTS_TIME_LIMIT,
                mcts_type_2opt=MCTS_TYPE_FLAG
            ),
            weight_path=check_and_ckpt_to_pt(
                WEIGHT_PATH_DICT[NODES_NUM],
                save_path=os.path.join(repo_folder, "weights", f"tmp_{TASK}{NODES_NUM}_GP_UECO.pt")
            )
        )
    )
    print("[TEST INFO] Test started ...")
    print(f"[TEST INFO] Test data: {TEST_TYPE}")
    print(f"[TEST INFO] Test setting: {SOLVING_SETTINGS}")
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print_results(*solver.evaluate(calculate_gap=True), avg_time=solver.avg_time)
