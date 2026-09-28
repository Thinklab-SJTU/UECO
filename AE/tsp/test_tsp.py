import os
import sys
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
sys.path.append(root_folder)
from model import UECOModel, UECOEncoder, COExpanderDecoder
from env import COExpanderEnv
from solver import COExpanderTSPSolver
from tabulate import tabulate


###############################
#          Settings           #
###############################

# Problem Settings
TASK = "TSP"
NODES_NUM = 50
SPARSE_FACTOR = 50 # 50 for TSP-500 and TSP-10K, 100 for TSP-1K
TEST_TYPE = (TASK, "uniform", NODES_NUM)

# Solving Settings
DEVICE = "cuda:0"
INFERENCE_STEP = 5 # Is
DETERMINATE_STEP = 3 # Ds
SAMPLING_NUM = 1 # S

# Local Search Settings
USE_2OPT = True

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
    table = [[obj, ref_obj, gap, gap_std, avg_time]]
    headers = ["obj", "ref_obj", "gap (%)", "gap_std", "time (s/inst.)"]
    print("[TEST INFO] Test result:")
    print(tabulate(table, headers=headers, floatfmt=".5f"))

# Test File & Pretrained Weights
TEST_FILE_DICT = {
    ("TSP", "uniform", 50): "../../data/test/TSP/tsp50_concorde_5.68759.txt",
    ("TSP", "uniform", 100): "../../data/test/TSP/tsp100_concorde_7.75585.txt",
    ("TSP", "uniform", 500): "../../data/test/TSP/tsp500_concorde_16.54581.txt",
}

WEIGHT_PATH_DICT = {
    50: "../../weights/AE/tsp/TSP50_AE_UECO.pt",
    100: "../../weights/AE/tsp/TSP100_AE_UECO.pt",
    500: "../../weights/AE/tsp/TSP500_AE_UECO.pt",
}

################################
#             Main             #
################################

# Main
if __name__ == "__main__":
    solver = COExpanderTSPSolver(
        model=UECOModel(
            env=COExpanderEnv(
                task="TSP", sparse_factor=SPARSE_FACTOR, device=DEVICE
            ),
            encoder=UECOEncoder(
                task="TSP",
                paradigm="AE",
                sparse=SPARSE_FACTOR>0,
                sparse_factor=SPARSE_FACTOR,
                **encoder_params
            ),
            decoder=COExpanderDecoder(
                decode_kwargs={"use_2opt": USE_2OPT}
            ),
            weight_path=check_and_ckpt_to_pt(
                WEIGHT_PATH_DICT[NODES_NUM],
                save_path=f"../../weights/AE/tsp/tmp_{TASK}{NODES_NUM}_AE_UECO.pt"
            ),
            load_strict=False,
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    print("[TEST INFO] Test started ...")
    print(f"[TEST INFO] Test data: {TEST_TYPE}")
    print(f"[TEST INFO] Test setting: Greedy {'+ 2OPT' if USE_2OPT else ''}")
    print(f"[TEST INFO] Test setting: Is={INFERENCE_STEP}, Ds={DETERMINATE_STEP}, S={SAMPLING_NUM}")
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print_results(*solver.evaluate(calculate_gap=True), avg_time=solver.avg_time)
