import os
import sys
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
os.chdir(root_folder)
sys.path.append(root_folder)
from env import GNN4COEnv
from model import UECOModel, UECOEncoder, MCutDecoder
from solver import GNN4COMCutSolver
from tabulate import tabulate

###############################
#          Settings           #
###############################

# Problem Settings
TASK = "MCut"
NODES_NUM = "200-300"
TEST_TYPE = (TASK, "BA", NODES_NUM)

# Solving Settings
DEVICE = "cuda:0"
SOLVING_SETTINGS = ("greedy", None) # ("greedy", "rlsa") / ("beam", None) / ("beam", "rlsa")

# Model Settings
encoder_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 3,
    "num_att_layers": 4,
    "num_heads": 16,
    "qkv_dim": 16,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "dual": False
}

# Local Search Settings
RLSA_SETTINGS_DICT = {
    ("MCut", "BA", "200-300"): (1, 20, 1000, 1000),
    ("MCut", "BA", "800-1200"): (1, 50, 1000, 1000),
}

###############################
#            Utils            #
###############################

def check_and_ckpt_to_pt(ckpt_path: str, save_path: str):
    if ckpt_path.split(".")[-1] == "pt":
        return ckpt_path
    import torch
    weight = torch.load(ckpt_path)["state_dict"]
    torch.save(weight, save_path)
    print(f"[TEST INFO] Saving '{ckpt_path}' to '{save_path}' done!")
    return save_path

def print_results(obj: float, ref_obj: float, gap: float, gap_std: float, avg_time: float):
    table = [[obj, ref_obj, f"{gap:.5f}%", gap_std, avg_time]]
    headers = ["obj", "ref_obj", "gap", "gap_std", "time (s/inst.)"]
    print("[TEST INFO] Test result:")
    print(tabulate(table, headers=headers, floatfmt=".5f"))

# Test File & Pretrained Weights
TEST_FILE_DICT = {
    ("MCut", "BA", "200-300"): "../data/test/MCut/mcut_ba-small_gurobi_60s_727.844.txt",
    ("MCut", "BA", "800-1200"): "../data/test/MCut/mcut_ba-large_gurobi_300s_2936.886.txt",
}

WEIGHT_PATH_DICT = {
    ("MCut", "BA", "200-300"): "../weights/GP/mcut/MCut_BA_200-300_GP_UECO.pt",
    ("MCut", "BA", "800-1200"): "../weights/GP/mcut/MCut_BA_800-1200_GP_UECO.pt",
}

################################
#             Main             #
################################

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = GNN4COMCutSolver(
        model=UECOModel(
            env=GNN4COEnv(
                task="MCut", mode="solve", sparse_factor=1, device=DEVICE,
            ),
            encoder=UECOEncoder(
                task=TASK,
                paradigm="GP",
                sparse=True,
                **encoder_params
            ),
            decoder=MCutDecoder(
                decoding_type=SOLVING_SETTINGS[0], 
                local_search_type=SOLVING_SETTINGS[1],
                rlsa_tau=rlsa_settings[0],
                rlsa_d=rlsa_settings[1],
                rlsa_k=rlsa_settings[2],
                rlsa_t=rlsa_settings[3]
            ),
            weight_path=check_and_ckpt_to_pt(
                WEIGHT_PATH_DICT[TEST_TYPE],
                save_path=f"../weights/GP/mcut/tmp_{TEST_TYPE}_GP_UECO.pt"
            )
        )
    )
    print(f"[TEST INFO] Test started on {DEVICE}")
    print(f"[TEST INFO] Test data: {TEST_TYPE}")
    print(f"[TEST INFO] Test setting: {SOLVING_SETTINGS}")
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print_results(*solver.evaluate(calculate_gap=True), avg_time=solver.avg_time)
