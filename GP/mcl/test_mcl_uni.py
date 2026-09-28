import os
import sys
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
sys.path.append(root_folder)
from env import GNN4COEnv
from model import UECOModel, UECOEncoder, MClDecoder
from solver import GNN4COMClSolver
from tabulate import tabulate


###############################
#          Settings           #
###############################

# Problem Settings
TASK = "MCl"
NODES_NUM = "200-300"
TEST_TYPE = (TASK, "RB", NODES_NUM)

# Solving Settings
DEVICE = "cuda"
SOLVING_SETTINGS = ("greedy", None) # ("greedy", "rlsa") / ("beam", None) / ("beam", "rlsa")

# Model Settings
encoder_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 3,
    "num_att_layers": 4, # 4 for RB-small
    "num_heads": 16,
    "qkv_dim": 16,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "dual": False
}

# Local Search Settings
RLSA_SETTINGS_DICT = {
    ("MCl", "RB", "200-300"): (0.01, 2, 200, 50, 1.02, 0.3),
    ("MCl", "RB", "800-1200"): (0.01, 2, 200, 200, 1.02, 0.3),
    ("MCl", "RB", "2000-3000"): (0.01, 2, 1000, 1000, 1.02, 0.3),
    ("MCl", "TWITTER", None): (0.01, 2, 200, 50, 4.0, 0.3),
    ("MCl", "COLLAB", None): (0.01, 2, 200, 20, 1.001, 0.3)
}

# Test File & Pretrained Settings
DATA_ROOT = os.environ.get("UECO_DATA_ROOT", os.path.join(os.path.dirname(root_folder), "data"))
TEST_FILE_DICT = {
    ("MCl", "RB", "200-300"): os.path.join(DATA_ROOT, "MCl", "mcl_rb-small_gurobi_19.082.txt"),
    ("MCl", "RB", "800-1200"): os.path.join(DATA_ROOT, "MCl", "mcl_rb-large_gurobi-300s_40.182.txt"),
}

WEIGHT_PATH_DICT = {
    ("MCl", "RB", "200-300"): os.path.join(repo_folder, "weights", "MCl_RB_200-300_GP_UECO.pt"),
    ("MCl", "RB", "800-1200"): os.path.join(repo_folder, "weights", "MCl_RB_800-1200_GP_UECO.pt"),
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

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = GNN4COMClSolver(
        model=UECOModel(
            env=GNN4COEnv(
                task="MCl", mode="solve", sparse_factor=1, device=DEVICE,
            ),
            encoder=UECOEncoder(
                task=TASK,
                paradigm="GP",
                sparse=True,
                **encoder_params
            ),
            decoder=MClDecoder(
                decoding_type=SOLVING_SETTINGS[0], 
                local_search_type=SOLVING_SETTINGS[1],
                rlsa_tau=rlsa_settings[0],
                rlsa_d=rlsa_settings[1],
                rlsa_k=rlsa_settings[2],
                rlsa_t=rlsa_settings[3],
                rlsa_beta=rlsa_settings[4],
                rlsa_alpha=rlsa_settings[5],
                rlsa_device="cuda"
            ),
            weight_path=check_and_ckpt_to_pt(
                WEIGHT_PATH_DICT[TEST_TYPE],
                save_path=os.path.join(repo_folder, "weights", f"tmp_{TEST_TYPE}_GP_UECO.pt")
            )
        )
    )
    print(f"[TEST INFO] Test started on {DEVICE}")
    print(f"[TEST INFO] Test data: {TEST_TYPE}")
    print(f"[TEST INFO] Test setting: {SOLVING_SETTINGS}")
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print_results(*solver.evaluate(calculate_gap=True), avg_time=solver.avg_time)
