import os
import sys
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
os.chdir(root_folder)
sys.path.append(root_folder)
from env import *
from model import *
from solver import *
from tabulate import tabulate


###############################
#          Settings           #
###############################

# Problem Settings
TASK = "MCl"
NODES_NUM = "200-300" #"2000-3000"
TEST_TYPE = (TASK, "RB", NODES_NUM)

# Solving Settings
DEVICE = "cuda:0"
USE_RLSA = False

# COExpander Settings
INFERENCE_STEP = 1 # Is; keep 1 for the default AE setting
DETERMINATE_STEP = 20 # Ds
SAMPLING_NUM = 1 # S
BEAM_SIZE = -1 # Beam Search

# Model Settings
NUM_ATT_LAYERS = {
    ("MCl", "RB", "200-300"): 4,
    ("MCl", "RB", "800-1200"): 3,
}[TEST_TYPE]

encoder_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 3,
    "num_att_layers": NUM_ATT_LAYERS,
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
TEST_FILE_DICT = {
    ("MCl", "RB", "200-300"): "../data/test/MCl/mcl_rb-small_gurobi_19.082.txt",
    ("MCl", "RB", "800-1200"): "../data/test/MCl/mcl_rb-large_gurobi-300s_40.182.txt",
}

WEIGHT_PATH_DICT = {
    ("MCl", "RB", "200-300"): "../weights/AE/mcl/MCl_RB_200-300_AE_UECO.pt",
    ("MCl", "RB", "800-1200"): "../weights/AE/mcl/MCl_RB_800-1200_AE_UECO.pt",
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
    solver = COExpanderMClSolver(
        model=UECOModel(
            env=COExpanderEnv(
                task="MCl", mode="solve", sparse_factor=1, device="cuda",
            ),
            encoder=UECOEncoder(
                task="MCl",
                paradigm="AE",
                sparse=True,
                sparse_factor=1,
                **encoder_params
            ),
            decoder=COExpanderDecoder(
                decode_kwargs={
                    "use_rlsa": USE_RLSA,
                    "rlsa_tau": rlsa_settings[0],
                    "rlsa_d": rlsa_settings[1],
                    "rlsa_k": rlsa_settings[2],
                    "rlsa_t": rlsa_settings[3],
                    "rlsa_beta": rlsa_settings[4],
                    "rlsa_alpha": rlsa_settings[5],
                }
            ),
            weight_path=check_and_ckpt_to_pt(
                WEIGHT_PATH_DICT[TEST_TYPE],
                save_path=f"../weights/AE/mcl/tmp_{TEST_TYPE}_AE_UECO.pt"
            ),
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP,
            beam_size=BEAM_SIZE,
        )
    )
    print(f"[TEST INFO] Test setting: Is={INFERENCE_STEP}, Ds={DETERMINATE_STEP}, S={SAMPLING_NUM}")
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    result_tuple = solver.evaluate(calculate_gap=True)
    print_results(*result_tuple, avg_time=solver.avg_time)
