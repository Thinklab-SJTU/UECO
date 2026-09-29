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
TASK = "MIS"
NODES_NUM = "200-300"
TEST_TYPE = (TASK, "RB", NODES_NUM)

# Solving Settings
DEVICE = "cuda:0"
USE_RLSA = False

# COExpander Settings
INFERENCE_STEP = 1 # Is; keep 1 for the default AE setting
DETERMINATE_STEP = 50 # Ds
SAMPLING_NUM = 1 # S
BEAM_SIZE = -1 # Beam Search

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
    ("MIS", "RB", "200-300"): (0.01, 5, 1000, 500, 1.02, 0.3),
    ("MIS", "RB", "800-1200"): (0.01, 5, 1000, 1000, 1.02, 0.3),
    ("MIS", "RB", "2000-3000"): (0.01, 5, 1000, 1000, 1.02, 0.3),
    ("MIS", "ER", "700-800"): (0.2, 10, 1000, 1000, 1.001, 0.3),
    ("MIS", "ER", "1400-1600"): (0.2, 10, 1000, 1000, 1.001, 0.3),
    ("MIS", "ER", "9000-11000"): (0.2, 10, 1000, 1000, 1.001, 0.3),
    ("MIS", "SATLIB", None): (0.01, 2, 1000, 1000, 1.02, 0.3),
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    ("MIS", "RB", "200-300"): "../data/test/MIS/mis_rb-small_kamis_10s_20.09.txt",
    ("MIS", "RB", "800-1200"): "../data/test/MIS/mis_rb-large_kamis-60s_43.004.txt",
    ("MIS", "ER", "700-800"): "../data/test/MIS/mis_er-700-800_kamis-60s_44.96875.txt",
    ("MIS", "SATLIB", None): "../data/test/MIS/mis_satlib_kamis_10s_425.954.txt",
}

WEIGHT_PATH_DICT = {
    ("MIS", "RB", "200-300"): "../weights/AE/mis/MIS_RB_200-300_AE_UECO.pt",
    ("MIS", "RB", "800-1200"): "../weights/AE/mis/MIS_RB_800-1200_AE_UECO.pt",
    ("MIS", "ER", "700-800"): "../weights/AE/mis/MIS_ER_700-800_AE_UECO.pt",
    ("MIS", "SATLIB", None): "../weights/AE/mis/MIS_SATLIB_AE_UECO.pt",
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

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = COExpanderMISSolver(
        model=UECOModel(
            env=COExpanderEnv(
                task="MIS", mode="solve", sparse_factor=1, device=DEVICE,
            ),
            encoder=UECOEncoder(
                task="MIS",
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
                save_path=f"../weights/AE/mis/tmp_{TEST_TYPE}_AE_UECO.pt"
            ),
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP,
            beam_size=BEAM_SIZE,
        )
    )
    print("[TEST INFO] Test started ...")
    print(f"[TEST INFO] Test data: {TEST_TYPE}")
    print(f"[TEST INFO] Test setting: Greedy {'+ RLSA' if USE_RLSA else ''}")
    print(f"[TEST INFO] Test setting: Is={INFERENCE_STEP}, Ds={DETERMINATE_STEP}, S={SAMPLING_NUM}")
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print_results(*solver.evaluate(calculate_gap=True), avg_time=solver.avg_time)
