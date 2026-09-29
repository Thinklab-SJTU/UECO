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
TASK = "MCut"
NODES_NUM = "200-300"
TEST_TYPE = (TASK, "BA", NODES_NUM)

# Solving Settings
DEVICE = "cuda:0"
USE_RLSA = False

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
    ("MCut", "BA", "2000-3000"): (1, 100, 1000, 1000)
}

# COExpander Settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 20 # Ds
SAMPLING_NUM = 1 # S
BEAM_SIZE = -1 # Beam Search

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
TEST_FILE_DICT = {
    ("MCut", "BA", "200-300"): "../data/test/MCut/mcut_ba-small_gurobi_60s_727.844.txt",
    ("MCut", "BA", "800-1200"): "../data/test/MCut/mcut_ba-large_gurobi_300s_2936.886.txt",
}

WEIGHT_PATH_DICT = {
    ("MCut", "BA", "200-300"): "../weights/AE/mcut/MCut_BA_200-300_AE_UECO.pt",
    ("MCut", "BA", "800-1200"): "../weights/AE/mcut/MCut_BA_800-1200_AE_UECO.pt",
}

################################
#             Main             #
################################

# main
if __name__ == "__main__":
    # res = []
    # for i in range(1, 11):
        # DETERMINATE_STEP = i

        rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
        solver = COExpanderMCutSolver(
            model=UECOModel(
                env=COExpanderEnv(
                    task="MCut", mode="solve", sparse_factor=1, device=DEVICE,
                ),
                encoder=UECOEncoder(
                    task="MCut",
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
                        "rlsa_t": rlsa_settings[3]
                    }
                ),
                inference_steps=INFERENCE_STEP,
                determinate_steps=DETERMINATE_STEP,
                beam_size=BEAM_SIZE,
                weight_path=check_and_ckpt_to_pt(
                    WEIGHT_PATH_DICT[TEST_TYPE],
                    save_path=f"../weights/AE/mcut/tmp_{TEST_TYPE}_AE_UECO.pt"
                )
            )
        )
        # print("[TEST INFO] Test started ...")
        # print(f"[TEST INFO] Test data: {TEST_TYPE}")
        # print(f"[TEST INFO] Test setting: Greedy {'+ RLSA' if USE_RLSA else ''}")
        print(f"[TEST INFO] Test setting: Is={INFERENCE_STEP}, Ds={DETERMINATE_STEP}, S={SAMPLING_NUM}")
        solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
        solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
        result_tuple = solver.evaluate(calculate_gap=True)
        print_results(*result_tuple, avg_time=solver.avg_time)

        # res.append((i, result_tuple[0], result_tuple[2], solver.avg_time))
    # print(res)
