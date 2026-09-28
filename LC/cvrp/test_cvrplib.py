##########################################################################################
# Machine Environment Config

import os

USE_CUDA = True
CUDA_DEVICE_NUM = int(os.environ.get("UECO_CUDA_DEVICE", "0"))


##########################################################################################
# Path Config

import sys
import torch
import numpy as np
import random
import tqdm

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(TASK_DIR))
os.chdir(TASK_DIR)
sys.path.insert(0, os.path.dirname(TASK_DIR))
sys.path.insert(0, REPO_DIR)


##########################################################################################
# import

from CVRPTester_UECO import CVRPTester as Tester


##########################################################################################
# parameters
TEST_DATA = ("CVRP", "uniform", 50)
TEST_DIR = "../../data/test/CVRPLIB"
# NODES_NUM = TEST_DATA[-1]
NUM_AUG = 16
USE_LS = True

WEIGHT_DICT = {
    ("CVRP", "uniform", 50): "../../weights/LC/cvrp/CVRP50_LC_UECO.pt",
    ("CVRP", "uniform", 100): "../../weights/LC/cvrp/CVRP100_LC_UECO.pt",
    ("CVRP", "uniform", 200): "../../weights/LC/cvrp/CVRP200_LC_UECO.pt",
    ("CVRP", "uniform", 500): "../../weights/LC/cvrp/CVRP500_LC_UECO.pt",
}

""" Results   
    [G]                   50           100           200         500
                          41.875\%     15.141\%      30.358\%    12.985\%   (0.338s)

    setting:              100 [G(16)+LS]               100 [G(8)+LS]               500 [G(8)+LS]
    101-200               28.794, 3.485\%, 0.201s      28.799, 3.495\%, 0.175s
    204-1001              79.090, 5.980\%, 2.655s      79.427, 6.351\%, 1.347s
    101-1001 (full)       68.090, 5.467\%, 2.107s      68.362, 5.807\%, 1.081s     68.385, 6.752\%, 1.023s


    101-513 (bench101)    47.748, 4.440\%, 0.757s
"""


model_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 1,
    "num_att_layers": 4,
    "qkv_dim": 16,
    "num_heads": 16,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "eval_type": "argmax",
}

tester_params = {
    "use_cuda": USE_CUDA,
    "cuda_device_num": CUDA_DEVICE_NUM,
    "use_ls": USE_LS,
    "model_load": WEIGHT_DICT[TEST_DATA],
    "test_file": None,
    "aug_factor": NUM_AUG,
    "num_instances": 1
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


##########################################################################################
# main

def main():
    setup_seed(1234)

    _print_config()

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"CVRPLIB directory not found: {TEST_DIR}."
        )

    test_path_list = []
    test_node_cnt_list = []
    for filename in sorted(os.listdir(TEST_DIR)):
        test_txt = os.path.join(TEST_DIR, filename)
        num_nodes = int(filename.split("-")[1][1:]) - 1
        test_path_list.append(test_txt)
        test_node_cnt_list.append(num_nodes)

    gaps_list = []
    objs_list = []
    time_list = []

    for inst, num_nodes in zip(test_path_list, test_node_cnt_list):
        env_params = {
            "task": TEST_DATA[0],
            "node_cnt": num_nodes,
            "pomo_size": max(num_nodes, 100),
        }
        tester_params["test_file"] = inst
        tester = Tester(
            env_params=env_params,
            model_params=model_params,
            tester_params=tester_params
        )
        metrics, inst_time = tester.run()
        gaps_list.append(metrics[2])
        time_list.append(inst_time)
        objs_list.append(metrics[0])

    print(f"[Test Result] Avg. obj : {sum(objs_list) / len(objs_list):.5f}")
    print(f"[Test Result] Avg. gap : {sum(gaps_list) / len(gaps_list):.5f}%")
    print(f"[Test Result] Avg. time: {sum(time_list) / len(time_list):.5f}s")

def _print_config():
    print("[Test Config] Dataset:", TEST_DATA)
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] Local Search:", USE_LS)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])

##########################################################################################

if __name__ == "__main__":
    main()
