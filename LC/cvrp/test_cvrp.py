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
TEST_DATA = ("CVRP", "uniform", 500)
NODES_NUM = TEST_DATA[-1]
NUM_AUG = 1
USE_LS = False

WEIGHT_DICT = {
    ("CVRP", "uniform", 50): "../../weights/LC/cvrp/CVRP50_LC_UECO.pt",
    ("CVRP", "uniform", 100): "../../weights/LC/cvrp/CVRP100_LC_UECO.pt",
    ("CVRP", "uniform", 200): "../../weights/LC/cvrp/CVRP200_LC_UECO.pt",
    ("CVRP", "uniform", 500): "../../weights/LC/cvrp/CVRP500_LC_UECO.pt",
}

TEST_PATH_DICT = {
    ("CVRP", "uniform", 50): "../../data/test/CVRP/cvrp50_hgs-1s_10.366.txt",
    ("CVRP", "uniform", 100): "../../data/test/CVRP/cvrp100_hgs-20s_15.563.txt",
    ("CVRP", "uniform", 200): "../../data/test/CVRP/cvrp200_hgs-60s_19.630.txt",
    ("CVRP", "uniform", 500): "../../data/test/CVRP/cvrp500_hgs-300s_37.154.txt",
}

env_params = {
    "task": TEST_DATA[0],
    "pos_embedding_dim": 256,
    "node_cnt": TEST_DATA[2],
    "pomo_size": NODES_NUM,
}

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
    "model_load": WEIGHT_DICT[TEST_DATA],
    "test_file": TEST_PATH_DICT[TEST_DATA],
    "augmentation_enable": True,
    "aug_factor": NUM_AUG,
    "use_ls": USE_LS,
    "num_instances": int(os.environ.get(
        "UECO_TEST_NUM_INSTANCES", "10000" if NODES_NUM <= 100 else "100"
    )),
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

    tester = Tester(
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params
    )
    tester.run()

def _print_config():
    print("[Test Config] Dataset:", TEST_DATA)
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] Local Search:", USE_LS)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])

##########################################################################################

if __name__ == "__main__":
    main()
