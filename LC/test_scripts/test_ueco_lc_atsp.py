import os
import random
import sys

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARADIGM_DIR = os.path.dirname(SCRIPT_DIR)
TASK_DIR = os.path.join(PARADIGM_DIR, "atsp")
REPO_DIR = os.path.dirname(PARADIGM_DIR)
os.chdir(TASK_DIR)
sys.path.insert(0, TASK_DIR)
sys.path.insert(0, PARADIGM_DIR)
sys.path.insert(0, REPO_DIR)

from ATSPTester_UECO import ATSPTester as Tester


USE_CUDA = True
CUDA_DEVICE_NUM = int(os.environ.get("UECO_CUDA_DEVICE", "0"))

TEST_DATA = ("ATSP", "uniform", 50)
NODES_NUM = TEST_DATA[-1]
NUM_AUG = 16
USE_TWO_OPT = True

WEIGHT_DICT = {
    ("ATSP", "uniform", 50): "../../weights/LC/atsp/ATSP50_LC_UECO.pt",
    ("ATSP", "uniform", 100): "../../weights/LC/atsp/ATSP100_LC_UECO.pt",
    ("ATSP", "uniform", 200): "../../weights/LC/atsp/ATSP200_LC_UECO.pt",
    ("ATSP", "uniform", 500): "../../weights/LC/atsp/ATSP500_LC_UECO.pt",
}

TEST_PATH_DICT = {
    ("ATSP", "uniform", 50): "../../data/test/ATSP/atsp50_uniform_lkh_1000_1.55448.txt",
    ("ATSP", "uniform", 100): "../../data/test/ATSP/atsp100_uniform_lkh_1000_1.56604.txt",
    ("ATSP", "uniform", 200): "../../data/test/ATSP/atsp200_uniform_lkh_1000_1.56472.txt",
    ("ATSP", "uniform", 500): "../../data/test/ATSP/atsp500_uniform_lkh_1000_1.57345.txt",
}

env_params = {
    "test_type": TEST_DATA,
    "pos_embedding_dim": 256,
    "node_cnt": NODES_NUM,
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
    "use_2opt": USE_TWO_OPT,
    "num_instances": int(os.environ.get("UECO_TEST_NUM_INSTANCES", "2500")),
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(1234)
    _print_config()
    tester = Tester(
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params,
    )
    tester.run()


def _print_config():
    print("[Test Config] Dataset:", TEST_DATA)
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] 2-OPT:", USE_TWO_OPT)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])


if __name__ == "__main__":
    main()
