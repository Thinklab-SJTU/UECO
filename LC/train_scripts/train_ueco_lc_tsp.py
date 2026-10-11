
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARADIGM_DIR = os.path.dirname(SCRIPT_DIR)
TASK_DIR = os.path.join(PARADIGM_DIR, "tsp")
REPO_DIR = os.path.dirname(PARADIGM_DIR)
os.chdir(TASK_DIR)
sys.path.insert(0, TASK_DIR)
sys.path.insert(0, PARADIGM_DIR)
sys.path.insert(0, REPO_DIR)
import logging
from utils.utils import create_logger
from TSPTrainer_UECO import TSPTrainer as Trainer
import warnings
warnings.filterwarnings("ignore")

########################
#    Training Config   #
########################

USE_CUDA = True
CUDA_DEVICE_NUM = 0

NODES_NUM = 50
MAX_EPOCH_DICT = {
    50: 8000,
    100: 3000, #12000
    500: 1000,
}
BATCH_SIZE_DICT = {
    50: 100,
    100: 25,
    500: 1,
}
# SAMPLES_PER_EPOCH = 10000
SAMPLES_PER_EPOCH = 1000

TRAIN_TYPE = ("TSP", "uniform", NODES_NUM)

VAL_PATH_DICT = {
    ("TSP", "uniform", 50): "../../data/test/TSP/tsp50_concorde_5.68759.txt",
    ("TSP", "uniform", 100): "../../data/test/TSP/tsp100_concorde_7.75585.txt",
    ("TSP", "uniform", 500): "../../data/test/TSP/tsp500_concorde_16.54581.txt",
}

env_params = {
    "task": TRAIN_TYPE[0],
    "node_cnt": NODES_NUM,
    "pomo_size": NODES_NUM  # same as node_cnt
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

optimizer_params = {
    "optimizer": {
        # "lr": 2e-4,
        "lr": 5e-5,
        "weight_decay": 1e-5
    },
    "scheduler": {
        # "milestones": [2001, 2101],  # if further training is needed
        "milestones": [501, 601],
        "gamma": 0.1
    }
}

trainer_params = {
    "use_cuda": USE_CUDA,
    "cuda_device_num": CUDA_DEVICE_NUM,
    "epochs": MAX_EPOCH_DICT[NODES_NUM],
    "train_episodes": SAMPLES_PER_EPOCH,
    "train_batch_size": BATCH_SIZE_DICT[NODES_NUM],
    "val_interval": 1,
    "val_file": VAL_PATH_DICT[TRAIN_TYPE],
    "val_num": 1280,
    "val_aug_factor": 1,
    "logging": {
        "model_save_interval": 100,
        "img_save_interval": 200,
        "log_image_params_1": {
            "json_foldername": "log_image_style",
            "filename": "style.json"
        },
        "log_image_params_2": {
            "json_foldername": "log_image_style",
            "filename": "style_loss.json"
        },
    },
    "model_load": {
        "enable": False,   # enable loading pre-trained model
        "only_weight": False,
        "path": "",
        "epoch": 1,  # epoch version of pre-trained model to load.
    },
}

logger_params = {
    "log_file": {
        "desc": f"train_{TRAIN_TYPE[0]}{NODES_NUM}_LC_UECO",
        "filename": "log.txt"
    }
}

##########################################################################################
# main

def main():
    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    trainer.run()


def _print_config():
    logger = logging.getLogger("root")
    logger.info(f"DEBUG_MODE: {sys.gettrace() is not None}")
    logger.info(f"USE_CUDA: {USE_CUDA}, CUDA_DEVICE_NUM: {CUDA_DEVICE_NUM}")
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith("params")]


if __name__ == "__main__":
    main()
