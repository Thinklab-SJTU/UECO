import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARADIGM_DIR = os.path.dirname(SCRIPT_DIR)
TASK_DIR = os.path.join(PARADIGM_DIR, "atsp")
REPO_DIR = os.path.dirname(PARADIGM_DIR)
os.chdir(TASK_DIR)
sys.path.insert(0, TASK_DIR)
sys.path.insert(0, PARADIGM_DIR)
sys.path.insert(0, REPO_DIR)
import logging
from utils.utils import create_logger
from ATSPTrainer_UECO import ATSPTrainer as Trainer
import warnings
warnings.filterwarnings("ignore")

########################
#    Training Config   #
########################

USE_CUDA = True
CUDA_DEVICE_NUM = 1

NODES_NUM = 50
MAX_EPOCH_DICT = {
    50: 8000,
    100: 3000, #12000
    200: 1000,
    500: 500
}
BATCH_SIZE_DICT = {
    50: 100,
    100: 25,
    200: 5,
    500: 1
}
SAMPLES_PER_EPOCH = 10000

TRAIN_TYPE = ("ATSP", "uniform", NODES_NUM)

VAL_PATH_DICT = {
    ("ATSP", "uniform", 50): "../../data/test/ATSP/atsp50_uniform_lkh_1000_1.55448.txt",
    ("ATSP", "uniform", 100): "../../data/test/ATSP/atsp100_uniform_lkh_1000_1.56604.txt",
    ("ATSP", "uniform", 200): "../../data/test/ATSP/atsp200_uniform_lkh_1000_1.56472.txt",
    ("ATSP", "uniform", 500): "../../data/test/ATSP/atsp500_uniform_lkh_1000_1.57345.txt",
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
        "lr": 2e-4 if NODES_NUM <= 100 else 1e-4,
        "weight_decay": 1e-5
    },
    "scheduler": {
        "milestones": [501, 601], # [2001, 2101] for ATSP-50/100, [501, 601] for ATSP-200
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
        "only_weight": True,
        "path": "",
        "epoch": 0,  # epoch version of pre-trained model to load.
    },
}

logger_params = {
    "log_file": {
        "desc": f"train_{TRAIN_TYPE[0]}{NODES_NUM}_LC_UECO",
        "filename": "log.txt"
    }
}


##############
# main

def main():
    create_logger(**logger_params)
    _print_config()
    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )
    trainer.run()

def _print_config():
    logger = logging.getLogger("root")
    logger.info("USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith("params")]


##########################################################################################

if __name__ == "__main__":
    main()
