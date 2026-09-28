import os
import sys
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(TASK_DIR))
os.chdir(TASK_DIR)
sys.path.insert(0, os.path.dirname(TASK_DIR))
sys.path.insert(0, REPO_DIR)
import logging
from utils.utils import create_logger, copy_all_src
from CVRPTrainer_UECO import CVRPTrainer as Trainer
import warnings
warnings.filterwarnings("ignore")

########################
#    Training Config   #
########################

USE_CUDA = True
CUDA_DEVICE_NUM = 4

NODES_NUM = 100
MAX_EPOCH_DICT = {
    50: 3000,
    100: 2000,
    200: 1000,
    500: 500
}
BATCH_SIZE_DICT = {
    50: 200,
    100: 50,
    200: 5,
    500: 1
}
#Batch_size for RL training: 200 for n=50, 128 for n=100, 16 for n=200, 4 for 500
SAMPLES_PER_EPOCH = 10000

TRAIN_TYPE = ("CVRP", "uniform", NODES_NUM)

DATA_ROOT = os.environ.get("UECO_DATA_ROOT", os.path.join(REPO_DIR, "data"))
VAL_PATH_DICT = {
    ("CVRP", "uniform", 50): os.path.join(DATA_ROOT, "CVRP", "cvrp50_hgs-1s_10.366.txt"),
    ("CVRP", "uniform", 100): os.path.join(DATA_ROOT, "CVRP", "cvrp100_hgs-20s_15.563.txt"),
    ("CVRP", "uniform", 200): os.path.join(DATA_ROOT, "CVRP", "cvrp200_hgs-60s_19.630.txt"),
    ("CVRP", "uniform", 500): os.path.join(DATA_ROOT, "CVRP", "cvrp500_hgs-300s_37.154.txt"),
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
    "dual": True
}

optimizer_params = {
    "optimizer": {
        "lr": 2e-4 if NODES_NUM <= 50 else 1e-4,
        "weight_decay": 1e-5
    },
    "scheduler": {
        "milestones": [1501, 1601], # [2001, 2101] for CVRP-50, [1501, 1601] for CVRP-100, [501, 601] for CVRP-200
        "gamma": 0.1
    }
}

trainer_params = {
    "use_cuda": USE_CUDA,
    "cuda_device_num": CUDA_DEVICE_NUM,
    "epochs": MAX_EPOCH_DICT[NODES_NUM],
    "train_episodes": SAMPLES_PER_EPOCH,
    "train_batch_size": BATCH_SIZE_DICT[NODES_NUM],
    "val_interval": 3,
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
    copy_all_src(trainer.result_folder)
    trainer.run()

def _print_config():
    logger = logging.getLogger("root")
    logger.info("USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith("params")]


##########################################################################################

if __name__ == "__main__":
    main()
