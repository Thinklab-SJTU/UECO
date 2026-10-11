import os
import sys
import time
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
os.chdir(root_folder)
sys.path.insert(0, root_folder)
from ml4co_kit import Trainer
from env import GNN4COEnv
from model import UECOModel, UECOEncoder, TSPDecoder

##########################
#    Training Configs    #
##########################

TRAIN_TYPE = ("TSP", "uniform", 50)
SPARSE_FACTOR = {50: -1, 100: -1, 500: 50}[TRAIN_TYPE[2]]
SPARSE_TYPE = (SPARSE_FACTOR > 0, SPARSE_FACTOR)

TRAIN_FOLDER_DICT = {
    ("TSP", "uniform", 50): "../data/train/TSP/TSP50_Uniform_Concorde",
    ("TSP", "uniform", 100): "../data/train/TSP/TSP100_Uniform_Concorde",
    ("TSP", "uniform", 500): "../data/train/TSP/TSP500_Uniform_Concorde",
}

VAL_PATH_DICT = {
    ("TSP", "uniform", 50): "../data/test/TSP/tsp50_concorde_5.68759.txt",
    ("TSP", "uniform", 100): "../data/test/TSP/tsp100_concorde_7.75585.txt",
    ("TSP", "uniform", 500): "../data/test/TSP/tsp500_concorde_16.54581.txt",
}

VAL_SAMPLES_DICT = {
    50: 1280,
    100: 1280,
    500: 128,
}

BATCH_SIZE_DICT = {
    50: 20,
    100: 16,
    500: 3,
}

encoder_params = {
    "embedding_dim": 512,
    "num_gcn_layers": 3,
    "num_att_layers": 5,
    "num_heads": 16,
    "qkv_dim": 16,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "dual": True
}

##########################
#          Main          #
##########################

if __name__ == "__main__":
    timestamp = time.time() 
    formatted_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))[-8:] # month-day-hour-minute
    model=UECOModel(
        env=GNN4COEnv(
            task="TSP",
            mode="train", 
            train_data_size=12800, # 128000 for tsp-50/100; 64000 for tsp-500
            val_data_size=VAL_SAMPLES_DICT[TRAIN_TYPE[2]],
            train_batch_size=BATCH_SIZE_DICT[TRAIN_TYPE[2]],
            val_batch_size=1,
            sparse_factor=SPARSE_TYPE[1], 
            device="cuda:0",
            train_folder=TRAIN_FOLDER_DICT[TRAIN_TYPE],
            val_path=VAL_PATH_DICT[TRAIN_TYPE],
            store_data=False # set False if your server's memory is not large enough.
        ),
        encoder=UECOEncoder(
            task="TSP",
            paradigm="GP",
            sparse=SPARSE_TYPE[0],
            sparse_factor=SPARSE_TYPE[1],
            **encoder_params
        ),
        decoder=TSPDecoder("greedy"),
        learning_rate=2e-4,
        weight_decay=5e-5
    )

    run_name = f"{TRAIN_TYPE[0]}{TRAIN_TYPE[2]}_GP_UECO_{formatted_time}"
    trainer = Trainer(
        model=model, devices=[0], max_epochs=100,
        wandb_logger_name=run_name,
        ckpt_monitor="val/costs_avg",
        ckpt_filename="epoch={epoch}-obj={val/costs_avg:.5f}",
        save_top_k=5
    )
    trainer.model_train()
