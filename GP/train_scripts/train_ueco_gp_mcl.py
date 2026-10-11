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
from model import UECOModel, UECOEncoder, MClDecoder


##########################
#    Training Configs    #
##########################

TRAIN_TYPE = ("MCl", "RB", "200-300")

TRAIN_FOLDER_DICT = {
    ("MCl", "RB", "200-300"): "../data/train/MCl/MCl-RB-SMALL_GUROBI-60s",
    ("MCl", "RB", "800-1200"): "../data/train/MCl/MCl-RB-LARGE_GUROBI-300s",
}

VAL_PATH_DICT = {
    ("MCl", "RB", "200-300"): "../data/test/MCl/mcl_rb-small_gurobi_19.082.txt",
    ("MCl", "RB", "800-1200"): "../data/test/MCl/mcl_rb-large_gurobi-300s_40.182.txt",
}


encoder_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 3,
    "num_att_layers": 4, # 3 for large
    "num_heads": 16,
    "qkv_dim": 16,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "dual": False
}

VAL_SAMPLES_DICT = {
    "200-300": 500,
    "800-1200": 500
}

BATCH_SIZE_DICT = {
    "200-300": 8,
    "800-1200": 4
}

##########################
#          Main          #
##########################

if __name__ == "__main__":
    timestamp = time.time() 
    formatted_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))[-8:]
    model=UECOModel(
        env=GNN4COEnv(
            task="MCl", 
            mode="train", 
            train_data_size=64000, # 64000 for small/ 6400 for large
            val_data_size=VAL_SAMPLES_DICT[TRAIN_TYPE[2]],
            train_batch_size=BATCH_SIZE_DICT[TRAIN_TYPE[2]],
            val_batch_size=1,
            sparse_factor=1, 
            device="cuda:0",
            train_folder=TRAIN_FOLDER_DICT[TRAIN_TYPE],
            val_path=VAL_PATH_DICT[TRAIN_TYPE],
            store_data=False # set False if your server's memory is not large enough.
        ),
        encoder=UECOEncoder(
            task="MCl",
            paradigm="GP",
            sparse=True,
            sparse_factor=1,
            **encoder_params
        ),
        decoder=MClDecoder("greedy"),
        learning_rate=2e-4,
        weight_decay=1e-4
    )

    run_name = f"{TRAIN_TYPE[0]}_{TRAIN_TYPE[1]}_{TRAIN_TYPE[2]}_GP_UECO_{formatted_time}"
    trainer = Trainer(
        model=model, devices=[0], max_epochs=50,
        wandb_logger_name=run_name,
        ckpt_monitor="val/costs_avg",
        ckpt_filename="epoch={epoch}-obj={val/costs_avg:.5f}",
        save_top_k=5,
        mode="max"
    )
    trainer.model_train()
