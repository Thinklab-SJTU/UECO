import os
import sys
import time
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
sys.path.insert(0, root_folder)
from ml4co_kit import Trainer
from model import UECOModel, UECOEncoder, COExpanderDecoder
from env import COExpanderEnv


TRAIN_TYPE = ("TSP", "uniform", 500)
SPARSE_TYPE = (False, -1)

TRAIN_FOLDER_DICT = {
    ("TSP", "uniform", 50): "../../data/train/TSP/TSP50_Uniform_Concorde",
    ("TSP", "uniform", 100): "../../data/train/TSP/TSP100_Uniform_Concorde",
    ("TSP", "uniform", 500): "../../data/train/TSP/TSP500_Uniform_Concorde",
    ("TSP", "uniform", 1000): "../../data/train/TSP/TSP1000_Uniform_LKH-1000",
    ("TSP", "uniform", 10000): "../../data/train/TSP/TSP10000_Uniform_ALKH-1000"
}

VAL_PATH_DICT = {
    ("TSP", "uniform", 50): "../../data/test/TSP/tsp50_concorde_5.68759.txt",
    ("TSP", "uniform", 100): "../../data/test/TSP/tsp100_concorde_7.75585.txt",
    ("TSP", "uniform", 500): "../../data/test/TSP/tsp500_concorde_16.54581.txt",
    ("TSP", "uniform", 1000): "../../data/test/TSP/tsp1000_concorde_23.11812.txt",
    ("TSP", "uniform", 10000): "../../data/test/TSP/tsp10000_lkh_500_71.75483.txt",
    ("TSP", "uniform", 100000): "../../data/test/TSP/tsp100000_lkh_1_226.26067.txt"
}

VAL_SAMPLES_DICT = {
    50: 1280,
    100: 1280,
    500: 128,
    1000: 128,
    10000: 16
}

BATCH_SIZE_DICT = {
    50: 20,
    100: 10,
    500: 1,
    1000: 2,
    10000: 1
}

encoder_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 3,
    "num_att_layers": 3,
    "num_heads": 16,
    "qkv_dim": 16,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "dual": True
}

if __name__ == "__main__":
    timestamp = time.time() 
    formatted_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))[-8:]
    model = UECOModel(
        env=COExpanderEnv(
            task="TSP",
            mode="train",
            train_data_size=20000,
            val_data_size=VAL_SAMPLES_DICT[TRAIN_TYPE[2]],
            train_batch_size=BATCH_SIZE_DICT[TRAIN_TYPE[2]],
            val_batch_size=1,
            num_workers=4,
            sparse_factor=SPARSE_TYPE[1],
            device="cuda",
            train_folder=TRAIN_FOLDER_DICT[TRAIN_TYPE],
            val_path=VAL_PATH_DICT[TRAIN_TYPE],
            store_data=False # set False if your server's memory is not large enough.
        ),
        encoder=UECOEncoder(
            task="TSP",
            paradigm="AE",
            sparse=SPARSE_TYPE[0],
            sparse_factor=SPARSE_TYPE[1],
            **encoder_params
        ),
        decoder=COExpanderDecoder(),
        learning_rate=2e-4,
        weight_decay=5e-5,
        inference_steps=1,
        determinate_steps=1
    )

    run_name = f"{TRAIN_TYPE[0]}{TRAIN_TYPE[2]}_AE_UECO_{formatted_time}"
    trainer = Trainer(
        model=model, devices=[0], max_epochs=50,
        wandb_logger_name=run_name,
        ckpt_monitor="val/costs_avg",
        ckpt_filename="epoch={epoch}-obj={val/costs_avg:.5f}",
        save_top_k=5
    )
    trainer.model_train()
