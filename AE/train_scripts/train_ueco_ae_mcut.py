import os
import sys
import time
task_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(task_folder)
repo_folder = os.path.dirname(root_folder)
os.chdir(root_folder)
sys.path.insert(0, root_folder)
from ml4co_kit import Trainer
from model import UECOModel, UECOEncoder, COExpanderDecoder
from env import COExpanderEnv


TRAIN_TYPE = ("MCut", "BA", "800-1200")

TRAIN_FOLDER_DICT = {
    ("MCut", "BA", "200-300"): "../data/train/MCut/MCut_BA-SMALL_Gurobi-60s",
    ("MCut", "BA", "800-1200"): "../data/train/MCut/MCut_BA-LARGE_Gurobi-300s",
}

VAL_PATH_DICT = {
    ("MCut", "BA", "200-300"): "../data/test/MCut/mcut_ba-small_gurobi_60s_727.844.txt",
    ("MCut", "BA", "800-1200"): "../data/test/MCut/mcut_ba-large_gurobi_300s_2936.886.txt",
}

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

VAL_SAMPLES_DICT = {
    "200-300": 500,
    "800-1200": 500
}

BATCH_SIZE_DICT = {
    "200-300": 8,
    "800-1200": 4
}

if __name__ == "__main__":
    timestamp = time.time() 
    formatted_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))[-8:]
    model = UECOModel(
        env=COExpanderEnv(
            task="MCut",
            mode="train",
            train_data_size=128000,
            val_data_size=VAL_SAMPLES_DICT[TRAIN_TYPE[2]],
            train_batch_size=BATCH_SIZE_DICT[TRAIN_TYPE[2]],
            val_batch_size=1,
            num_workers=4,
            sparse_factor=1,
            device="cuda",
            train_folder=TRAIN_FOLDER_DICT[TRAIN_TYPE],
            val_path=VAL_PATH_DICT[TRAIN_TYPE],
            store_data=False # set False if your server's memory is not large enough.
        ),
        encoder=UECOEncoder(
            task="MCut",
            paradigm="AE",
            sparse=True,
            sparse_factor=1,
            **encoder_params
        ),
        decoder=COExpanderDecoder(),
        learning_rate=5e-5, # 2e-4 for 200-300
        weight_decay=5e-5,
        inference_steps=1,
        determinate_steps=1
    )

    run_name = f"{TRAIN_TYPE[0]}_{TRAIN_TYPE[1]}_{TRAIN_TYPE[2]}_AE_UECO_{formatted_time}"
    trainer = Trainer(
        model=model, devices=[0,1,2], max_epochs=50,
        wandb_logger_name=run_name,
        ckpt_monitor="val/costs_avg",
        ckpt_filename="epoch={epoch}-obj={val/costs_avg:.5f}",
        save_top_k=5,
        mode="max"
    )
    trainer.model_train()
