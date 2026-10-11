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


##########################
#    Training Configs    #
##########################

TRAIN_TYPE = ("MIS", "RB", "200-300")

TRAIN_FOLDER_DICT = {
    ("MIS", "RB", "200-300"): "../data/train/MIS/MIS_RB-SMALL_KaMIS-10s",
    ("MIS", "RB", "800-1200"): "../data/train/MIS/MIS_RB-LARGE_KaMIS-60s",
    ("MIS", "ER", "700-800"): "../data/train/MIS/MIS_ER-700-800-0.15_KaMIS-20s",
    ("MIS", "SATLIB", None): "../data/train/MIS/MIS_SATLIB_KAMIS-60s"
}

VAL_PATH_DICT = {
    ("MIS", "RB", "200-300"): "../data/test/MIS/mis_rb-small_kamis_10s_20.09.txt",
    ("MIS", "RB", "800-1200"): "../data/test/MIS/mis_rb-large_kamis-60s_43.004.txt",
    ("MIS", "ER", "700-800"): "../data/test/MIS/mis_er-700-800_kamis-60s_44.96875.txt",
    ("MIS", "SATLIB", None): "../data/test/MIS/mis_satlib_kamis_10s_425.954.txt"
}

encoder_params = {
    "embedding_dim": 256,
    "num_gcn_layers": 3,
    "num_att_layers": 4, # 3 for ER-700-800
    "num_heads": 16,
    "qkv_dim": 16,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "dual": False
}

VAL_SAMPLES_DICT = {
    "200-300": 500,
    "700-800": 128,
    "800-1200": 500,
    None: 500
}

BATCH_SIZE_DICT = {
    "200-300": 8,
    "700-800": 4,
    "800-1200": 4,
    None: 4
}

##########################
#          Main          #
##########################

if __name__ == "__main__":
    timestamp = time.time() 
    formatted_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))[-8:]
    model = UECOModel(
        env=COExpanderEnv(
            task="MIS",
            mode="train",
            train_data_size=39500, # 64000 for RB-200-300; 12800 for ER-700-800; 39500 for SATLIB
            val_data_size=VAL_SAMPLES_DICT[TRAIN_TYPE[2]],
            train_batch_size=BATCH_SIZE_DICT[TRAIN_TYPE[2]],
            val_batch_size=1,
            num_workers=4,
            sparse_factor=1,
            device="cuda:0",
            train_folder=TRAIN_FOLDER_DICT[TRAIN_TYPE],
            val_path=VAL_PATH_DICT[TRAIN_TYPE],
            store_data=False # set False if your server's memory is not large enough.
        ),
        encoder=UECOEncoder(
            task="MIS",
            paradigm="AE",
            sparse=True,
            sparse_factor=1,
            **encoder_params
        ),
        decoder=COExpanderDecoder(),
        learning_rate=2e-4,
        weight_decay=5e-5,
        inference_steps=1,
        determinate_steps=1
        # weight_path="path/to/pretrained/file"
    )

    run_name = f"{TRAIN_TYPE[0]}_{TRAIN_TYPE[1]}_{TRAIN_TYPE[2]}_AE_UECO_{formatted_time}"
    trainer = Trainer(
        model=model, devices=[0], max_epochs=50,
        wandb_logger_name=run_name,
        ckpt_monitor="val/costs_avg",
        ckpt_filename="epoch={epoch}-obj={val/costs_avg:.5f}",
        save_top_k=5,
        mode="max"
    )
    trainer.model_train()
