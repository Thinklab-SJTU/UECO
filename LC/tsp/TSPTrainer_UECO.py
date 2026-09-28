
import torch
from torch import Tensor
from typing import Tuple
from logging import getLogger
from TSPEnv import TSPEnv as Env
from TSPModel_UECO import UECOLCModel as Model
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from utils.utils import *
from ml4co_kit import *
import sys
from importlib import import_module


class TSPTrainer:
    def __init__(
            self,
            env_params: dict,
            model_params: dict,
            optimizer_params: dict,
            trainer_params: dict
        ):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.best_score = 1e10
        self.best_cls_scores = None

        # cuda
        USE_CUDA = self.trainer_params["use_cuda"]
        if USE_CUDA:
            cuda_device_num = self.trainer_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params["optimizer"])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params["scheduler"])
        self.task = env_params["task"]

        # Restore
        self.start_epoch = 1
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_fullname = model_load["path"]
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if not model_load["only_weight"]:
                self.start_epoch = 1 + model_load["epoch"]
                self.result_log.set_raw_data(checkpoint["result_log"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.last_epoch = model_load["epoch"] - 1
            self.logger.info("Saved Model Loaded!")
            self.plot_baseline = False

        self.plot_names = ["val_score"]  
        if trainer_params["logging"].get("base_results") is not None:
            self.plot_baseline = True
            sys.path.append(os.path.dirname(trainer_params["logging"]["base_results"]))
            base_results = import_module("base_data")
            val_score_list_base: list = getattr(base_results, "val_score_list_base", None)
            val_score_list_rect: list = getattr(base_results, "val_score_list_rect", None)
            val_score_list_il: list = getattr(base_results, "val_score_list_il", None)
            if val_score_list_base is not None:
                self.plot_names.append("val_score_base")
                for epoch, value in enumerate(val_score_list_base):
                    self.result_log.append("val_score_base", epoch + 1, value)
            if val_score_list_rect is not None:
                self.plot_names.append("val_score_rect")
                for epoch, value in enumerate(val_score_list_rect):
                    self.result_log.append("val_score_rect", epoch + 1, value)
            if val_score_list_il is not None:
                self.plot_names.append("val_score_il")
                for epoch, value in enumerate(val_score_list_il):
                    self.result_log.append("val_score_il", epoch + 1, value)
        else:
            self.plot_baseline = False

        # utility
        self.time_estimator = TimeEstimator()

        # validation
        self.tmp_solver = TSPSolver()
        self.tmp_solver.from_txt(self.trainer_params["val_file"], ref=True, show_time=True)
        self.val_num = min(trainer_params["val_num"], self.tmp_solver.points.shape[0])
        self.best_val_score = 1e6

    def run(self):
        # self._validation()

        self.time_estimator.reset(self.start_epoch)
        val_interval = self.trainer_params["val_interval"]
        for epoch in range(self.start_epoch, self.trainer_params["epochs"]+1):
            self.logger.info("=================================================================")

            # Param scheduler 
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, train_score)
            self.result_log.append("train_loss", epoch, train_loss)

            # Validation
            if epoch % val_interval == 0:
                val_score = self._validation() # 8x augmented
                self.result_log.append("val_score", epoch, val_score)
                self.logger.info(f"Epoch {epoch:3d}: Val    Score: {val_score:.4f}")
                if val_score < self.best_val_score:
                    self.best_val_score = val_score
                    self.logger.info("Saving new best_model upon validation")
                    checkpoint_dict = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "result_log": self.result_log.get_raw_data()
                    }
                    torch.save(checkpoint_dict, f"{self.result_folder}/checkpoint-{epoch}-{val_score:.4f}.pt")


            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params["epochs"])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params["epochs"], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params["epochs"])
            model_save_interval = self.trainer_params["logging"]["model_save_interval"]
            img_save_interval = self.trainer_params["logging"]["img_save_interval"]

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = "{}/latest".format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params["logging"]["log_image_params_1"],
                                    self.result_log, labels=["train_score"])
                util_save_log_image_with_label(image_prefix, self.trainer_params["logging"]["log_image_params_2"],
                                    self.result_log, labels=["train_loss"])
                if epoch % val_interval == 0:
                    util_save_log_image_with_label(
                        image_prefix, 
                        self.trainer_params["logging"]["log_image_params_1"],
                        self.result_log, 
                        labels=self.plot_names
                    )

            if epoch % model_save_interval == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict,
                    f"{self.result_folder}/checkpoint-{epoch}.pt")

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params["train_episodes"]
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params["train_batch_size"], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info("Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}"
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info("Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}"
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model.forward(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _validation(self):
        self.time_estimator.reset()
        self.model.eval()
        
        # Augmentation
        ###############################################
        aug_factor = self.trainer_params["val_aug_factor"]
        
        val_score_AM = AverageMeter()
        
        for i in range(self.val_num):
            points = self.tmp_solver.points[i]
            dist = points_to_distmat(torch.tensor(points))
            dist = dist.unsqueeze(0) # [1, N, N]
            points_tensor = torch.tensor(points).unsqueeze(0) # [1, N, 2]
            
            if aug_factor > 1:
                problems_batched = dist.repeat(aug_factor, 1, 1)
                points_batched = points_tensor.repeat(aug_factor, 1, 1)
            else:
                problems_batched = dist
                points_batched = points_tensor
            
            with torch.no_grad():
                self.env.load_problems_manual(problems_batched, points_batched)
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state)
                state, reward, done = self.env.pre_step()
                while not done:
                    selected, _ = self.model.forward(state)
                    state, reward, done = self.env.step(selected)
                
                batch_size = 1
                aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
                max_pomo_reward, _ = aug_reward.max(dim=2)
                max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
                aug_score = -max_aug_pomo_reward.float().mean()
                val_score_AM.update(aug_score.item(), 1)
        
        return val_score_AM.avg
