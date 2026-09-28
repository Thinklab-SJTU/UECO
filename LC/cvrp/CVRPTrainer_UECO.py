import os
import sys
import torch
from torch import Tensor
from logging import getLogger
from importlib import import_module

from CVRPEnv import CVRPEnv as Env
from CVRPModel_UECO import UECOLCModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from ml4co_kit import *


class CVRPTrainer:
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
        self.task = env_params.get("task", "CVRP")

        # Restore
        self.start_epoch = 1
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_fullname = model_load["path"]
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if not model_load.get("only_weight", False):
                self.start_epoch = 1 + model_load["epoch"]
                self.result_log.set_raw_data(checkpoint["result_log"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # keep scheduler in sync with restored epoch
                self.scheduler.last_epoch = model_load["epoch"] - 1
            self.logger.info("Saved Model Loaded!")
            self.plot_baseline = False

        # utility
        self.time_estimator = TimeEstimator()

        # validation data loader
        if self.task == "CVRP":
            self.tmp_solver = CVRPSolver()
        self.tmp_solver.from_txt(self.trainer_params["val_file"], ref=True, show_time=True)
        self.val_num = trainer_params["val_num"]
        self.best_val_score = 1e10

    def run(self):
        # self._validation()

        self.time_estimator.reset(self.start_epoch)
        val_interval = self.trainer_params["val_interval"]
        model_save_interval = self.trainer_params["logging"]["model_save_interval"]

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            self.logger.info("=================================================================")

            # Param scheduler
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, train_score)
            self.result_log.append("train_loss", epoch, train_loss)

            # Validation
            if epoch % val_interval == 0:
                val_score = self._validation()
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

            # Save checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params["epochs"])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params["epochs"], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params["epochs"])
            model_save_interval = self.trainer_params["logging"]["model_save_interval"]
            img_save_interval = self.trainer_params["logging"]["img_save_interval"]

            # Plot
            if epoch > 1:
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
                        labels=["val_score"]
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

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, "{}/checkpoint-{}.pt".format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch: int):

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
        self.logger.info(f"Epoch {epoch:3d}: Train  Score: {score_AM.avg:.4f},  Loss: {loss_AM.avg:.4f}")

        return score_AM.avg, loss_AM.avg
    
    def _validation(self) -> float:
        objs = []
        aug_factor = 1
        batch_size = 1
        self.model.eval()

        for idx in range(self.val_num):
            if self.task == "CVRP":
                depots = torch.tensor(self.tmp_solver.depots[idx]).unsqueeze(0) 
                points = torch.tensor(self.tmp_solver.points[idx]) 
                demands = torch.tensor(self.tmp_solver.demands[idx] / self.tmp_solver.capacities[idx])
            else:
                raise NotImplementedError()
            depots_batched = depots.unsqueeze(0)
            points_batched = points.unsqueeze(0)
            demands_batched = demands.unsqueeze(0)

            with torch.no_grad():
                self.env.load_problems_manual(depots_batched, points_batched, demands_batched)
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state)

                state, reward, done = self.env.pre_step()
                while not done:
                    selected, _ = self.model.forward(state)
                    state, reward, done = self.env.step(selected)
                aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
                max_pomo_reward, best_pomo_idx = aug_reward.max(dim=2)  # get best results from pomo
                best_pomo_tours = self.env.selected_node_list[torch.arange(aug_factor), best_pomo_idx.view(-1), :]
                max_aug_pomo_reward, best_aug_idx = max_pomo_reward.max(dim=0)  # get best results from augmentation
                aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

                objs.append(aug_score.item())

        self.model.train()
        avg_val_obj = sum(objs) / len(objs)
        return avg_val_obj

    def _train_one_batch(self, batch_size: int):
        # Prep
        self.model.train()
        self.env.load_problems(batch_size, aug_factor=1)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        batch_size, pomo_size = self.env.batch_size, self.env.pomo_size
        prob_list = torch.zeros(size=(batch_size, pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model.forward(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        # Score
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean: Tensor = -max_pomo_reward.float().mean()

        # Step
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return score_mean.item(), loss_mean.item()
