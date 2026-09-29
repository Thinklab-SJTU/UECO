
import torch
from torch import Tensor
import numpy as np
import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel_UECO import UECOLCModel as Model

from utils.utils import *
from CVRProblemDef import *
from tqdm import tqdm, trange
from ml4co_kit import *
from tabulate import tabulate
import time


class CVRPTester:
    def __init__(
        self,
        env_params: dict,
        model_params: dict,
        tester_params: dict
    ):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params["use_cuda"]
        if USE_CUDA:
            cuda_device_num = self.tester_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = self.tester_params["model_load"]
        checkpoint = torch.load(model_load, map_location=device)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except:
            self.model.load_state_dict(checkpoint, strict=False)

        # utility
        self.time_estimator = TimeEstimator()
        self.tmp_solver = CVRPSolver()
        self.tmp_solver.from_txt(file_path=tester_params["test_file"], ref=True, show_time=True)
        self.use_ls = tester_params["use_ls"]

        self.num_instances = min(tester_params.get("num_instances", self.tmp_solver.ref_tours.shape[0]),
                                    self.tmp_solver.ref_tours.shape[0])
        if self.num_instances < self.tmp_solver.ref_tours.shape[0]:
            self.tmp_solver.ori_points = self.tmp_solver.points[:self.num_instances, :, :]
            self.tmp_solver.points = self.tmp_solver.points[:self.num_instances, :, :]
            self.tmp_solver.depots = self.tmp_solver.depots[:self.num_instances, :]
            self.tmp_solver.demands = self.tmp_solver.demands[:self.num_instances, :]
            self.tmp_solver.ref_tours = self.tmp_solver.ref_tours[:self.num_instances, :]

    def run(self):
        tours_list = []
        ori_objs = []
        tot_time = 0
        tbar = trange(self.num_instances)
        for idx in tbar:
            depots_tensor = torch.tensor(self.tmp_solver.depots[idx]).unsqueeze(0)
            points_tensor = torch.tensor(self.tmp_solver.points[idx])
            demands_tensor = torch.tensor(self.tmp_solver.demands[idx] / self.tmp_solver.capacities[idx])
            t0 = time.time()
            obj, tour = self._test_one_instance(
                depots_tensor,
                points_tensor,
                demands_tensor
            )
            t1 = time.time()
            ori_objs.append(obj)
            t2, t3 = 0, 0
            if self.use_ls:
                t2 = time.time()
                tour = cvrp_classic_local_search(
                    init_tour=tour, depot=self.tmp_solver.depots[idx],
                    points=self.tmp_solver.points[idx], 
                    demands=self.tmp_solver.demands[idx] / self.tmp_solver.capacities[idx]
                ).reshape(-1).tolist()
                t3 = time.time()
            tours_list.append(tour)
            tot_time += (t1 - t0 + t3 - t2)
        # solving_t = tbar.format_dict['elapsed']
        self.tmp_solver.from_data(tours=tours_list, ref=False)
        metrics = self.tmp_solver.evaluate(calculate_gap=True)
        
        if self.num_instances == 1:  # single-instance evaluation
            print("[Test Instance]", self.tester_params["test_file"].split("/")[-1])
            # print(f"[Opt] {metrics[1]}")
            print(f"[Obj] {metrics[0]}")
            print(f"[Gap] {metrics[2]}")
            # print(f"[Time] {tot_time / self.num_instances}")
            # print("\n")
            return metrics, tot_time
        else:
            print("[Test Result] Original Obj:", sum(ori_objs) / len(ori_objs))
            print("[Test Result]", metrics)
            avg_time = tot_time / self.num_instances
            print_results(*metrics, avg_time)


    def _test_one_instance(self, depots: Tensor, points: Tensor, demands: Tensor):

        # Augmentation
        ###############################################
        aug_factor = self.tester_params["aug_factor"]
        depots_batched = depots.unsqueeze(0)
        points_batched = points.unsqueeze(0)
        demands_batched = demands.unsqueeze(0)
        if aug_factor > 1:
            depots_batched = augment_data_hybrid(depots_batched, aug_factor)
            points_batched = augment_data_hybrid(points_batched, aug_factor)
            demands_batched = demands_batched.repeat(aug_factor, 1)
        # elif aug_factor > 1:
        #     raise NotImplementedError()

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(depots_batched, points_batched, demands_batched)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        batch_size = 1
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, best_pomo_idx = aug_reward.max(dim=2)  # get best results from pomo, shape: (augmentation, batch)
        
        best_pomo_tours = self.env.selected_node_list[torch.arange(aug_factor), best_pomo_idx.view(-1), :]

        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, best_aug_idx = max_pomo_reward.max(dim=0)  # get best results from augmentation, shape: (batch,)

        best_aug_tour = best_pomo_tours[best_aug_idx.view(-1), :]

        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        best_tour = best_aug_tour.squeeze().tolist()
        best_obj = aug_score.item()

        return best_obj, best_tour

def print_results(obj: float, ref_obj: float, gap: float, gap_std: float, avg_time: float):
    table = [[obj, ref_obj, f"{gap:.5f}%", gap_std, avg_time]]
    headers = ["obj", "ref_obj", "gap", "gap_std", "time (s/inst.)"]
    print("[TEST INFO] Test result:")
    print(tabulate(table, headers=headers, floatfmt=".5f"))
