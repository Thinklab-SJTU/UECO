
import torch
import numpy as np
from copy import deepcopy
from logging import getLogger
from torch import Tensor
from TSPEnv import TSPEnv as Env
from TSPModel_UECO import UECOLCModel as Model
from utils.utils import get_result_folder, TimeEstimator
from typing import Tuple
from tqdm import trange
from ml4co_kit import *
from tabulate import tabulate
import time


class TSPTester:
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
        self.task = env_params["test_type"][0]

        # Restore
        model_load = self.tester_params["model_load"]
        checkpoint = torch.load(model_load, map_location=device)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except:
            self.model.load_state_dict(checkpoint, strict=False)

        # utility
        self.time_estimator = TimeEstimator()
        self.tmp_solver = TSPSolver()
        self.tmp_solver.from_txt(file_path=tester_params["test_file"], ref=True, show_time=True)
        self.use_2opt = tester_params["use_2opt"]

        self.num_instances = min(tester_params.get("num_instances", self.tmp_solver.ref_tours.shape[0]),
                                    self.tmp_solver.ref_tours.shape[0])
        if self.num_instances < self.tmp_solver.ref_tours.shape[0]:
            self.tmp_solver.ori_points = self.tmp_solver.points[:self.num_instances, :, :]
            self.tmp_solver.points = self.tmp_solver.points[:self.num_instances, :, :]
            self.tmp_solver.ref_tours = self.tmp_solver.ref_tours[:self.num_instances, :]

    def run_single_inst(self, points, dist):
        # for idx in trange(self.num_instances):
        obj, tour = self._test_one_instance(dist, to_tensor(points))
        tour_back = tour.reshape(-1)
        if self.use_2opt:
            tour_back = tsp_mcts_local_search(
                init_tours=tour_back,
                heatmap=np.zeros((self.env.node_cnt, self.env.node_cnt)),
                points=points,
                time_limit=0.0, # set time limit as 0.0 is equal to 2opt
                type_2opt=2
            )
        return obj, to_tensor(tour)

    def run(self):
        np_tours = []
        ori_objs = []
        tot_time = 0
        for idx in trange(self.num_instances):
            points = self.tmp_solver.points[idx]
            dist_tensor = points_to_distmat(torch.tensor(points))
            points_tensor = torch.tensor(points)
            
            t0 = time.time()
            obj, tour = self._test_one_instance(dist_tensor, points_tensor)
            t1 = time.time()
            tour_back = tour.reshape(-1)
            ori_objs.append(obj)
            t2, t3 = 0, 0
            if self.use_2opt:
                t2 = time.time()
                tour_back = tsp_mcts_local_search(
                    init_tours=tour_back,
                    heatmap=np.zeros((self.env.node_cnt, self.env.node_cnt)),
                    points=self.tmp_solver.points[idx],
                    time_limit=0.0, # set time limit as 0.0 is equal to 2opt
                    type_2opt=2
                )
                t3 = time.time()

            tour_back = np.expand_dims(tour_back, axis=0)
            if tour_back.shape[-1] == self.env.node_cnt:
                assert sorted(tour_back[0].tolist()) == [i for i in range(self.env.node_cnt)]
            else:
                assert sorted(tour_back[0].tolist()[:-1]) == [i for i in range(self.env.node_cnt)]
            np_tours.append(tour_back)
            tot_time += (t1 - t0 + t3 - t2)
        # print("origianl length: ", sum(ori_objs) / len(ori_objs))
        np_tours = np.concatenate(np_tours, axis=0)
        if np_tours.shape[-1] == self.env.node_cnt:
            np_tours = np.concatenate((np_tours, np_tours[:, :1]), axis=1)
        self.tmp_solver.tours = np_tours
        # print(self.tmp_solver.evaluate(calculate_gap=True))
        avg_time = tot_time / self.num_instances
        print_results(*self.tmp_solver.evaluate(calculate_gap=True), avg_time)
        

    def _test_one_instance(self, dist: Tensor, points: Tensor) -> Tuple[float, np.ndarray]:
        dist = dist.unsqueeze(0) # [1, N, N]
        points = points.unsqueeze(0) # [1, N, 2]
        # Augmentation
        ###############################################
        if self.tester_params["augmentation_enable"]:
            aug_factor = self.tester_params["aug_factor"]
            problems_batched = dist.repeat(aug_factor, 1, 1)
            points_batched = points.repeat(aug_factor, 1, 1)
        else:
            aug_factor = 1
            problems_batched = dist
            points_batched = points
        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(deepcopy(problems_batched), deepcopy(points_batched))
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model.forward(state)# shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
            # Return
            ###############################################
            batch_size = 1
            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size) # shape: (augmentation, batch, pomo)
            max_pomo_reward, best_pomo_idx = aug_reward.max(dim=2)  # get best results from pomo, shape: (augmentation, batch)
            best_pomo_tours = self.env.selected_node_list[torch.arange(aug_factor), best_pomo_idx.view(-1), :]
            no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
            max_aug_pomo_reward, best_aug_idx = max_pomo_reward.max(dim=0)  # get best results from augmentation, shape: (batch,)
            best_aug_tour = best_pomo_tours[best_aug_idx.view(-1), :]
            aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
            best_obj = aug_score.item()
            best_tour = best_aug_tour.cpu().numpy()
            return best_obj, best_tour
        
def print_results(obj: float, ref_obj: float, gap: float, gap_std: float, avg_time: float):
    table = [[obj, ref_obj, f"{gap:.5f}%", gap_std, avg_time]]
    headers = ["obj", "ref_obj", "gap", "gap_std", "time (s/inst.)"]
    print("[TEST INFO] Test result:")
    print(tabulate(table, headers=headers, floatfmt=".5f"))
