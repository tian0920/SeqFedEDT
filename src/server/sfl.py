from omegaconf import DictConfig
from copy import deepcopy
from src.server.fedavg import FedAvgServer
from src.utils.trainer import FLbenchTrainer
from src.utils.constants import MODE
from scipy import stats
import random, math
import numpy as np
from collections import Counter
from collections import defaultdict


class SFLServer(FedAvgServer):
    algorithm_name: str = "SFL"
    return_diff = False

    def __init__(self, args: DictConfig):
        args.mode = "sequential"
        super().__init__(args)

        # (1) normal random shuffling
        # size = max(1, int(self.client_num * self.args.common.join_ratio))
        # epochs = self.args.common.global_epoch
        # clients = self.train_clients
        # lists_per_shuffle = self.client_num // size  # 每次洗牌可划分多少轮
        # self.client_sample_stream = []
        # while len(self.client_sample_stream) < epochs:
        #     shuffled = random.sample(clients, len(clients))  # 洗牌一次
        #     for i in range(lists_per_shuffle):
        #         start = i * size
        #         end = start + size
        #         if len(self.client_sample_stream) < epochs:
        #             self.client_sample_stream.append(shuffled[start:end])
        #         else:
        #             break

        # (2) sobol random shuffling
        # for _ in range(self.args.common.global_epoch):
        #     # 按采样次数对客户端进行分组
        #     clients_by_count = defaultdict(list)
        #     for client in self.train_clients:
        #         clients_by_count[sampling_counts[client]].append(client)
        #
        #         # 按采样次数从低到高排序
        #     sorted_counts = sorted(clients_by_count.keys())
        #
        #     sampled_clients = []
        #     remaining = sample_size
        #
        #     # 优先从采样次数最少的组开始选择
        #     for count in sorted_counts:
        #         clients_in_group = clients_by_count[count]
        #
        #         # 如果当前组的客户端数量足够，使用Sobol序列采样
        #         if len(clients_in_group) >= remaining:
        #             # 使用Sobol序列采样
        #             sampler = stats.qmc.Sobol(d=1, scramble=True)  # 启用扰动以提高采样质量
        #             sample_indices = sampler.random(n=remaining)
        #
        #             # 将0-1范围的随机数转换为客户端索引
        #             indices = (sample_indices * len(clients_in_group)).astype(int)
        #             indices = np.clip(indices, 0, len(clients_in_group) - 1)  # 确保索引在有效范围内
        #
        #             # 选择客户端
        #             group_sampled = [clients_in_group[idx[0]] for idx in indices]
        #             sampled_clients.extend(group_sampled)
        #             break
        #             # 否则选择该组的所有客户端，继续从下一组选择
        #         else:
        #             sampled_clients.extend(clients_in_group)
        #             remaining -= len(clients_in_group)
        #
        #             # 更新采样计数
        #     for client in sampled_clients:
        #         sampling_counts[client] += 1
        #
        #     self.client_sample_stream.append(sampled_clients)  # 存储每轮的采样客户端

        # (3) normal random
        # for _ in range(self.args.common.global_epoch):
        #     sampled_clients = random.sample(
        #         sorted(self.train_clients, key=lambda c: sampling_counts[c]),  # 优先选择被采样次数少的客户端
        #         sample_size
        #     )
        #
        #     # 更新采样计数
        #     for client in sampled_clients:
        #         sampling_counts[client] += 1
        #
        #     self.client_sample_stream.append(sampled_clients)  # 确保采样数据不会为空

        # (4) sobol random
        # for _ in range(self.args.common.global_epoch):
        #     # 对客户端列表排序，优先选择被采样次数少的客户端
        #     sorted_clients = sorted(self.train_clients, key=lambda c: sampling_counts[c])
        #
        #     # 使用 Sobol 序列生成随机数
        #     sampler = stats.qmc.Sobol(d=1, scramble=True)  # 使用Sobol序列，并启用扰动
        #     sample_indices = sampler.random(n=sample_size)
        #
        #     # 将 0-1 范围的随机数转换为客户端索引
        #     indices = (sample_indices * len(sorted_clients)).astype(int)
        #     indices = np.clip(indices, 0, len(sorted_clients) - 1)  # 确保索引在有效范围内
        #
        #     # 选择客户端
        #     sampled_clients = [sorted_clients[idx[0]] for idx in indices]
        #
        #     # 更新采样计数
        #     for client in sampled_clients:
        #         sampling_counts[client] += 1
        #
        #     self.client_sample_stream.append(sampled_clients)

        # 获取当前脚本的路径
        # current_path = Path(__file__).resolve()
        # self.client_sample_stream = multi_overlap_sorted_client_sample_stream(
        #     self.client_sample_stream,
        #     dataset_name=args.dataset.name,
        #     data_path=current_path.parents[2] / "data",   # 获取比当前目录高两级的路径
        #     n=5,
        # )

    def init_trainer(self, **extras):
        """Initiate the FL-bench trainier that responsible to client training.

        Args:
            `extras`: Arguments of `self.client_cls.__init__()` that NOT included in
        `[model, args, optimizer_cls, lr_scheduler_cls, dataset, data_indices,
        device, return_diff]`.
        """
        if self.args.mode == "sequential":
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=self.client_cls,
                mode=MODE.SEQUENTIAL,
                num_workers=0,
                init_args=dict(
                    model=deepcopy(self.model),
                    optimizer_cls=self.get_client_optimizer_cls(),
                    lr_scheduler_cls=self.get_client_lr_scheduler_cls(),
                    args=self.args,
                    dataset=self.dataset,
                    data_indices=self.client_data_indices,
                    device=self.device,
                    return_diff=self.return_diff,
                    **extras,
                ),
            )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""

        self.public_model_params = self.trainer.train()
        self.model.load_state_dict(self.public_model_params, strict=False)