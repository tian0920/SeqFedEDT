from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.client.fedrod import FedRoDClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES

from pathlib import Path
import os
from src.utils.logger import Logger
from rich.console import Console


class FedRoDServer(FedAvgServer):
    algorithm_name: str = "FedRoD"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedRoDClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--gamma", type=float, default=1)
        parser.add_argument("--hyper", type=int, default=0)
        parser.add_argument("--hyper_lr", type=float, default=0.1)
        parser.add_argument("--hyper_hidden_dim", type=int, default=32)
        parser.add_argument("--eval_per", type=int, default=1)
        parser.add_argument("--bsm", type=bool, default=True)
        return parser.parse_args(args_list)

    def __init__(
        self, args: DictConfig):
        super().__init__(args, False)
        self.hyper_params_dict = None
        self.hypernetwork: nn.Module = None
        if self.args.fedrod.hyper:
            output_dim = (
                self.model.classifier.weight.numel()
                + self.model.classifier.bias.numel()
            )
            input_dim = NUM_CLASSES[self.args.dataset.name]
            self.hypernetwork = HyperNetwork(
                input_dim, self.args.fedrod.hyper_hidden_dim, output_dim
            )
            self.hyper_params_dict = OrderedDict()
            for key, param in self.hypernetwork.named_parameters():
                self.hyper_params_dict[key] = param.data.clone()
        self.first_time_selected = [True for _ in self.train_clients]
        self.init_trainer(hypernetwork=self.hypernetwork)

        bsm_name = "BSM" if self.args.fedrod.bsm else "NoBSM"
        G_name = "G" if self.args.fedrod.ghead else "NoG"
        P_name = "P" if self.args.fedrod.phead else "NoP"
        hyper_name = "Hyper" if self.args.fedrod.hyper else "NoHyper"

        # Directly combine into a complete output directory
        self.output_dir = (
                Path("./out")
                / self.args.method
                / self.args.dataset.name
                / f"{bsm_name}_{G_name}_{P_name}_{hyper_name}"
        )

        print(self.output_dir)

        if not os.path.isdir(self.output_dir) and (
                self.args.common.save_log
                or self.args.common.save_learning_curve_plot
                or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        stdout = Console(log_path=False, log_time=False, soft_wrap=True, tab_size=4)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.common.save_log,
            logfile_path=self.output_dir / "main.log",
        )

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["first_time_selected"] = self.first_time_selected[client_id]
        server_package["hypernet_params"] = self.hyper_params_dict
        return server_package

    @torch.no_grad()
    def aggregate_client_updates(self, client_packages: dict[int, dict[str, Any]]):
        for client_id in client_packages.keys():
            self.first_time_selected[client_id] = False
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)
        clients_model_params_list = [
            list(package["regular_model_params"].values())
            for package in client_packages.values()
        ]
        for old_param, zipped_new_param in zip(
            self.public_model_params.values(), zip(*clients_model_params_list)
        ):
            old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                dim=-1
            )

        if self.args.fedrod.hyper:
            clients_hypernet_params_list = [
                list(package["hypernet_params"].values())
                for package in client_packages.values()
            ]
            for old_param, zipped_new_param in zip(
                self.hyper_params_dict.values(), zip(*clients_hypernet_params_list)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )


class HyperNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)
