from collections import Counter
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient
from src.utils.models import DecoupledModel


def balanced_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    label_counts: torch.Tensor,
):
    logits = logits + (label_counts**gamma).unsqueeze(0).expand(logits.shape).log()
    loss = F.cross_entropy(logits, targets, reduction="mean")
    return loss


class FedRoDClient(FedAvgClient):
    def __init__(self, hypernetwork: torch.nn.Module, **commons):
        # commons["model"] = FedRoDModel(
        commons["model"] = FedRoDModel_new(
            commons["model"], commons["args"].fedrod.eval_per, commons["args"].fedrod.phead, commons["args"].fedrod.freeze_generic_classifier
        )
        super().__init__(**commons)
        self.hypernetwork: torch.nn.Module = None
        self.hyper_optimizer = None
        if self.args.fedrod.hyper:
            self.hypernetwork = hypernetwork.to(self.device)
            self.hyper_optimizer = torch.optim.SGD(
                self.hypernetwork.parameters(), lr=self.args.fedrod.hyper_lr
            )
            self.first_time_selected = True
        self.personal_params_name.extend(
            [key for key, _ in self.model.named_parameters() if "personalized" in key]
        )
        self.clients_label_counts = []
        for indices in self.data_indices:
            counter = Counter(np.array(self.dataset.targets)[indices["train"]])
            self.clients_label_counts.append(
                torch.tensor(
                    [counter.get(i, 0) for i in range(len(self.dataset.classes))],
                    device=self.device,
                )
            )

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.first_time_selected = package["first_time_selected"]
        if self.args.fedrod.hyper and self.first_time_selected:
            self.hypernetwork.load_state_dict(package["hypernet_params"])

    def package(self):
        client_package = super().package()
        if self.args.fedrod.hyper:
            client_package["hypernet_params"] = deepcopy(self.hypernetwork.state_dict())
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        if self.args.fedrod.hyper:
            if self.args.fedrod.hyper and self.first_time_selected:
                self.hypernetwork.to(self.device)
                # if using hypernetwork for generating personalized classifier parameters and client is first-time selected
                classifier_params = self.hypernetwork(
                    self.clients_label_counts[self.client_id]
                    / self.clients_label_counts[self.client_id].sum()
                )
                clf_weight_numel = self.model.generic_model.classifier.weight.numel()
                self.model.personalized_classifier.weight.data = (
                    classifier_params[:clf_weight_numel]
                    .reshape(self.model.personalized_classifier.weight.shape)
                    .detach()
                    .clone()
                )
                self.model.personalized_classifier.bias.data = (
                    classifier_params[clf_weight_numel:]
                    .reshape(self.model.personalized_classifier.bias.shape)
                    .detach()
                    .clone()
                )

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit_g, logit_p = self.model(x)
                loss_p = self.criterion(logit_p, y)

                if self.args.fedrod.bsm and self.args.fedrod.ghead:
                    loss_g = balanced_softmax_loss(
                        logit_g,
                        y,
                        self.args.fedrod.gamma,
                        self.clients_label_counts[self.client_id],
                    )
                    if self.args.fedrod.phead:
                        loss = loss_g + loss_p
                    else:
                        loss = loss_g
                        self.optimizer.zero_grad()
                        loss_g.backward()
                        self.optimizer.step()
                elif self.args.fedrod.ghead:
                    loss_g = self.criterion(logit_g, y)
                    if self.args.fedrod.phead:
                        loss = loss_g + loss_p
                        self.optimizer.zero_grad()
                        loss_g.backward()
                        self.optimizer.step()
                    else:
                        loss = loss_g
                        self.optimizer.zero_grad()
                        loss_g.backward()
                        self.optimizer.step()
                else:
                    loss = loss_p
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # 检查梯度
                # feature_extractor_has_grad = False
                # for name, param in self.model.generic_model.named_parameters():
                #     if 'classifier' not in name and param.grad is not None:
                #         print(f"Feature extractor layer {name} gradient mean: {param.grad.abs().mean().item()}")
                        # feature_extractor_has_grad = True
                # if not feature_extractor_has_grad:
                    # if _ == 4:
                    #     print("No gradients in feature extractor for loss_p")

            # 检查zg和zp
            # zg = self.model.zg
            # zp = self.model.zp
            # if not torch.equal(zg, zp):
            #     diff = (zg - zp).abs().mean().item()
            #     print(f"Average difference between zg and zp: {diff}")

            if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        if self.args.fedrod.hyper and self.first_time_selected:
            # This part has no references on the FedRoD paper
            trained_classifier_params = torch.cat(
                [
                    torch.flatten(self.model.personalized_classifier.weight.data),
                    torch.flatten(self.model.personalized_classifier.bias.data),
                ]
            )
            hyper_loss = F.mse_loss(
                classifier_params, trained_classifier_params, reduction="sum"
            )
            self.hyper_optimizer.zero_grad()
            hyper_loss.backward()
            self.hyper_optimizer.step()
            self.hypernetwork.cpu()


    def finetune(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.test.client.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                if self.args.fedrod.eval_per:
                    _, logit_p = self.model(x)
                    loss = self.criterion(logit_p, y)
                else:
                    logit_g, _ = self.model(x)
                    loss = self.criterion(logit_g, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class FedRoDModel(DecoupledModel):
    def __init__(self, generic_model: DecoupledModel, eval_per):
        super().__init__()
        self.generic_model = generic_model
        self.personalized_classifier = deepcopy(generic_model.classifier)
        self.eval_per = eval_per

    def forward(self, x):
        zg = self.generic_model.get_last_features(x, detach=False)
        zp = self.generic_model.get_last_features(x, detach=True)
        logit_g = self.generic_model.classifier(zg)
        logit_p = self.personalized_classifier(zp)
        if self.training:
            return logit_g, logit_p
        else:
            if self.eval_per:
                return logit_p          # 论文里面的结果：logit_p+ logit_g
            else:
                return logit_g


class FedRoDModel_new(DecoupledModel):
    def __init__(self, generic_model: DecoupledModel, eval_per, phead, freeze_generic_classifier=True):
        super().__init__()
        self.generic_model = generic_model
        self.personalized_classifier = deepcopy(generic_model.classifier)
        self.eval_per = eval_per
        self.phead = phead
        self.freeze_generic_classifier = freeze_generic_classifier
        if self.freeze_generic_classifier:
            for param in self.generic_model.classifier.parameters():
                param.requires_grad = False
            print("Frozen generic_model.classifier parameters")
        else:
            for param in self.generic_model.classifier.parameters():
                param.requires_grad = True
            print("Updated generic_model.classifier parameters")



    def forward(self, x):
        zg = self.generic_model.get_last_features(x, detach=False)
        zp = self.generic_model.get_last_features(x, detach=True)

        logit_g = self.generic_model.classifier(zg)
        logit_p = self.personalized_classifier(zp)
        if self.training:
            return logit_g, logit_p
        else:
            if self.eval_per:
                if self.phead:
                    return logit_p          # 论文里面的结果：logit_p + logit_g
                else:
                    return logit_g
            else:
                return logit_g