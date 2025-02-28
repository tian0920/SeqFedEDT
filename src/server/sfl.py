from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig
from src.server.fedavg import FedAvgServer

class SFLServer(FedAvgServer):
    algorithm_name: str = "SFL"
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        args.mode = "sequential"
        super().__init__(args)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""

        self.public_model_params = self.trainer.train()
        self.model.load_state_dict(self.public_model_params, strict=False)