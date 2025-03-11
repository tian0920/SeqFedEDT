from collections import Counter
import numpy as np
from src.client.fedavg import FedAvgClient


class SFLClient(FedAvgClient):
    clients_label_counts = {}

    def __init__(self):
        super().__init__()
        for client_id, indices in enumerate(self.data_indices):
            counter = Counter(np.array(self.dataset.targets)[indices["train"]])
            SFLClient.clients_label_counts[client_id] = dict(counter)