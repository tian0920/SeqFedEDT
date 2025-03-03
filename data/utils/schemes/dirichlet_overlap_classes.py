from collections import Counter
from typing import Any, Dict, Set
import numpy as np


def dirichlet_overlap_classes(
        targets: np.ndarray,
        target_indices: np.ndarray,
        label_set: Set[int],
        client_num: int,
        class_num: int,
        overlap_num: int,
        alpha: float,
        min_samples_per_client: int,
        partition: Dict[str, Any],
        stats: Dict[int, Dict[str, Any]],
):
    """根据指定的类别数量和客户端之间的类别重叠情况，划分数据集。

    参数:
        targets (np.ndarray): 数据标签数组。
        target_indices (np.ndarray): 标签的索引。如果不是 `--iid`，它将是 `np.arange(len(targets))`。
                                      否则，它包含完整数据集的标签索引。
        label_set (Set[int]): 唯一标签的集合。
        client_num (int): 客户端的数量。
        class_num (int): 每个客户端拥有的类别数量。
        overlap_num (int): 相邻客户端之间的重叠类别数量。
        min_samples_per_client (int): 每个客户端至少应拥有的样本数量。
        partition (Dict[str, Any]): 存储输出数据索引的字典。
        stats (Dict[int, Dict[str, Any]]): 存储每个客户端数据分布的字典。
    """

    # 将每个标签映射到目标数组中的索引
    indices_per_label = {label: np.where(targets == label)[0] for label in label_set}

    # 对标签进行排序，以确保类别分配顺序一致
    sorted_labels = sorted(label_set)
    total_classes = len(sorted_labels)

    # 确保重叠类别数不超过每个客户端的类别数
    overlap_num = min(overlap_num, class_num)

    # 初始化每个客户端的数据索引为空列表
    partition["data_indices"] = [[] for _ in range(client_num)]

    # 为每个客户端分配类别，确保相邻客户端之间有重叠的类别
    assigned_labels = []
    for client_id in range(client_num):
        start_idx = (client_id * (class_num - overlap_num)) % total_classes
        assigned_labels.append(sorted_labels[start_idx:start_idx + class_num])

    # 确保每个客户端的数据样本数量达到最小值
    while True:
        # 清空当前的客户端数据索引
        for client_id in range(client_num):
            partition["data_indices"][client_id] = []

        # 只将数据分配给分配了该类别的客户端
        for label in sorted_labels:
            assigned_clients = [client_id for client_id in range(client_num) if label in assigned_labels[client_id]]
            if not assigned_clients:
                continue

            # 打乱当前标签对应的索引
            np.random.shuffle(indices_per_label[label])

            # 生成Dirichlet分布，用于将数据划分给分配了该类别的客户端
            distribution = np.random.dirichlet(np.repeat(alpha, len(assigned_clients)))

            # 计算累积分布，得到划分位置
            cumulative_distribution = np.cumsum(distribution) * len(indices_per_label[label])
            split_indices_position = cumulative_distribution.astype(int)[:-1]

            # 根据计算得到的划分位置，将数据进行拆分
            split_indices = np.split(indices_per_label[label], split_indices_position)

            # 将拆分的数据分配给相应的客户端
            for i, client_id in enumerate(assigned_clients):
                partition["data_indices"][client_id].extend(split_indices[i])

        # 检查每个客户端的数据样本数量是否满足最小样本数要求
        min_size = min(len(indices) for indices in partition["data_indices"])
        if min_size >= min_samples_per_client:
            break

    # 收集统计信息，并为每个客户端准备输出数据
    for client_id in range(client_num):
        stats[client_id]["x"] = len(targets[partition["data_indices"][client_id]])
        stats[client_id]["y"] = dict(
            Counter(targets[partition["data_indices"][client_id]].tolist())
        )

        # 更新数据索引，使用原始的目标标签索引
        partition["data_indices"][client_id] = target_indices[
            partition["data_indices"][client_id]
        ]

    # 计算每个客户端的样本数量，并更新统计信息
    sample_counts = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {
        "mean": sample_counts.mean().item(),
        "stddev": sample_counts.std().item(),
    }
