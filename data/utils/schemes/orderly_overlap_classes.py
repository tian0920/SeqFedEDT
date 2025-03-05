from typing import Any, Dict, List, Set
import numpy as np


def orderly_overlap_classes(
        targets: np.ndarray,
        target_indices: np.ndarray,
        label_set: Set[int],
        client_num: int,
        class_num: int,
        overlap_num: int,
        partition: Dict[str, List[np.ndarray]],
        stats: Dict[int, Dict[str, Any]],
):
    """
    按照类别顺序为每个客户端分配数据，并确保相邻客户端有可调节的类别重叠。
    均匀分配该样本到各个拥有该类别的客户端。

    参数:
        targets (np.ndarray): 数据集的标签数组。
        target_indices (np.ndarray): 目标索引。
        label_set (Set[int]): 数据集中所有类别的集合。
        client_num (int): 客户端的总数。
        class_num (int): 每个客户端分配的类别数量。
        overlap_num (int): 相邻客户端之间共享的类别数量。
        partition (Dict[str, List[np.ndarray]]): 每个客户端的数据索引分配结果。
        stats (Dict[int, Dict[str, Any]]): 记录每个客户端的数据分布信息。
    """

    # 创建类别到数据索引的映射
    class_indices = {label: np.where(targets == label)[0] for label in label_set}

    # 对类别进行排序
    sorted_labels = sorted(label_set)
    total_classes = len(sorted_labels)

    # 确保类别重叠数不超过每个客户端的类别数
    overlap_num = min(overlap_num, class_num)

    # 采用滑动窗口方式为每个客户端分配类别
    assigned_labels = [
        sorted_labels[(client_id * (class_num - overlap_num)) % total_classes:
                      (client_id * (class_num - overlap_num)) % total_classes + class_num]
        for client_id in range(client_num)
    ]

    # 根据类别分配数据索引
    for label in sorted_labels:
        # 获取该类别的所有样本索引
        indices = class_indices[label]
        np.random.shuffle(indices)  # 打乱样本顺序

        # 找到所有拥有该类别的客户端
        clients_with_label = [client_id for client_id, classes in enumerate(assigned_labels) if label in classes]

        # 将样本均匀分配给拥有该类别的客户端
        split_indices = np.array_split(indices, len(clients_with_label))
        for client_id, split in zip(clients_with_label, split_indices):
            partition["data_indices"][client_id].extend(split)
            stats[client_id]["y"][label] = len(split)

    # 记录每个客户端拥有的数据量
    for client_id in range(client_num):
        stats[client_id]["x"] = len(partition["data_indices"][client_id])
        partition["data_indices"][client_id] = target_indices[partition["data_indices"][client_id]]

    # 计算每个客户端的数据量统计信息
    sample_counts = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {"mean": sample_counts.mean().item(),
                                   "stddev": sample_counts.std().item()}