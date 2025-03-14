from typing import Any, Dict, List, Set
import numpy as np


def orderly_nested_classes(
        targets: np.ndarray,
        target_indices: np.ndarray,
        label_set: Set[int],
        client_num: int,
        partition: Dict[str, List[np.ndarray]],
        stats: Dict[int, Dict[str, Any]],
):
    """
    俄罗斯套娃式数据划分（按类别样本数排序）：
    - 先计算所有类别的样本数量
    - 选择样本最多的类别作为套娃的第一个类别
    - 按样本数量从多到少的顺序分配类别
    - 同类别数据在拥有该类别的客户端间均分

    参数:
        targets (np.ndarray): 数据集的标签数组。
        target_indices (np.ndarray): 目标索引。
        label_set (Set[int]): 数据集中所有类别的集合。
        client_num (int): 客户端的总数。
        partition (Dict[str, List[np.ndarray]]): 每个客户端的数据索引分配结果。
        stats (Dict[int, Dict[str, Any]]): 记录每个客户端的数据分布信息。
    """

    # 创建类别到数据索引的映射
    class_indices = {label: np.where(targets == label)[0] for label in label_set}

    # **计算每个类别的样本数量，并按降序排序**
    sorted_labels = sorted(label_set, key=lambda lbl: len(class_indices[lbl]), reverse=True)

    # **按照降序排列的类别分配给客户端**
    assigned_labels = [
        set(sorted_labels[: (client_id % len(sorted_labels)) + 1])  # 俄罗斯套娃式类别分配
        for client_id in range(client_num)
    ]

    # 初始化客户端数据分配
    for client_id in range(client_num):
        partition["data_indices"][client_id] = []
        stats[client_id] = {"x": 0, "y": {}}

    # 分配数据索引
    for label in sorted_labels:  # **按照类别样本数量从多到少遍历**
        # 获取该类别的所有样本索引
        indices = class_indices[label]
        np.random.shuffle(indices)  # 打乱样本顺序

        # 找到所有拥有该类别的客户端
        clients_with_label = [client_id for client_id, classes in enumerate(assigned_labels) if label in classes]

        # 将该类别的数据均分给所有拥有该类别的客户端
        split_indices = np.array_split(indices, len(clients_with_label))

        for client_id, split in zip(clients_with_label, split_indices):
            partition["data_indices"][client_id].extend(split.tolist())
            stats[client_id]["y"][label] = len(split)

    # 记录每个客户端的数据量
    for client_id in range(client_num):
        stats[client_id]["x"] = len(partition["data_indices"][client_id])
        partition["data_indices"][client_id] = target_indices[partition["data_indices"][client_id]]

    # 计算每个客户端的数据量统计信息
    sample_counts = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {
        "mean": sample_counts.mean().item(),
        "stddev": sample_counts.std().item()
    }
