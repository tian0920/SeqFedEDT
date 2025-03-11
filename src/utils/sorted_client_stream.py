import json
import os
from typing import List, Dict, Set


def generate_sorted_client_sample_stream(client_sample_stream: List[List[str]], dataset_name: str, data_path: str) -> List[List[str]]:
    """
    使用邻接表和重叠度字典预计算方式，高效地对每轮随机采样的客户端组进行排序优化。

    :param client_sample_stream: 客户端采样的多个组
    :param dataset_name: 数据集名称
    :param data_path: all_stats.json 文件的路径
    :return: 优化排序后的客户端组列表
    """
    import os

    # 读取客户端类别数据
    file_path = os.path.join(data_path, dataset_name, "all_stats.json")
    with open(file_path, "r") as f:
        all_stats = json.load(f)

    client_classes = {
        str(cid): set(map(int, info["y"].keys()))
        for cid, info in all_stats.items()
        if str(cid).isdigit()
    }

    # 预计算重叠度字典（确保键是按数值大小排序的元组）
    overlap_dict = {}
    clients = list(client_classes.keys())
    for i, c1 in enumerate(clients):
        for c2 in clients[i + 1:]:
            key = (str(min(int(c1), int(c2))), str(max(int(c1), int(c2))))
            overlap = len(client_classes[c1] & client_classes[c2])
            overlap_dict[key] = overlap

    # 快速重排序函数（每轮实时调用）
    def reorder_clients_fast(sampled_clients: List[str], start_client: str) -> List[str]:
        unvisited = set(sampled_clients)
        current = start_client
        sorted_clients = [current]
        unvisited.remove(current)

        while unvisited:
            next_client = max(
                unvisited,
                key=lambda c: overlap_dict.get(
                    (str(min(int(current), int(c))), str(max(int(current), int(c)))), 0
                ),
            )
            sorted_clients.append(next_client)
            unvisited.remove(next_client)
            current = next_client

        return sorted_clients

    # 每轮采样进行快速排序
    final_ordered_groups = []
    for idx, group in enumerate(client_sample_stream):
        if idx == 0:
            start_client = group[0]
        else:
            last_client_prev_group = final_ordered_groups[-1][-1]
            start_client = max(
                group,
                key=lambda c: overlap_dict.get(
                    (str(min(int(last_client_prev_group), int(c))), str(max(int(last_client_prev_group), int(c)))), 0
                ),
            )
        sorted_group = reorder_clients_fast(group, start_client)
        final_ordered_groups.append(sorted_group)

    return final_ordered_groups
