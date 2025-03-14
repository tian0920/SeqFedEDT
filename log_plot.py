import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 日志文件存放路径
log_dir = "D:\\pycharm\\SFL\\experiment_logs"  # 修改为你的log文件夹路径

# 匹配日志文件名的正则表达式（支持 `+` 符号的 method）
filename_pattern = re.compile(r"(?P<method>[\w+]+)_(?P<dataset>\w+)_cc(?P<cc>\d+)_oc(?P<oc>\d+)\.log")

# 匹配 accuracy 记录的正则表达式
accuracy_pattern = re.compile(r"before fine-tuning:\s*([\d.]+)%")

# 存储数据结构
data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# 遍历日志文件
for filename in os.listdir(log_dir):
    match = filename_pattern.match(filename)
    if match:
        method = match.group("method")  # 例如：sfl+random
        dataset = match.group("dataset")
        cc = int(match.group("cc"))
        oc = int(match.group("oc"))

        file_path = os.path.join(log_dir, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        acc_match = accuracy_pattern.search(content)
        if acc_match:
            accuracy = float(acc_match.group(1))
            data[cc][oc][(method, dataset)] = accuracy

# 按 cc 分组绘制子图
cc_values = sorted(data.keys())
fig, axes = plt.subplots(len(cc_values), 1, figsize=(8, 5 * len(cc_values)))

if len(cc_values) == 1:
    axes = [axes]  # 确保即使只有一个 cc 也能正确索引

for ax, cc in zip(axes, cc_values):
    oc_values = sorted(data[cc].keys())  # 当前 cc 下所有的 oc 作为横轴

    # 收集所有方法和数据集的唯一值
    methods = set()
    datasets = set()

    for oc in oc_values:
        for (method, dataset) in data[cc][oc]:
            methods.add(method)
            datasets.add(dataset)

    # 颜色和线型映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))  # 数据集用颜色
    linestyles = ["-", "--", "-.", ":"]  # 方法用线型
    dataset_color_map = {ds: color for ds, color in zip(datasets, colors)}
    method_linestyle_map = {m: ls for m, ls in zip(methods, linestyles)}

    for key in data[cc][oc_values[0]]:  # 遍历 (method, dataset) 组合
        method, dataset = key
        color = dataset_color_map.get(dataset, "black")  # 颜色按数据集
        linestyle = method_linestyle_map.get(method, "-")  # 线型按方法

        acc_values = [data[cc][oc].get(key, np.nan) for oc in oc_values]  # 仅当前 cc 的 oc 轴
        ax.plot(oc_values, acc_values, linestyle=linestyle, color=color, label=f"{method}-{dataset}")

    ax.set_title(f"cc = {cc}")
    # ax.set_xlabel("oc (overlap categories)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(oc_values)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
