import os
import re
import matplotlib.pyplot as plt


def extract_data_from_log(file_path):
    """ 从日志文件提取 epoch 和 accuracy """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        matches = re.finditer(r'"(\d+)": \{.*?"accuracy": "(\d+\.\d+)%', log_content, re.DOTALL)
        for match in matches:
            epoch = int(match.group(1))
            accuracy = float(match.group(2))
            data[epoch] = accuracy
    return data


def plot_accuracy(sfl_dir):
    """
    以 epoch 为横轴，accuracy 为纵轴
    每个 .log 文件作为一个图例
    cifar10 和 cifar100 分别生成一个子图
    """
    datasets = ['cifar10', 'cifar100']
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # 颜色列表
    linestyles = ['-', ]  # 线条样式  '--', ':', '-.'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 创建两个子图

    for dataset_idx, dataset in enumerate(datasets):
        ax = axes[dataset_idx]  # 选择当前子图
        dataset_path = os.path.join(sfl_dir, dataset)

        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} 不存在，跳过该数据集。")
            continue

        log_files = [f for f in os.listdir(dataset_path) if f.endswith(".log")]

        for idx, log_file in enumerate(log_files):
            log_path = os.path.join(dataset_path, log_file)
            data = extract_data_from_log(log_path)

            if not data:
                print(f"Warning: {log_path} 没有提取到数据，跳过该文件。")
                continue

            epochs = sorted(data.keys())
            accuracies = [data[e] for e in epochs]

            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            ax.plot(epochs, accuracies, color=color, linestyle=linestyle, label=log_file[:-4])  # 去掉 `.log`

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid()

    plt.tight_layout()  # 自动调整子图间距
    plt.show()


# 调用函数，传入 nest 目录路径
sfl_directory = "D:\\pycharm\\SFL\\experiment_logs\\nest"  # 请修改为你的 SFL 文件夹路径
plot_accuracy(sfl_directory)
