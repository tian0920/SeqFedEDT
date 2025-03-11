import os
import re
import matplotlib.pyplot as plt


def extract_data_from_log(file_path):
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
    datasets = ['cifar10', 'cifar100']
    colors = {'cifar10': 'blue', 'cifar100': 'red'}
    linestyles = ['-', '--', ':', '-.']  # 多种线型区分 y 值

    plt.figure(figsize=(10, 6))

    title_parts = []
    for dataset in datasets:
        centralized_dirs = [d for d in os.listdir(os.path.join(sfl_dir, dataset)) if d.startswith("centralized_test")]
        parsed_dirs = []
        for centralized_dir in centralized_dirs:
            match = re.search(r'centralized_test \((\d+),\s*(\d+)\)', centralized_dir)
            if match:
                x, y = map(int, match.groups())
                parsed_dirs.append((x, y, centralized_dir))

        parsed_dirs.sort(key=lambda item: item[1])  # 按 y 进行排序

        for idx, (x, y, centralized_dir) in enumerate(parsed_dirs):
            title_parts.append(f"{dataset} ({x},{y})")
            log_path = os.path.join(sfl_dir, dataset, centralized_dir, 'main.log')
            if os.path.exists(log_path):
                data = extract_data_from_log(log_path)
                epochs = sorted(data.keys())
                accuracies = [data[e] for e in epochs]
                label = f"{dataset} (y={y})"
                plt.plot(epochs, accuracies, color=colors[dataset], linestyle=linestyles[idx % len(linestyles)],
                         label=label)
            else:
                print(f"Warning: {log_path} does not exist.")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(' vs '.join(title_parts))
    plt.legend()
    plt.grid()
    plt.show()


# 调用函数，传入 sfl 文件夹路径
sfl_directory = "D:\\virtualLive\\SFL\\out\\sfl"  # 请修改为你的sfl文件夹路径
plot_accuracy(sfl_directory)
