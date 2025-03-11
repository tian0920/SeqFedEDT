import os
import re
import matplotlib.pyplot as plt

# 设置数据文件夹路径
root_folder_path = 'D:\\pycharm\\SFL\\out'  # 请替换成你的文件根路径

# 存储实验数据 {method: {variable_type: {variable_value: [accuracies]}}}
experiment_data = {'sfl': {}, 'fedavg': {}}

# 遍历文件夹
for root, dirs, files in os.walk(root_folder_path):
    # 获取当前路径的层级
    path_parts = root.split(os.sep)

    # 确保当前路径符合 out/method/dataset/experiment/
    if len(path_parts) >= 3:
        method = path_parts[-3]  # sfl 或 fedavg
        dataset = path_parts[-2]  # cifar10
        experiment_folder = path_parts[-1]  # data_overlap + client_orderly (0.1, 5, 0, 50)

        # 只处理 cifar10 数据集
        if dataset == 'cifar10' and method in experiment_data:
            # 正则提取 "orderly" 或 "random" 以及括号中的 **第三个数字**
            match = re.search(r"(orderly|random)\s?\((?:[\d\.]+),\s*(?:[\d\.]+),\s*([\d\.]+),", experiment_folder)
            if match:
                variable_type = match.group(1)  # 提取 orderly 或 random
                variable_value = int(float(match.group(2)))  # 确保转换为整数

                # 初始化字典存储每个实验的准确率数据
                if variable_type not in experiment_data[method]:
                    experiment_data[method][variable_type] = {'values': [], 'variable_values': []}

                # 检查是否存在 main.log 文件
                log_file_path = os.path.join(root, 'main.log')
                if 'main.log' in files:
                    with open(log_file_path, 'r') as f:
                        for line in f:
                            # 匹配 "before fine-tuning: 32.19% at epoch 200"
                            match = re.search(r"before fine-tuning: (\d+\.\d+)% at epoch (\d+)", line)
                            if match:
                                accuracy = float(match.group(1))  # 提取准确率
                                experiment_data[method][variable_type]['values'].append(accuracy)
                                experiment_data[method][variable_type]['variable_values'].append(variable_value)

# 绘制每个方法的实验图表
plt.figure(figsize=(10, 6))

# 定义颜色和线型样式
line_styles = {'random': '-', 'orderly': '--'}
colors = {'sfl': 'blue', 'fedavg': 'red'}

# 遍历sfl和fedavg的不同实验
for method, method_data in experiment_data.items():
    for variable_type, data in method_data.items():
        if data['values']:  # 如果该实验有数据
            # 确保 variable_values 只包含整数
            variable_values = [int(v) for v in data['variable_values']]
            accuracies = data['values']

            # 按 variable_values 排序，以确保横坐标有序
            sorted_data = sorted(zip(variable_values, accuracies), key=lambda x: x[0])
            variable_values, accuracies = zip(*sorted_data)

            # 绘制每个变量的曲线
            plt.plot(variable_values, accuracies, label=f'{method}_{variable_type}',
                     linestyle=line_styles[variable_type], color=colors[method])

plt.xlabel('Classes Overlap Number')
plt.ylabel('Accuracy (%)')
plt.title('dirichlet_overlap_classes (0.1, 5, x, 50)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.xticks(sorted(set(variable_values)))  # 只显示整数刻度
plt.tight_layout()
plt.show()
