import os
import re


def extract_log_info(log_file_path):
    """
    从日志文件中提取 method, dataset name, alpha 和 psfl+diff+fisher 中的 fisher_threshold 的值
    """
    method = None
    dataset = None
    alpha = None
    fisher_threshold = None  # 从 'psfl+diff+fisher' 中提取的值
    inside_dataset = False  # 用于标记是否处于 'dataset' 字典内
    inside_feddpa = False  # 用于标记是否处于 'psfl+diff+fisher' 字典内
    inside_psfl = False
    ig_ratio = None
    score = None

    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # 提取 method
                if "'method': " in line:
                    method_match = re.search(r"'method': '([^']+)'", line)
                    if method_match:
                        method = method_match.group(1)

                # 判断是否进入 'dataset' 字典
                if "'dataset': {" in line:
                    inside_dataset = True
                    dataset = None
                    alpha = None

                # 如果已经进入了 dataset 字典并且读取到了 name 和 alpha 信息
                if inside_dataset:
                    if "'name': '" in line:
                        dataset_match = re.search(r"'name': '([^']+)'", line)
                        if dataset_match:
                            dataset = dataset_match.group(1)
                    if "'alpha': " in line:
                        alpha_match = re.search(r"'alpha': ([0-9.]+)", line)
                        if alpha_match:
                            alpha = float(alpha_match.group(1))
                    # 当读取到 name 和 alpha 信息后，结束 dataset 字典的解析
                    if dataset and alpha is not None:
                        inside_dataset = False  # 退出 dataset 字典的解析

                # 进入 'psfl+diff+fisher' 字典并提取 fisher_threshold
                if "'feddpa': {" in line:
                    inside_feddpa = True

                if inside_feddpa:
                    if "'fisher_threshold': " in line:
                        fisher_threshold_match = re.search(r"'fisher_threshold': ([0-9.]+)", line)
                        if fisher_threshold_match:
                            fisher_threshold = float(fisher_threshold_match.group(1))
                    # 当读取到 fisher_threshold 后，结束 psfl+diff+fisher 字典的解析
                    if fisher_threshold is not None:
                        inside_feddpa = False

                # 进入 'psfl+diff' 字典并提取 ig_ratio
                if "'psfl': {" in line:
                    inside_psfl = True

                if inside_psfl:
                    if "'ig_ratio': " in line:
                        ig_ratio_match = re.search(r"'ig_ratio': ([0-9.]+)", line)
                        if ig_ratio_match:
                            ig_ratio = float(ig_ratio_match.group(1))
                    if "'score': " in line:
                        score_match = re.search(r"'score': '([^']+)'", line)
                        if score_match:
                            score = score_match.group(1)
                    # 当读取到 ig_ratio 后，结束 psfl+diff 字典的解析
                    if ig_ratio and score is not None:
                        inside_psfl = False

                # 如果 method, dataset, alpha 都找到了，可以提前退出循环
                if (method is not None and dataset is not None and alpha is not None and fisher_threshold is not None) or (
                        method is not None and dataset is not None and alpha is not None and ig_ratio is not None and score is not None):
                    break

    except Exception as e:
        print(f"处理文件 {log_file_path} 时发生错误: {e}")
    return method, dataset, alpha, fisher_threshold, ig_ratio, score


def rename_folder_based_on_log(folder_path):
    """
    遍历指定文件夹，查找符合条件的 log 文件并重命名文件夹
    """
    for root, dirs, files in os.walk(folder_path):
        # 只遍历当前目录的第二层文件夹
        # 遍历子目录的子文件夹（即时间命名的文件夹）
        for subfolder in dirs:
            subfolder_path = os.path.join(root, subfolder)
            for subsubfolder in os.listdir(subfolder_path):
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                if os.path.isdir(subsubfolder_path):
                    # 继续检查该子文件夹中的 log 文件
                    for file in os.listdir(subsubfolder_path):
                        if file.endswith(".log"):
                            log_file_path = os.path.join(subsubfolder_path, file)
                            print(log_file_path)
                            method, dataset, alpha, fisher_threshold, ig_ratio, score = extract_log_info(log_file_path)
                            if method and dataset and alpha is not None:
                                # 生成新的文件夹名称
                                new_folder_name = f"{method}_{dataset}_{alpha}"

                                # 如果 fisher_threshold 存在，则将其添加到新文件夹名称
                                if fisher_threshold is not None:
                                    new_folder_name += f"_{fisher_threshold}"

                                if ig_ratio is not None:
                                    new_folder_name += f"_{ig_ratio}"
                                    if score is not None:
                                        new_folder_name += f"_{score}"

                                parent_folder_path = os.path.dirname(subsubfolder_path)
                                new_folder_path = os.path.join(parent_folder_path, new_folder_name)

                                # 检查目标文件夹是否已经存在，如果存在则添加后缀
                                counter = 2
                                while os.path.exists(new_folder_path):
                                    new_folder_path = os.path.join(parent_folder_path, f"{new_folder_name} ({counter})")
                                    counter += 1

                                # 重命名文件夹
                                try:
                                    print(f"重命名文件夹 {subsubfolder_path} 为 {new_folder_path}")
                                    os.rename(subsubfolder_path, new_folder_path)
                                except Exception as e:
                                    print(f"重命名文件夹 {subsubfolder_path} 时发生错误: {e}")
                                break  # 一旦找到匹配的 log 文件，停止遍历该文件夹


if __name__ == "__main__":
    # 设置根文件夹路径
    root_folder = "out/feddpa"  # 修改为你的根文件夹路径
    rename_folder_based_on_log(root_folder)
