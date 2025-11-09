import math
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk


def simplify_label(eval_name):
    """将文件名简化为更直观的label格式，利用文件命名的规范性进行简化提取"""
    # 获取文件名（去掉路径和扩展名）
    file_name = eval_name.split("/")[-1].split(".")[0]

    # 基于规范的文件名格式进行简化处理
    # 标准格式：dataset_type_{train|test}[_checkpoint-number_...]
    parts = file_name.split("_")

    # 提取数据集类型和用途（通常是前两部分）
    dataset_type = parts[0]
    data_purpose = parts[1] if len(parts) >= 2 else ""

    # 判断是否是checkpoint文件
    if len(parts) >= 3 and parts[2].startswith("checkpoint"):
        # 提取checkpoint数字
        checkpoint_part = parts[2]
        checkpoint_num = (
            checkpoint_part.split("-")[1]
            if "-" in checkpoint_part
            else checkpoint_part[10:]
        )
        return f"{dataset_type}_{data_purpose}_{checkpoint_num}"
    else:
        # 基础数据集
        return f"{dataset_type}_{data_purpose}_base"


def find_matched_directory(eval_name: str) -> str:
    """查找匹配的 ground_truth 数据目录"""
    data_dir = Path("./data")
    test_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]

    for test_dir in test_dirs:
        if eval_name.split("/")[-1].startswith(test_dir):
            return test_dir

    raise ValueError(f"无法为 {eval_name} 找到匹配的ground_truth目录")


def load_data(eval_name: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """加载并返回 ground_truth、输出数据以及每个样本的尝试次数"""
    matched_dir = find_matched_directory(eval_name)
    ground_truth: pd.DataFrame = load_from_disk(
        Path("./data") / matched_dir
    ).to_pandas()  # type: ignore
    output_data = pd.read_parquet(eval_name)

    n_tries = output_data.shape[0] // ground_truth.shape[0]
    assert ground_truth.shape[0] * n_tries == output_data.shape[0], "数据量不匹配"

    return ground_truth, output_data, n_tries


def calculate_metrics(
    output_data: pd.DataFrame, ground_truth: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    """计算模型预测的关键指标"""
    outputs = output_data["output"].str.split().str[-1]
    true_labels = ground_truth.loc[output_data["index"], "sentiment"].values
    correct_per_row = outputs == true_labels  # 每行是否预测正确

    # 计算每个题目的正确样本数
    c_i_series = correct_per_row.groupby(output_data["index"]).sum()
    num_problems = len(c_i_series)

    return outputs, correct_per_row, c_i_series, num_problems


def calculate_pass_k(c_i_series: pd.Series, n_tries: int) -> list[float]:
    """计算不同 k 值下的 pass@k 指标"""
    pass_k_list = []

    for k in range(1, n_tries + 1):
        total_pass = 0.0
        for c in c_i_series:
            comb_num = math.comb(n_tries - c, k) if (n_tries - c) >= k else 0
            comb_den = math.comb(n_tries, k)
            problem_pass = 1 - (comb_num / comb_den)
            total_pass += problem_pass

        pass_k = total_pass / len(c_i_series)
        pass_k_list.append(pass_k)

    return pass_k_list


def plot_accuracy_distribution(
    ax, error_rates: pd.Series, simplified_label: str, position=0, total_bars=1
):
    """绘制准确率分布图，支持分组显示"""
    bins = [i / 10 for i in range(11)]

    # 计算每个柱状图的位置偏移，实现分组效果
    bin_width = bins[1] - bins[0]
    bar_width = bin_width * 0.8 / total_bars  # 总宽度占bin宽度的80%，按bar数量分配
    offset = (position - (total_bars - 1) / 2) * bar_width

    # 计算直方图数据
    hist_values, bin_edges = np.histogram(error_rates, bins=bins)

    # 绘制分组柱状图
    x_positions = bin_edges[:-1] + offset + bin_width / 2
    ax.bar(x_positions, hist_values, width=bar_width, label=simplified_label, alpha=0.7)

    ax.set_xlabel("acc rate")
    ax.set_ylabel("count")
    ax.set_title("acc rate distribution")
    ax.legend()
    ax.grid(True)


def plot_pass_k_curve(ax, pass_k_list: list[float], simplified_label: str):
    """绘制 pass@k 曲线图"""
    ax.plot(range(1, len(pass_k_list) + 1), pass_k_list, label=simplified_label)
    ax.set_xlabel("Number of Samples k")
    ax.set_ylabel("pass@k")
    ax.set_xscale("log")
    ax.set_title("pass@k Curve")
    ax.legend()
    ax.grid(True)


def update_results_dataframe(
    results_df: pd.DataFrame,
    simplified_label: str,
    outputs: pd.Series,
    correct_per_row: pd.Series,
) -> pd.DataFrame:
    """更新结果 DataFrame"""
    total = len(outputs)
    wrong_format = total - outputs.isin(["positive", "negative", "neutral"]).sum()
    correct = correct_per_row.sum()
    accuracy = correct / total

    # 将结果添加到 DataFrame
    results_df.loc[len(results_df)] = [
        simplified_label,
        correct,
        wrong_format,
        total,
        accuracy,
    ]

    return results_df


def print_results(results_df: pd.DataFrame):
    """打印评估结果汇总"""
    print("\n评估结果汇总：")
    results_df_display = results_df.copy()
    results_df_display["准确率"] = results_df_display["准确率"].apply(
        lambda x: f"{x:.3f}"
    )
    print(results_df_display)


def main(eval_list: list[str]):
    """主函数，协调各个子函数完成评估任务"""
    # 创建 DataFrame 用于存储结果指标
    results_df = pd.DataFrame(
        columns=["数据集", "正确样本数", "格式错误样本数", "总样本数", "准确率"]
    )

    # 初始化绘图
    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 预先加载所有数据，以便进行分组绘制
    all_data = []
    for eval_name in eval_list:
        # 加载数据
        ground_truth, output_data, n_tries = load_data(eval_name)

        # 计算指标
        outputs, correct_per_row, c_i_series, _ = calculate_metrics(
            output_data, ground_truth
        )

        error_rates = c_i_series / n_tries
        simplified_label = simplify_label(eval_name)
        pass_k_list = calculate_pass_k(c_i_series, n_tries)

        all_data.append(
            (simplified_label, error_rates, pass_k_list, outputs, correct_per_row)
        )

        # 更新结果 DataFrame
        results_df = update_results_dataframe(
            results_df, simplified_label, outputs, correct_per_row
        )

    # 绘制所有图表
    total_bars = len(all_data)
    for i, (simplified_label, error_rates, pass_k_list, _, _) in enumerate(all_data):
        # 绘制分组柱状图
        plot_accuracy_distribution(axs[0], error_rates, simplified_label, i, total_bars)

        # 绘制 pass@k 曲线
        plot_pass_k_curve(axs[1], pass_k_list, simplified_label)

    # 打印结果汇总
    print_results(results_df)
    results_df.to_csv("./results/results.csv", index=False)

    # 显示图表
    plt.tight_layout()
    plt.savefig("./results/results.svg")
    plt.show()


if __name__ == "__main__":
    # 作为基线的数据，用于对比
    base_list = [
        "./eval/restaurant_test.parquet",
        # "./eval/restaurant_train.parquet",
        # "./eval/laptop_test.parquet",
        # "./eval/laptop_train.parquet",
    ]

    # 对比模型在checkpoint的数据
    checkpoint_list = [
        "./eval/restaurant_test_checkpoint-30_20251109_214809.parquet",
        "./eval/restaurant_test_checkpoint-60_20251109_215334.parquet",
        "./eval/restaurant_test_checkpoint-90_20251109_215853.parquet",
        "./eval/restaurant_test_checkpoint-111_20251109_214158.parquet",
    ]
    main(base_list + checkpoint_list)
