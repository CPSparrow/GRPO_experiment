import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import f1_score


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


def find_matched_directory(eval_name: str) -> str:
    """查找匹配的 ground_truth 数据目录"""
    data_dir = Path("./data")
    test_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]

    for test_dir in test_dirs:
        if eval_name.split("/")[-1].startswith(test_dir):
            return test_dir

    raise ValueError(f"无法为 {eval_name} 找到匹配的ground_truth目录")


def calculate_metrics(
    output_data: pd.DataFrame, ground_truth: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series, int, pd.Series]:
    """
    计算模型预测的关键指标

    :param output_data: 模型输出数据，包含 "index" 和 "output" 列
    :param ground_truth: 真实标签数据，包含 "index" 和 "sentiment" 列
    :return: 包含模型预测结果、每行是否正确、每个题目正确样本数、题目总数、是否隐式情感的Series
    """
    outputs = output_data["output"].str.split().str[-1]
    true_labels = ground_truth.loc[output_data["index"], "sentiment"].values
    is_implicit: pd.Series = ground_truth.loc[
        output_data["index"], "is implicit"
    ].values  # type: ignore
    correct_per_row = outputs == true_labels  # 每行是否预测正确

    # 计算每个题目的正确样本数
    c_i_series = correct_per_row.groupby(output_data["index"]).sum()
    num_problems = len(c_i_series)

    return outputs, correct_per_row, c_i_series, num_problems, is_implicit


def simplify_label(eval_name):
    """
    将文件名简化为更直观的label格式

    我们认为文件名的格式为：
    /some_path/dataset_{train|test}[_checkpoint-number_...].parquet
    """
    # 去掉路径和扩展名
    file_name = eval_name.split("/")[-1].split(".")[0]

    # 提取数据集类型和用途（通常是前两部分）
    parts = file_name.split("_")
    dataset_type = parts[0]  # "laptop" or "restaurant"
    data_purpose = parts[1]  # "train" or "test"

    if len(parts) > 2 and parts[2].startswith("checkpoint"):
        # 提取checkpoint数字
        checkpoint_part = parts[2]
        checkpoint_num = (
            checkpoint_part.split("-")[1]
            if "-" in checkpoint_part
            else checkpoint_part[10:]
        )
        return f"{dataset_type}_{data_purpose}_{checkpoint_num}"
    else:
        return f"{dataset_type}_{data_purpose}_base"


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


def update_results_dataframe(
    results_df: pd.DataFrame,
    simplified_label: str,
    outputs: pd.Series,
    correct_per_row: pd.Series,
    ground_truth: pd.DataFrame,
    output_data: pd.DataFrame,
) -> pd.DataFrame:
    """更新结果 DataFrame，包含ISA和All的F1和Acc指标"""
    true_labels = ground_truth.loc[output_data["index"], "sentiment"].values
    is_implicit = ground_truth.loc[output_data["index"], "is implicit"].values == "True"

    total = len(outputs)
    wrong_format = total - outputs.isin(["positive", "negative", "neutral"]).sum()

    all_f1, all_acc = calculate_f1_and_accuracy(outputs, true_labels)
    isa_f1, isa_acc = calculate_f1_and_accuracy(outputs, true_labels, is_implicit)

    # 将结果添加到 DataFrame
    results_df.loc[len(results_df)] = [
        simplified_label,
        correct_per_row.sum(),
        wrong_format,
        total,
        isa_f1,
        isa_acc,
        all_f1,
        all_acc,
    ]

    return results_df


def calculate_f1_and_accuracy(
    predictions, true_labels, mask=None, filter_invalid=False
):
    """计算F1分数和准确率，可以选择应用掩码过滤特定样本"""
    if mask is not None:
        # 应用掩码，即统计 ISA 样本的 F1 和 Acc
        predictions = predictions[mask]
        true_labels = true_labels[mask]

        if len(predictions) == 0:
            return 0.0, 0.0

    if filter_invalid:
        valid_mask = predictions.isin(["positive", "negative", "neutral"])
        valid_predictions = predictions[valid_mask]
        valid_true_labels = true_labels[valid_mask]

        if len(valid_predictions) == 0:
            return 0.0, 0.0

        f1 = f1_score(valid_true_labels, valid_predictions, average="macro")
        accuracy = (valid_predictions == valid_true_labels).mean()

    else:
        f1 = f1_score(true_labels, predictions, average="macro")
        accuracy = (predictions == true_labels).mean()

    return f1, accuracy


def plot_accuracy_distribution(
    ax, error_rates: pd.Series, simplified_label: str, position=0, total_bars=1
):
    """绘制准确率分布图，支持分组显示"""
    n_bins = 20
    bins = [i / n_bins for i in range(n_bins + 1)]

    # 计算每个柱状图的位置偏移，实现分组效果
    bin_width = bins[1] - bins[0]
    bar_width = bin_width * 0.8 / total_bars  # 总宽度占bin宽度的80%，按bar数量分配
    offset = (position - (total_bars - 1) / 2) * bar_width

    # 计算直方图数据
    hist_values, bin_edges = np.histogram(error_rates, bins=bins)

    # 转换为比例
    total_samples = len(error_rates)
    hist_ratios = hist_values / total_samples

    # 绘制分组柱状图
    x_positions = bin_edges[:-1] + offset + bin_width / 2
    ax.bar(x_positions, hist_ratios, width=bar_width, label=simplified_label, alpha=0.7)

    ax.set_xlabel("acc rate")
    ax.set_ylabel("ratio")
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


def print_results(results_df: pd.DataFrame):
    """打印评估结果汇总"""
    print("\n评估结果汇总：")
    results_df_display = results_df.copy()

    print("       数据集           ISA_F1  ISA_Acc All_F1  All_Acc")
    print("=" * 55)
    for _, row in results_df_display.iterrows():
        print(
            f"{row['数据集']:<22}: "
            f"{row['ISA_F1']:.4f}  "
            f"{row['ISA_Acc']:.4f}  "
            f"{row['All_F1']:.4f}  "
            f"{row['All_Acc']:.4f}"
        )


def main(eval_list: list[str]):
    """主函数，协调各个子函数完成评估任务"""
    results_df = pd.DataFrame(
        columns=[
            "数据集",
            "正确样本数",
            "格式错误样本数",
            "总样本数",
            "ISA_F1",
            "ISA_Acc",
            "All_F1",
            "All_Acc",
        ]
    )

    # 初始化绘图
    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    all_data = []
    for eval_name in eval_list:
        ground_truth, output_data, n_tries = load_data(eval_name)

        outputs, correct_per_row, c_i_series, _, _ = calculate_metrics(
            output_data, ground_truth
        )

        error_rates = c_i_series / n_tries
        simplified_label = simplify_label(eval_name)
        pass_k_list = calculate_pass_k(c_i_series, n_tries)

        all_data.append(
            (simplified_label, error_rates, pass_k_list, outputs, correct_per_row)
        )

        # 更新结果 DataFrame，传入ground_truth和output_data以计算新指标
        results_df = update_results_dataframe(
            results_df,
            simplified_label,
            outputs,
            correct_per_row,
            ground_truth,
            output_data,
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

    # 保存结果到CSV文件
    # 确保results目录存在
    Path("./results").mkdir(exist_ok=True)
    results_df.to_csv("./results/results.csv", index=False)

    # 显示图表
    plt.tight_layout()
    plt.savefig("./results/results.svg")
    # plt.show()


if __name__ == "__main__":
    # 作为基线的数据，用于对比
    base_list = [
        "./eval/restaurant_test.parquet",
        # "./eval/restaurant_train.parquet",
        "./eval/laptop_test.parquet",
        # "./eval/laptop_train.parquet",
    ]

    # 对比模型在checkpoint的数据
    checkpoint_list = [
        # grpo_1 laptop
        # "./eval/grpo_1/laptop_test_checkpoint-200_20251107_223953.parquet",
        # "./eval/grpo_1/laptop_test_checkpoint-400_20251107_221740.parquet",
        # "./eval/grpo_1/laptop_test_checkpoint-470_20251107_214508.parquet",
        # grpo_1 restaurant
        # "./eval/grpo_1/restaurant_test_checkpoint-200_20251108_234158.parquet",
        # "./eval/grpo_1/restaurant_test_checkpoint-470_20251108_231708.parquet",
        # grpo_2 restaurant
        # "./eval/restaurant_test_checkpoint-30_20251109_214809.parquet",
        # "./eval/restaurant_test_checkpoint-60_20251109_215334.parquet",
        # "./eval/restaurant_test_checkpoint-90_20251109_215853.parquet",
        # "./eval/restaurant_test_checkpoint-111_20251109_214158.parquet",
        # grpo_2 laptop
        # "./eval/laptop_test_checkpoint-30_20251109_214809.parquet",
        # "./eval/laptop_test_checkpoint-60_20251109_215334.parquet",
        # "./eval/laptop_test_checkpoint-90_20251109_215853.parquet",
        # "./eval/laptop_test_checkpoint-111_20251109_214158.parquet",
    ]
    main(base_list + checkpoint_list)
