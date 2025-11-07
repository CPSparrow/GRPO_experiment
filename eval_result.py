import math

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk


def main():
    eval_list = [
        # "restaurant_test",
        # "laptop_test",
        "restaurant_test_114",
    ]

    _, axs = plt.subplots(1, 2, figsize=(10, 6))
    for eval_name in eval_list:
        ground_truth: pd.DataFrame = load_from_disk(
            f"./data/{eval_name[:-4]}"
        ).to_pandas()  # type: ignore
        output_data = pd.read_parquet(f"./eval/{eval_name}.parquet")

        print(f"{ground_truth.shape = }, 列{ground_truth.columns.tolist()}")
        print(f"{output_data.shape = }, 列{output_data.columns.tolist()}")

        n_tries = output_data.shape[0] // ground_truth.shape[0]

        assert ground_truth.shape[0] * n_tries == output_data.shape[0], "数据量不匹配"

        outputs = output_data["output"].str.split().str[-1]
        true_labels = ground_truth.loc[output_data["index"], "sentiment"].values
        correct_per_row = outputs == true_labels  # 每行是否预测正确

        # 计算每个题目的正确样本数c_i（核心用于后续统计）
        c_i_series = correct_per_row.groupby(output_data["index"]).sum()
        num_problems = len(c_i_series)

        print("\n=== 题目错误率分布 ===")
        error_rates = (
            c_i_series / n_tries
        )  # 每个题目的错误率（(总样本-正确样本)/总样本）
        bins = [i / 10 for i in range(11)]  # 区间：0.0-0.1, 0.1-0.2,...,0.9-1.0
        error_bins = pd.cut(error_rates, bins=bins, right=False)  # 左闭右开区间
        error_dist = error_bins.value_counts().sort_index()  # 统计每个区间的题目数

        # 输出分布结果
        for interval, count in error_dist.items():
            print(f"{interval =}, {count = }")

        # 绘制错误率直方图
        axs[0].hist(
            error_rates, bins=bins, label=f"{eval_name}", alpha=0.6, density=True
        )
        axs[0].set_xlabel("acc rate")
        axs[0].set_ylabel("count")
        axs[0].set_title("acc rate distribution")
        axs[0].legend()
        axs[0].grid(True)

        pass_k_list = []

        for k in range(1, n_tries + 1):
            total_pass = 0.0
            for c in c_i_series:
                # 计算组合数 C(n-c, k)（n-c < k时为0）
                comb_num = math.comb(n_tries - c, k) if (n_tries - c) >= k else 0
                comb_den = math.comb(n_tries, k)
                # 单题pass@k：1 - C(n-c, k)/C(n, k)
                problem_pass = 1 - (comb_num / comb_den)
                total_pass += problem_pass
            # 所有题目的平均pass@k
            pass_k = total_pass / num_problems
            pass_k_list.append(pass_k)

        # 绘制pass@k曲线
        axs[1].plot(range(1, n_tries + 1), pass_k_list, label=f"{eval_name}")
        axs[1].set_xlabel("Number of Samples k")
        axs[1].set_ylabel("pass@k")
        axs[1].set_xscale("log")
        axs[1].set_title("pass@k Curve")
        axs[1].legend()
        axs[1].grid(True)

        print("\n=== 样本级统计 ===")
        total = output_data.shape[0]
        wrong_format = total - outputs.isin(["positive", "negative", "neutral"]).sum()
        correct = correct_per_row.sum()

        print(
            f"正确样本数: {correct}, 格式错误样本数: {wrong_format}, 总样本数: {total}"
        )
        print(f"样本准确率: {correct / total:.3f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
