"""筛选laptop_train数据集中的困难样本"""

import pandas as pd
from datasets import load_from_disk


def calc_acc(
    eval_output: pd.DataFrame, dataset_df: pd.DataFrame, n_tries: int
) -> pd.Series:
    """计算每个样本的正确率"""
    outputs = eval_output["output"].str.split().str[-1]
    true_labels = dataset_df.loc[eval_output["index"], "sentiment"].values
    correct_per_row = outputs == true_labels

    # 计算每个题目的正确样本数
    c_i_series = correct_per_row.groupby(eval_output["index"]).sum()
    return c_i_series / n_tries


def main(threshold: float):
    dataset = load_from_disk("./data/laptop_train")
    eval_output = pd.read_parquet("./eval/laptop_train.parquet")
    n_try = 16

    acc_per_sample = calc_acc(eval_output, dataset.to_pandas(), n_try)  # type: ignore
    hard_sample_indices = acc_per_sample[acc_per_sample <= threshold].index
    hard_samples = dataset.select(hard_sample_indices)  # type: ignore
    input("enter to continue")
    hard_samples.save_to_disk("./data/laptop_train_hard")


if __name__ == "__main__":
    threshold = 0.8
    main(threshold)
