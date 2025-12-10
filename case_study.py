import time
import random
import pandas as pd
from pathlib import Path
from datasets import load_from_disk


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


def calculate_correctness_for_problems(
    output_data: pd.DataFrame, ground_truth: pd.DataFrame
) -> tuple[pd.Series, dict, dict]:
    """计算每个题目的正确率、错误回复和正确回复"""
    outputs = output_data["output"]
    predictions = outputs.str.split().str[-1]
    true_labels = ground_truth.loc[output_data["index"], "sentiment"].values

    correct_per_row = predictions == true_labels

    problem_correctness = correct_per_row.groupby(output_data["index"]).mean()

    wrong_responses = {}
    correct_responses = {}
    for idx in output_data["index"].unique():
        mask_wrong = (output_data["index"] == idx) & (~correct_per_row)
        mask_correct = (output_data["index"] == idx) & (correct_per_row)
        wrong_responses[idx] = outputs[mask_wrong].tolist()
        correct_responses[idx] = outputs[mask_correct].tolist()

    return problem_correctness, wrong_responses, correct_responses


def main(target: str, seed: int | float):
    random.seed(seed)
    print(f"随机数种子seed: {seed}")

    eval_name = target
    ground_truth, output_data, _ = load_data(eval_name)

    correctness, wrong_responses, correct_responses = (
        calculate_correctness_for_problems(output_data, ground_truth)
    )

    zero_accuracy_count = sum(correctness == 0)
    print(f"\n1. 正确率为0的样本个数: {zero_accuracy_count}")

    low, high = 0.2, 0.4
    low_accuracy_count = sum((correctness > low) & (correctness <= high))
    print(f"2. 正确率大于{low}小于等于{high}的样本个数: {low_accuracy_count}")

    low_accuracy_problems = [
        idx
        for idx in correctness.index
        if low < correctness[idx] <= high  # type: ignore
    ]

    selected_problem_idx = random.choice(low_accuracy_problems)

    sentence = ground_truth.loc[selected_problem_idx, "sentence"]
    senti_object = ground_truth.loc[selected_problem_idx, "object"]
    correct_label = ground_truth.loc[selected_problem_idx, "sentiment"]
    is_implicit = ground_truth.loc[selected_problem_idx, "is implicit"]

    # 打印题目信息
    print(f"\n3. 选中的样本信息:")
    print(f"样本id: {selected_problem_idx}")
    print(f"sentence: {sentence}")
    print(f"object: {senti_object}")
    print(f"sentiment: {correct_label}")
    print(f"is implicit: {is_implicit}")
    print(
        f"正确率: {correctness[selected_problem_idx]:.2%} "
        f"错误回复数量: {len(wrong_responses[selected_problem_idx])} "
        f"正确回复数量: {len(correct_responses[selected_problem_idx])}"
    )

    print("\n3个错误回复:")
    selected_wrong = random.sample(wrong_responses[selected_problem_idx], 3)
    for i, response in enumerate(selected_wrong, 1):
        print(f"{i}. {response}")

    print("\n1个正确回复:")
    if correct_responses[selected_problem_idx]:
        selected_correct = random.sample(correct_responses[selected_problem_idx], 1)[0]
        print(f"1. {selected_correct}")
    else:
        print("没有正确回复")


if __name__ == "__main__":
    # seed = 42
    seed = time.time()
    # target = "./eval/laptop_test.parquet"
    target = "./eval/restaurant_test.parquet"
    main(target, seed)
