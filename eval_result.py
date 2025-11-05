import pandas as pd

from inference import load_data


def main(test_name: str):
    ground_truth = load_data(test_name, seed=None, n_sample=-1)
    output_data = pd.read_parquet(f"./data/output/{test_name}_test.parquet")

    # 数据验证
    print(ground_truth.shape)
    print(output_data.shape)
    print(ground_truth.iloc[0])
    print(output_data.iloc[0])

    if ground_truth.shape[0] * 128 != output_data.shape[0]:
        print("items do not match")
        return

    # 预定义映射字典
    ans_to_num = {"positive": 0, "negative": 1, "neutral": 2}

    # 向量化处理输出数据
    outputs = output_data["output"].str.split().str[-1]
    mapped_outputs = outputs.map(ans_to_num)

    # 计算统计量
    total = len(output_data)
    wrong_format = mapped_outputs.isna().sum()
    correct = (
        (mapped_outputs == ground_truth.loc[output_data["index"], "sentiment"].values)
        & mapped_outputs.notna()
    ).sum()

    print(f"{correct = }, {wrong_format = }, {total = }")
    print(f"Accuracy: {correct / total:.3f}")


if __name__ == "__main__":
    test_list = ["restaurant", "laptop"]
    test_name = test_list[1]

    main(test_name)
