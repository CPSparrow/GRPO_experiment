import pandas as pd

from datasets import load_from_disk


def main():
    ground_truth: pd.DataFrame = load_from_disk("./data/laptop_test").to_pandas()  # type: ignore
    output_data = pd.read_parquet(f"./eval/laptop_test.parquet")

    # 数据验证
    print(ground_truth.shape, ground_truth.columns)
    print(output_data.shape, output_data.columns)

    if ground_truth.shape[0] * 128 != output_data.shape[0]:
        print("items do not match")
        return

    # 处理输出数据
    outputs = output_data["output"].str.split().str[-1]

    # 计算统计量
    total = output_data.shape[0]
    wrong_format = total - outputs.isin(["positive", "negative", "neutral"]).sum()
    correct = (
        outputs == ground_truth.loc[output_data["index"], "sentiment"].values
    ).sum()

    print(f"{correct = }, {wrong_format = }, {total = }")
    print(f"Accuracy: {correct / total:.3f}")


if __name__ == "__main__":
    main()
