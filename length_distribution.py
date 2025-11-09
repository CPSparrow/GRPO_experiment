from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_set = load_from_disk("./data/laptop_train_hard")
    len_array = np.array(
        [len(x["sentence"]) for x in train_set]  # type: ignore
    )

    total_samples = len(len_array)
    print(
        f"均值: {len_array.mean():.2f}, 最大值: {len_array.max()}, 最小值: {len_array.min()}, 总样本数: {total_samples}"
    )

    # 排序并计算最大长度分布
    sorted_lengths = np.sort(len_array)
    percentiles = np.linspace(1, 100, 100)
    # 计算每个百分比对应的索引并获取最大长度
    indices = np.maximum(0, (percentiles / 100 * total_samples - 1).astype(int))
    max_lengths = sorted_lengths[indices]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(
        percentiles, max_lengths, marker="o", markersize=3, linestyle="-", linewidth=1
    )
    plt.title("Length Distribution", fontsize=14)
    plt.xlabel("top x of samples", fontsize=12)
    plt.ylabel("max length", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(0, 100)
    plt.ylim(0, sorted_lengths[-1] * 1.1)

    # 添加标记点
    for p in [90, 95]:
        idx = max(0, int(p / 100 * total_samples) - 1)
        max_len_p = sorted_lengths[idx]
        plt.axvline(x=p, color="r", linestyle="--", alpha=0.3)
        plt.axhline(y=max_len_p, color="r", linestyle="--", alpha=0.3)
        plt.text(p + 1, sorted_lengths[-1] * 0.05, f"{p}%", fontsize=9, rotation=90)
        plt.text(5, max_len_p + sorted_lengths[-1] * 0.02, f"{max_len_p}", fontsize=9)

    plt.tight_layout()
    plt.show()
