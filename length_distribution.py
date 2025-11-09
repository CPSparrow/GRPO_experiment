from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 加载数据集
    train_set = load_from_disk("./data/laptop_train_hard")
    
    # 计算每个样本的句子长度
    tmp_set = train_set.map(lambda x: {"length": len(x["sentence"])})
    len_array = np.array(tmp_set["length"])
    
    # 打印基本统计信息
    print(f"均值: {len_array.mean():.2f}")
    print(f"标准差: {len_array.std():.2f}")
    print(f"最大值: {len_array.max()}")
    print(f"最小值: {len_array.min()}")
    
    # 计算累积分布
    max_length = len_array.max()
    # 生成标准化的长度值（0到1之间）
    normalized_lengths = len_array / max_length
    
    # 排序并计算累积比例
    sorted_lengths = np.sort(normalized_lengths)
    cumulative_proportions = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_lengths, cumulative_proportions, marker='o', markersize=3, linestyle='-', linewidth=1)
    
    # 设置图表属性
    plt.title('句子长度累积分布', fontsize=14)
    plt.xlabel('标准化长度 (长度/最大长度)', fontsize=12)
    plt.ylabel('累积比例', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 添加一些有用的标记点
    for p in [0.25, 0.5, 0.75, 0.9, 0.95]:
        # 找到对应比例的长度值
        idx = int(p * len(sorted_lengths))
        length_p = sorted_lengths[idx]
        plt.axhline(y=p, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=length_p, color='r', linestyle='--', alpha=0.3)
        plt.text(0.02, p + 0.02, f'{p:.0%}', fontsize=9)
        plt.text(length_p + 0.02, 0.02, f'{length_p:.2f}', fontsize=9, rotation=90)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('length_distribution_cdf.png', dpi=300)
    plt.show()
