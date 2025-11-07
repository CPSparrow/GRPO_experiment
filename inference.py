import pandas as pd
from datasets import load_from_disk
from swanlab.plugin.notification import EmailCallback
from vllm import LLM  # type: ignore


def make_conversation_list(data: pd.DataFrame) -> list[list[dict]]:
    """将数据集转换为对话格式

    Args:
        data (pd.DataFrame): 输入的数据集

    Returns:
        list[list[dict]]: 储存为列表的对话数据
    """
    conversation_list = list()
    for _, row in data.iterrows():
        conversation_list.append(
            [
                {
                    "role": "system",
                    "content": "Analyze the sentiment of a certain object from a given text. Reason carefully before giving your response. Your response should end with a single word: positive, negative, or neutral, on a new line.",
                },
                {
                    "role": "user",
                    "content": f'sentence:"{row["sentence"]}", object:"{row["object"]}"',
                },
            ]
        )

    return conversation_list


def gather_outputs(index_list: pd.Series, output_list: list):
    data = list()
    for index, output in zip(index_list, output_list):
        for item in output.outputs:
            data.append(
                dict(index=index, output=item.text, output_tokens=len(item.token_ids))
            )
    return pd.DataFrame(data)


def main(llm, test_name: str, n_try: int):
    # 使用传入的已加载模型，不再重新创建
    sampling_params = llm.get_default_sampling_params()
    sampling_params.n = n_try
    sampling_params.temperature = 1
    sampling_params.max_tokens = 512
    if "test" in test_name:
        # 假如是在测试集上生成，则使用推荐和默认参数
        sampling_params.top_p = 0.9
    else:
        # 假如是在训练集上生成，则将生成参数调整到和GRPO的参数相同
        sampling_params.top_p = 1.0
        sampling_params.min_p = 0.1
        sampling_params.top_k = -1

    data: pd.DataFrame = load_from_disk(f"./data/{test_name}").to_pandas()  # type: ignore
    input_conversations = make_conversation_list(data)

    print(f"正在生成：{test_name}：{data.shape = } with {n_try = }")
    outputs = llm.chat(input_conversations, sampling_params, use_tqdm=True)

    output_data = gather_outputs(data["id"], outputs)
    # 使用与eval_result.py中一致的输出路径
    output_data.to_parquet(f"./eval/{test_name}.parquet")

    # 展示基本信息
    print(output_data.shape, output_data.columns)


if __name__ == "__main__":
    # inference config
    available = [
        # "restaurant_test",
        # "laptop_test",
        "restaurant_train",
        "laptop_train",
    ]

    # 只加载一次模型
    llm = LLM(
        model="./models/Qwen3-4B-Instruct-2507",
        max_model_len=1024,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
    )

    # 对每个数据集调用main函数，复用同一个模型实例
    for dataset_name in available:
        n_try = 128 if "test" in dataset_name else 16
        main(llm, dataset_name, n_try)

    # 初始化邮件通知插件
    email_callback = EmailCallback(
        sender_email="1186977762@qq.com",
        receiver_email="1186977762@qq.com",
        password="wzubregistphgdfc",
        smtp_server="smtp.qq.com",
        port=587,
        language="zh",
    )

    email_callback.send_email(
        subject="Inference Finished",  # 邮件标题
        content="All inference tasks have been completed.",  # 邮件内容
    )
