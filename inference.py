import datetime
import traceback

import pandas as pd
import torch
from datasets import load_from_disk
from swanlab.plugin.notification import EmailCallback
from unsloth import FastLanguageModel
from vllm import SamplingParams

torch._dynamo.config.cache_size_limit = 256

# 以下设置应该与训练脚本中的设置相同
# TODO 通过配置文件传递，从而保持训练和推理时的参数一致
max_seq_length = 1024  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower
system_prompt = "Analyze the sentiment of a certain object from a given text. Reason carefully before giving your response. Your response should end with a single word: positive, negative, or neutral, on a new line."


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
                    "content": system_prompt,
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


def load_unsloth_model(model_path, lora_rank, enable_lora=True):
    """加载unsloth模型

    Args:
        model_path: 模型路径
        lora_rank: LoRA rank
        enable_lora: 是否启用LoRA

    Returns:
        模型和分词器
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.95,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )

    return model, tokenizer


def inference(
    model, tokenizer, test_name: str, n_try: int, lora_path: str | None = None
) -> pd.DataFrame:
    # 设置采样参数
    sampling_params = SamplingParams(
        n=n_try,
        temperature=1.0,
        max_tokens=512,
    )
    if "test" in test_name:
        # 假如是在测试集上生成，则基本使用推荐/默认参数
        sampling_params.top_p = 0.9
    else:
        # 假如是在训练集上生成，则将生成参数调整到和GRPO的参数相同
        sampling_params.top_p = 1.0
        sampling_params.min_p = 0.1
        sampling_params.top_k = -1

    data: pd.DataFrame = load_from_disk(f"./data/{test_name}").to_pandas()  # type: ignore
    input_conversations = make_conversation_list(data)

    print("=" * 70)
    print(f"正在生成：{test_name}：{data.shape = } with {n_try = }")
    print("=" * 70)

    # 为每个对话应用chat template并进行推理
    texts = []
    for conv in input_conversations:
        text = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        texts.append(text)

    # 执行批量推理
    if lora_path is None:
        lora_request = None
    else:
        lora_request = model.load_lora(lora_path)

    outputs = model.fast_generate(
        texts, sampling_params=sampling_params, lora_request=lora_request
    )

    return gather_outputs(data["id"], outputs)


def main():
    # 推理的设置：
    # 要推理的数据集列表
    target_dataset_list = [
        # "restaurant_test",
        "laptop_test",
        # "restaurant_train",
        "laptop_train",
    ]

    # 要使用的LoRA路径，None表示使用基础模型
    # lora_path = None
    lora_path = "./outputs/checkpoint-470"

    model, tokenizer = load_unsloth_model(
        "./models/Qwen3-4B-Instruct-2507",
        lora_rank,
        enable_lora=True if lora_path is not None else False,
    )

    # 对每个数据集调用inference函数，复用同一个模型实例
    for dataset_name in target_dataset_list:
        n_try = 128 if "test" in dataset_name else 32
        output_data = inference(model, tokenizer, dataset_name, n_try, lora_path)

        # 使用与 eval_result.py 中一致的输出路径
        # 添加日期时间戳到输出文件名
        identifier = lora_path if lora_path is not None else "base"
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_data.to_parquet(
            f"./eval/{dataset_name}_{identifier.split('/')[-1]}_{current_time}.parquet"
        )


if __name__ == "__main__":
    # 初始化邮件通知插件
    email_callback = EmailCallback(
        sender_email="1186977762@qq.com",
        receiver_email="1186977762@qq.com",
        password="wzubregistphgdfc",
        smtp_server="smtp.qq.com",
        port=587,
        language="zh",
    )

    try:
        main()
        email_callback.send_email(
            subject="Inference Finished",
            body="All inference tasks have been completed.",
        )
    except Exception as e:
        # 发送异常信息
        email_callback.send_email(
            subject="Inference Failed",
            body=f"An error occurred: {str(e)}\n\n{traceback.format_exc()}",
        )
