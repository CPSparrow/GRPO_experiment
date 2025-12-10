"""
这个脚本实现了对指定数据集列表的推理，使用Qwen3-4B-Instruct模型。，用于测试新想法的可行性
方法说明：
1.首先，基于train set的推理，筛选出那些模型难以回答的问题(暂定为难度为1/8，随机筛选8个)
2.让模型对于这8个例子，推理出思考作为上下文；
3.基于这个上下文，重新在数据集上进行推理，评估模型的性能。
"""

import datetime
import json
import random
import traceback
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk
from swanlab.plugin.notification import EmailCallback
from unsloth import FastLanguageModel
from vllm import SamplingParams

torch._dynamo.config.cache_size_limit = 256

# 以下设置应该与训练脚本中的设置相同
max_seq_length = 1024  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower
system_prompt = "Analyze the sentiment of a certain object from a given text. Reason carefully before giving your response. Your response should end with a single word: positive, negative, or neutral, on a new line."


# 初始化邮件通知插件
email_callback = EmailCallback(
    sender_email="1186977762@qq.com",
    receiver_email="1186977762@qq.com",
    password="wzubregistphgdfc",
    smtp_server="smtp.qq.com",
    port=587,
    language="zh",
)


def inference_context(
    model, tokenizer, testset_name: str, seed: int = 42
) -> list[dict[str, str]]:
    """
    根据testset_name，加载对应的train_set的推理记录，依据脚本开头的规则筛选出8个例子,
    生成反思记录作为上下文。

    :params testset_name: 测试集名称
    :return: 包含8个例子的反思记录作为上下文
    """
    # 设置随机种子
    random.seed(seed)
    # 获取对应的训练集名称
    train_set_name = testset_name.replace("_test", "_train")
    train_file = Path(f"./eval/{train_set_name}.parquet")
    assert train_file.exists(), f"未找到{train_set_name}的推理记录文件"

    # 加载训练集的ground truth数据
    ground_truth = load_from_disk(f"./data/{train_set_name}").to_pandas()

    # 加载训练集的推理结果
    output_data = pd.read_parquet(train_file)

    # 计算每个题目的正确率
    output_list = output_data["output"]
    predictions = output_list.str.split().str[-1]
    true_labels = ground_truth.loc[output_data["index"], "sentiment"].values
    correct_per_row = predictions == true_labels
    problem_correctness = correct_per_row.groupby(output_data["index"]).mean()

    # 筛选出正确率为1/8的题目
    difficulty_threshold = 1 / 8
    hard_problems = problem_correctness[
        (problem_correctness <= difficulty_threshold+0.02) & (problem_correctness >= difficulty_threshold-0.02)
    ].index.tolist()

    # 如果符合条件的题目不足8个，考虑放宽条件
    if len(hard_problems) < 8:
        # 选择所有正确率大于0小于0.6的题目
        all_problems = problem_correctness[
            (problem_correctness > 0) & (problem_correctness < 0.6)
        ].index.tolist()
    assert len(all_problems) >= 8, f"筛选出的题目数量不足8个"
    hard_problems = random.sample(hard_problems, 8)

    # 生成反思记录作为上下文
    context_list = []
    for idx in hard_problems:
        # 获取题目信息
        sentence = ground_truth.loc[idx, "sentence"]
        obj = ground_truth.loc[idx, "object"]

        # 获取该题目的所有推理结果
        problem_outputs = output_data[output_data["index"] == idx]["output"].tolist()

        # 获取该题目的所有预测结果和正确标签
        problem_data = output_data[output_data["index"] == idx]
        problem_predictions = problem_data["output"].str.split().str[-1]
        problem_true_label = ground_truth.loc[idx, "sentiment"]

        # 将结果分为正确和错误两类
        correct_responses = []
        wrong_responses = []
        for i, (output, pred) in enumerate(zip(problem_outputs, problem_predictions)):
            if pred == problem_true_label:
                correct_responses.append(output)
            else:
                wrong_responses.append(output)

        # 重新设置随机种子
        random.seed(seed)

        # 随机抽取一个正确的和一个错误的答案
        ans_right = random.choice(correct_responses)
        ans_wrong = random.choice(wrong_responses)

        # 生成反思记录
        chat_log = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"sentence:{sentence}, object:{obj}"},
            {"role": "assistant", "content": ans_wrong},
            {
                "role": "user",
                "content": f"Your answer is wrong.Please try again and give a new response.",
            },
            {"role": "assistant", "content": ans_right},
            {
                "role": "user",
                "content": f"Your answer is right.Now please think carefully and make a conclusion:\n1.What mistake did you make while giving the wrong answer?\n2.How could you have avoided this mistake in the future?",
            },
        ]
        context_list.append(chat_log)

    # 调用模型，生成反思记录

    # 设置生成参数：
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=512,
        n=1,
        top_p=1.0,
        min_p=0.1,
        top_k=-1,
    )

    texts = []
    for chat_log in context_list:
        text = tokenizer.apply_chat_template(
            chat_log, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
    output_list = model.fast_generate(
        texts,
        sampling_params=sampling_params,
        lora_request=None,
    )

    for index, output_obj in enumerate(output_list):
        for item in output_obj.outputs:
            context_list[index].append({"role": "assistant", "content": item.text})

    context = [{"role": "system", "content": system_prompt}]
    for chat_log in context_list:
        for item in chat_log:
            if item["role"] != "system":
                context.append(item)

    # 将context保存为json：
    with open(f"./data/{train_set_name}_context.json", "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=4)

    # 测量context的长度
    token_list = tokenizer.apply_chat_template(
        context, tokenize=True, add_generation_prompt=True
    )
    print("=" * 60)
    print(f"context的token数为：{len(token_list)}")
    print("=" * 60)

    return context


def make_conversation_list(data: pd.DataFrame) -> list[list[dict]]:
    """将数据集转换为对话格式

    Args:
        data (pd.DataFrame): 输入的数据集

    Returns:
        list[list[dict]]: 储存为列表的对话数据
    """
    # TODO 补全上下文
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


def load_unsloth_model(model_path):
    """加载unsloth模型

    Args:
        model_path: 模型路径

    Returns:
        模型和分词器
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.95,
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


def gather_outputs(index_list: pd.Series, output_list: list):
    """
    从vllm的输出中提取生成结果

    Args:
        index_list (pd.Series): 输入数据的索引列
        output_list (list): vllm的输出列表

    Returns:
        pd.DataFrame: 包含索引、生成文本和文本长度的DataFrame
    """
    data = list()
    for index, output in zip(index_list, output_list):
        for item in output.outputs:
            data.append(
                dict(index=index, output=item.text, output_tokens=len(item.token_ids))
            )
    return pd.DataFrame(data)


def inference(
    model, tokenizer, test_name: str, lora_path: str | None = None
) -> pd.DataFrame:
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=512,
    )
    if "test" in test_name:
        # 假如是在测试集上生成，则基本使用推荐/默认参数
        sampling_params.n = 16
        sampling_params.top_p = 0.9
    else:
        # 假如是在训练集上生成，则将生成参数调整到和GRPO的参数相同
        sampling_params.n = 16
        sampling_params.top_p = 1.0
        sampling_params.min_p = 0.1
        sampling_params.top_k = -1

    data: pd.DataFrame = load_from_disk(f"./data/{test_name}").to_pandas()  # type: ignore
    input_conversations = make_conversation_list(data)

    print("=" * 70)
    print(f"正在生成：{test_name}：{data.shape = } with n_try ={sampling_params.n}")
    print("=" * 70)

    # 为每个对话应用chat template并进行推理
    texts = []
    for conv in input_conversations:
        text = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
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


def main(target_dataset_list: list[str], lora_path: str | None):
    """
    主函数, 对指定的数据集列表进行推理

    在这个脚本中，target_dataset_list中的元素只会是"restaurant_test","laptop_test"中的一个
    """

    # 生成任务时间戳，确保一次任务中的所有文件使用相同的时间戳
    task_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    assert lora_path is None, "不支持使用LoRA"
    identifier = "base"
    identifier_name = identifier.split("/")[-1]

    model, tokenizer = load_unsloth_model(
        "./models/Qwen3-4B-Instruct-2507",
    )

    # 对每个数据集调用inference函数，复用同一个模型实例
    for dataset_name in target_dataset_list:
        output_data = inference(model, tokenizer, dataset_name, lora_path)

        # 使用与 eval_result.py 中一致的输出路径
        # 使用任务开始时生成的时间戳
        output_data.to_parquet(
            f"./eval/{dataset_name}_{identifier_name}_{task_timestamp}.parquet"
        )


if __name__ == "__main__":
    # 推理的设置：
    # 要推理的数据集列表
    target_dataset_list = [
        "restaurant_test",
        "laptop_test",
        # "restaurant_train",
        # "laptop_train",
    ]

    # 要使用的LoRA路径，然而我们不使用lora
    lora_path = None

    try:
        main(target_dataset_list, lora_path)
        email_callback.send_email(
            subject="Swanlab | Inference Finished",
            body=f"All inference tasks on {target_dataset_list} have been completed with self improve.",
        )
    except Exception as e:
        # 发送异常信息
        email_callback.send_email(
            subject="Swanlab | Inference Failed",
            body=f"An error occurred: \n{str(e)}\n\n{traceback.format_exc()}",
        )
