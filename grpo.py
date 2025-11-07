import os

import numpy as np
import swanlab
import torch
import unsloth
from datasets import load_from_disk
from swanlab.integration.transformers import SwanLabCallback
from swanlab.plugin.notification import EmailCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams

torch._dynamo.config.cache_size_limit = 512

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"


def reward_answer(completions, answer, **kwargs):
    """
    基于回答和正确答案计算奖励分数

    Args:
        completions: 模型生成的回答列表，每个元素是一个包含多个完成的列表
        answer: 正确答案列表，每个元素是一个字符串

    Returns:
        scores: 奖励分数列表，每个元素是一个浮点数
    """
    extract_answer = lambda output: output.strip().split()[-1]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [extract_answer(r) for r in responses]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess == true_answer:
            score = 1
        elif guess in ["positive", "negative", "neutral"]:
            score = 0
        else:
            score = -1
        scores.append(score)
    return scores


def load_and_prepare_model(max_seq_length, lora_rank):
    """加载预训练模型并配置LoRA"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./models/Qwen3-4B-Instruct-2507",
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.95,  # Reduce if out of memory
    )

    # 配置LoRA
    # TODO 了解lora的超参数，考察优化的方向
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


def prepare_dataset(tokenizer, system_prompt, max_seq_length):
    """加载和预处理数据集"""
    dataset = load_from_disk("./data/laptop_train")
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f'sentence:"{x["sentence"]}", object:"{x["object"]}"',
                },
            ],
            "answer": x["sentiment"],
        }
    )

    # 过滤样本大小
    tokenized = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=True,
    )
    print(tokenizer.decode(tokenized[0]["tokens"]))
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

    max_prompt_length = int(np.quantile(tokenized["L"], 0.9))  # type: ignore
    print("Max Length = ", max_prompt_length)

    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= max_prompt_length)[0])  # type: ignore
    del tokenized

    max_prompt_length = max_prompt_length + 1  # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length

    print("=" * 50)
    print(f"{max_prompt_length = }, {max_completion_length = }")
    print("=" * 50)

    # 分割训练集和测试集
    splited_dataset = dataset.train_test_split(test_size=0.1)

    return splited_dataset, max_prompt_length, max_completion_length


def setup_vllm_sampling_params(tokenizer):
    """设置vLLM采样参数"""
    return SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )


def setup_training_config(
    vllm_sampling_params, max_prompt_length, max_completion_length
):
    """设置训练参数"""
    return GRPOConfig(
        # vllm configs
        vllm_sampling_params=vllm_sampling_params,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        # training configs
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",  # TODO what to chose?
        logging_dir="logs",
        logging_steps=2,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,  # Increase to 4 for smoother training
        num_generations=16,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=1,  # Set to 1 for a full training run
        # max_steps=200,
        save_steps=50,
        output_dir="outputs",
        # For optional training + evaluation
        fp16_full_eval=True,
        per_device_eval_batch_size=128,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=100,
    )


def setup_callbacks(lora_rank, max_seq_length, training_args, train_set_size):
    """设置各种回调函数"""
    # 初始化邮件通知插件
    email_callback = EmailCallback(
        sender_email="1186977762@qq.com",
        receiver_email="1186977762@qq.com",
        password="wzubregistphgdfc",
        smtp_server="smtp.qq.com",
        port=587,
        language="zh",
    )
    # 注册邮件回调
    swanlab.register_callbacks([email_callback])

    # 初始化SwanLab回调
    swanlab_callback = SwanLabCallback(
        resume=True,
        project="RL for Sentiment Analysis",
        experiment_name="Qwen3-4B-Instruct-GRPO",
        description="测试Qwen3-4B-Instruct-GRPO模型在情感分析任务上的性能",
        config={
            "lora_rank": lora_rank,
            "max_seq_length": max_seq_length,
            "n_epoch": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "train_set_size": train_set_size,
        },
    )

    return swanlab_callback


def train_model(
    model,
    tokenizer,
    reward_funcs,
    training_args,
    callbacks,
    train_dataset,
    eval_dataset,
):
    """创建训练器并训练模型"""
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # 从上次中断处恢复训练
        resume_from_checkpoint=True,
    )
    trainer.train()

    return trainer


def main():
    """主函数，协调各个步骤"""
    # 设置参数
    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower
    system_prompt = "Analyze the sentiment of a certain object from a given text. Reason carefully before giving your response. Your response should end with a single word: positive, negative, or neutral, on a new line."

    # 加载和准备模型
    model, tokenizer = load_and_prepare_model(max_seq_length, lora_rank)

    # 准备数据集
    splited_dataset, max_prompt_length, max_completion_length = prepare_dataset(
        tokenizer, system_prompt, max_seq_length
    )

    # 设置vLLM采样参数
    vllm_sampling_params = setup_vllm_sampling_params(tokenizer)

    # 设置训练参数
    training_args = setup_training_config(
        vllm_sampling_params, max_prompt_length, max_completion_length
    )

    # 设置回调函数
    swanlab_callback = setup_callbacks(
        lora_rank, max_seq_length, training_args, splited_dataset["train"].shape[0]
    )

    # 训练模型
    _ = train_model(
        model,
        tokenizer,
        [reward_answer],
        training_args,
        [swanlab_callback],
        splited_dataset["train"],
        splited_dataset["test"],
    )

    # 保存模型
    model.save_lora("./outputs/grpo_saved_lora")


if __name__ == "__main__":
    main()
