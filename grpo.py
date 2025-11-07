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

torch._dynamo.config.cache_size_limit = 256

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"


def reward_answer(completions, answer, **kwargs):
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


max_seq_length = 1024  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower
system_prompt = "Analyze the sentiment of a certain object from a given text. Reason carefully before giving your response. Your response should end with a single word: positive, negative, or neutral, on a new line."

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/Qwen3-4B-Instruct-2507",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.95,  # Reduce if out of memory
)

# TODO config here can be optimized
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

# Filter only samples smaller than 90% max length
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

max_prompt_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", max_prompt_length)

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= max_prompt_length)[0])
del tokenized

max_prompt_length = max_prompt_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

print(
    "Max Prompt Length = ",
    max_prompt_length,
    " Max Completion Length = ",
    max_completion_length,
)

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)


training_args = GRPOConfig(
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
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # Increase to 4 for smoother training
    num_generations=8,  # Decrease if out of memory
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

# For optional training + evaluation
splited_dataset = dataset.train_test_split(test_size=0.1)


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
        "train_set_size": splited_dataset["train"].shape[0],
    },
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_answer,
    ],
    args=training_args,
    callbacks=[swanlab_callback],
    # train_dataset = dataset,
    # For optional training + evaluation:
    train_dataset=splited_dataset["train"],
    eval_dataset=splited_dataset["test"],
    # 从上次中断处恢复训练
    resume_from_checkpoint=True,
)
trainer.train()
