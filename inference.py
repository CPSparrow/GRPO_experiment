import pickle

import pandas as pd


def print_outputs(outputs):
    print("\nGenerated Outputs:\n" + "-" * 80)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\n")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)


def load_data(test_name: str) -> pd.DataFrame:
    """导入并且简单处理数据

    Args:
        n_sample (int, optional): 处理的数据行数. Defaults to -1.

    Returns:
        pd.DataFrame: 处理后的数据集
    """
    if test_name == "laptop":
        with open(
            "./data/preprocessed/laptops_base_google-flan-t5-base.pkl", "rb"
        ) as f:
            data = pickle.load(f)
    elif test_name == "restaurant":
        with open(
            "./data/preprocessed/restaurants_base_google-flan-t5-base.pkl", "rb"
        ) as f:
            data = pickle.load(f)
    else:
        raise ValueError("test_name must be 'laptop' or 'restaurant'")
    data = pd.DataFrame(data)
    data.columns = ["sentence", "object", "sentiment", "is implicit"]
    data = data.sort_index()
    return data


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


def gather_outputs(index_list: pd.Index, output_list: list):
    data = list()
    for index, output in zip(index_list, output_list):
        for item in output.outputs:
            data.append(
                dict(index=index, output=item.text, output_tokens=len(item.token_ids))
            )
    return pd.DataFrame(data)


def main(test_name: str, n_try: int):
    from vllm import LLM  # type: ignore

    llm = LLM(
        model="./models/Qwen3-4B-Instruct-2507",
        max_model_len=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
    )

    sampling_params = llm.get_default_sampling_params()
    sampling_params.n = n_try
    sampling_params.temperature = 1
    sampling_params.top_p = 0.9
    sampling_params.max_tokens = 1024

    data = load_data(test_name)
    input_conversations = make_conversation_list(data)

    outputs = llm.chat(input_conversations, sampling_params, use_tqdm=True)

    output_data = gather_outputs(data.index, outputs)
    output_data.to_parquet("./data/output/laptop_test.parquet")

    print_outputs(outputs)


if __name__ == "__main__":
    # inference config
    n_try = 128
    test_name = "restaurant"

    main(test_name, n_try)
