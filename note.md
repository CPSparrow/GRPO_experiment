# 实用命令

## 设置代码运行的环境变量

1. 激活环境

```bash
source .venv/bin/activate
```

2. 运行代码

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
```

3. 清华源

```bash
--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

4. 监控GPU占用

```bash
watch -n 1 -d nvidia-smi
```

# 依赖安装和环境配置

创建并激活 uv 环境：

```bash
uv venv --python 3.12 --seed
```

默认清华源：

```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

基于 vllm 的文档，使用以下命令安装：

> flash infer似乎装不上，所以暂时不使用

```bash
uv pip install vllm --torch-backend=auto --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install modelscope
# uv pip install flashinfer-python flashinfer-cubin  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
uv pip install unsloth --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

检查cuda版本：

```python
print("CUDA版本:", torch.version.cuda)
```


下载 Qwen3_4B 以及4B_Base (待定)

```bash
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ./models/Qwen3-4B-Instruct-2507
```

# 环境测试

运行：

```bash
vllm serve ./models/Qwen3-4B-Instruct-2507 \
--dtype bfloat16 \
--max-model-len 3096 
```

基础测试：

```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "./Qwen3-4B-Instruct-2507",
    "prompt": "San Francisco is a",
    "max_tokens": 200,
    "temperature": 0
}'
```

示例运行：
```bash
curl --request POST \
  --url http://localhost:8000/v1/chat/completions \
  --header 'Content-Type: application/json' \
  --data '{
  "model": "./models/Qwen3-4B-Instruct-2507",
  "messages": [
    {
      "role": "system",
      "content": "Analyze the sentiment of a certain object from a given text. Reason carefully before giving your response. Your response should end with a single word: positive, negative, or neutral, on a new line."
    },
    {
      "role": "user",
      "content": "sentence:\"boot time is super fast, around anywhere from 35 seconds to 1 minute.\",object:Boot time"
    }
  ],
  "max_tokens": 1024,
  "stream": false
}'
```
## 原先对于学校服务器环境的笔记：

**说明：**

流程是先用conda创建基础环境，然后在该环境下安装pipx(用于安装uv)。具体的py环境通过uv来管理。

手动激活uv的方式是输入以下命令：

> 目前的安装不会加入到PATH中，而且我们也不应该这样做。

```bash
export PATH=/root/.local/bin:$PATH
```
