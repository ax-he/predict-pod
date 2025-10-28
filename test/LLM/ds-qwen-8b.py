import requests
import json

# 直接把你的 Key 写在这里（注意不要把此文件提交到公共仓库）
API_KEY = "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"

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "messages": [
        {"role": "system", "content": "你是中文助教，回答要分点精炼。"},
        # 可选 few-shot：用 assistant 演示理想风格（不是工具！只是示例）
        {"role": "user", "content": "把 13 转成二进制"},
        {"role": "assistant", "content": "步骤：… 结果：1101"},
        # 正式问题
        {"role": "user", "content": "把 25 转成二进制，并解释步骤"}
    ]
        
  }),
  timeout=60,
)

# 基础的报错与输出处理
try:
    response.raise_for_status()
    data = response.json()
    print(data["choices"][0]["message"]["content"])
except Exception:
    # 如果接口或模型不可用/ID变更，会打印完整响应便于排查
    print("Request failed or unexpected response:")
    print(response.status_code, response.text)