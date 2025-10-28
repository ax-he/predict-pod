# filename: call_two_models_openrouter_timed.py
# pip install -U openai

import asyncio
from time import perf_counter
from openai import AsyncOpenAI

API_KEY = "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"  # 放你的 Key
BASE_URL = "https://openrouter.ai/api/v1"

MODELS = [
    "tngtech/deepseek-r1t-chimera:free",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
]

MESSAGES = [
    {"role": "system", "content": "你是中文助教，回答要分点精炼。"},
    {"role": "user", "content": "把 13 转成二进制"},
    {"role": "assistant", "content": "步骤：… 结果：1101"},
    {"role": "user", "content": "把 25 转成二进制，并解释步骤"}
]

async def ask_one(client: AsyncOpenAI, model: str):
    start = perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=MESSAGES,
            extra_headers={"HTTP-Referer": "https://example.com", "X-Title": "My Test"},
            extra_body={"usage": {"include": True}},  # 让 OpenRouter回传用量（若支持）
            timeout=120,
        )
        elapsed = perf_counter() - start
        text = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        return {
            "model": model,
            "ok": True,
            "text": text,
            "usage": usage,
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = perf_counter() - start
        return {
            "model": model,
            "ok": False,
            "error": str(e),
            "usage": None,
            "elapsed": elapsed,
        }

def _get_usage_pair(usage):
    """兼容不同 SDK 版本，把 usage 转成 (input_tokens, output_tokens) 或 (None, None)"""
    if usage is None:
        return None, None
    # 既支持属性访问也支持字典
    try:
        tin = getattr(usage, "input_tokens")
        tout = getattr(usage, "output_tokens")
        if tin is not None or tout is not None:
            return tin, tout
    except Exception:
        pass
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    return None, None

async def main():
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = await asyncio.gather(*(ask_one(client, m) for m in MODELS))

    print("\n===== Results =====")
    for r in results:
        print(f"\n=== {r['model']} ===")
        print(f"[time] {r['elapsed']:.3f} s")
        if r["ok"]:
            print(r["text"])
            tin, tout = _get_usage_pair(r["usage"])
            if tin is not None or tout is not None:
                total = (tin or 0) + (tout or 0)
                tps = (total / r["elapsed"]) if r["elapsed"] > 0 else None
                print(f"[usage] input={tin} output={tout} total={total}"
                      + (f" | throughput={tps:.1f} tok/s" if tps is not None else ""))
        else:
            print(f"[ERROR] {r['error']}")

if __name__ == "__main__":
    asyncio.run(main())
