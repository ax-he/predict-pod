# filename: confidence_gateA_only_v3.py
# pip install -U openai
import json
from openai import OpenAI

API_KEY  = "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"   # ← 换成你自己的
BASE_URL = "https://openrouter.ai/api/v1"
MODEL    = "deepseek/deepseek-r1-0528-qwen3-8b:free"

QUESTION = """请仅依据《欧盟人工智能法案》正式 OJ 文本（Regulation (EU) 2024/1689）完成下列任务：
1) 逐条引用与“高风险 AI 系统”认定直接相关的条款：Article 6 及 Annex III（请标明条号并给出原文短引）。
2) 假设场景：一家欧盟银行在 2025-10 使用由大语言模型驱动的多模态系统，对贷款申请人进行自动化风险评分，并结合面部表情识别作为 KYC 证据；模型的分数直接决定是否拒贷，无人工复核。请判定该系统是否属于高风险 AI，给出逐条法条依据（需要条号+短引）。
3) 列出该系统需遵守的强制义务清单（如风险管理、数据治理与数据集质量、技术文档、日志、透明度/信息提供、人工监督、准确性/稳健性、合规评估/CE 标识、事件报告等），并为每一项对应具体条款或附件条目编号。"""

THRESH   = 0.70  # 门控阈值

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def gate_score_json(question: str) -> float | None:
    """仅用 Gate A（JSON 模式）拿到 {"score": <0..1>}。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "只返回一个 JSON 对象：{\"score\": <0到1的小数>}。"
                    "不要输出任何多余字符（含解释、换行、markdown）。"
                )
            },
            {"role": "user", "content": question}
        ],
        response_format={"type": "json_object"},  # 强制 JSON 模式
        temperature=0,
        max_tokens=16,
        extra_body={"usage": {"include": True}},
    )
    raw = resp.choices[0].message.content or ""
    # 如果需要可打开这一行看完整响应：
    # print(resp.model_dump_json(indent=2, ensure_ascii=False))

    try:
        obj = json.loads(raw)
        score = float(obj["score"])
        return score if 0.0 <= score <= 1.0 else None
    except Exception:
        return None

def answer(question: str):
    """回答阶段：优先 content，若空则回退 reasoning/refusal，并打印 finish_reason/usage 便于排查。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "你是中文助教，回答要分点、精炼、引用要带条号与短引。务必在最后输出可直接引用的结论列表。"},
            {"role": "user", "content": question},
        ],
        # 法规类问题内容很长，把预算放大；同时请求不要在响应里返回思考文本
        max_tokens=1200,
        temperature=0.2,
        extra_body={
            "usage": {"include": True},
            "reasoning": {"exclude": True}  # 仅隐藏返回，不影响是否使用 reasoning。文档见链接。 
        }
    )

    choice = resp.choices[0]
    msg = choice.message
    text = (msg.content or "").strip()

    # 兼容回退：某些“thinking”模型可能只给 reasoning/refusal
    if not text:
        # OpenRouter 文档：reasoning tokens 会出现在 message.reasoning 字段；有些路由也可能用 refusal 字段。 
        # https://openrouter.ai/docs/use-cases/reasoning-tokens  &  https://openrouter.ai/docs/api-reference/overview
        reason = getattr(msg, "reasoning", None)
        refusal = getattr(msg, "refusal", None)
        if isinstance(reason, str) and reason.strip():
            text = "[fallback: reasoning]\n" + reason.strip()
        elif isinstance(refusal, str) and refusal.strip():
            text = "[refusal]\n" + refusal.strip()

    print("\n=== 最终回答 ===")
    print(text if text else "[EMPTY CONTENT FROM MODEL]")

    # 打印 finish_reason 与 token 用量，判断是否被长度截断
    try:
        fr = choice.finish_reason
        usage = resp.usage
        print(f"\n[finish_reason] {fr}")
        if usage:
            # OpenRouter 规范化的 usage 字段（非原生 tokenizer），详见 API 文档
            print(f"[usage] prompt={usage.prompt_tokens}  completion={usage.completion_tokens}  total={usage.total_tokens}")
    except Exception:
        pass

def main():
    score = gate_score_json(QUESTION)
    print(f"[gate] score={score:.2f}" if score is not None else "[gate] score=None")

    if score is None:
        print("[gate] 未能解析 gate A 分数，默认继续推理。")
        answer(QUESTION)
    elif score < THRESH:
        print(f"[ABSTAIN] 置信度不足（score={score:.2f} < {THRESH}），终止推理。")
    else:
        print(f"[gate] 通过门控（score={score:.2f} ≥ {THRESH}），开始推理。")
        answer(QUESTION)

if __name__ == "__main__":
    main()
