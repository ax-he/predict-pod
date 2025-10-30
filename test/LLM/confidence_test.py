# filename: confidence_gate_legal_hardguard.py
# pip install -U openai
import argparse, json, sys
from typing import TypedDict, Optional
from openai import OpenAI

API_KEY   = "***"  # ← 换成你的
BASE_URL  = "https://openrouter.ai/api/v1"
DEF_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"

# 可选：提供更强的备用模型用于法律/逐字引用（留空则直接置信度不足）
FALLBACK_MODEL = ""  # 例如: "perplexity/sonar-reasoning" 或你信任的更强模型

TASK_ENUM = [
    "arithmetic", "format_conversion", "programming", "math_proof",
    "legal_citation", "time_sensitive", "open_domain"
]

class GatePlan(TypedDict, total=False):
    task_type: str
    needs_external_docs: bool
    needs_verbatim_citations: bool
    self_score: float

def build_client():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

def gate_plan(client: OpenAI, model: str, question: str):
    """返回 (plan, valid)。valid=False → 门控输出无效/空，走回退策略，不当成0分。"""
    sys_prompt = (
        "你没有联网/外部资料访问能力。"
        "只返回一个 JSON 对象，字段："
        f'{{"task_type":"{"|".join(TASK_ENUM)}",'
        '"needs_external_docs":true/false,'
        '"needs_verbatim_citations":true/false,'
        '"self_score":0..1}}。'
        "判定规则：\n"
        "- 纯算术/进制转换/单位换算/格式化 → task_type=arithmetic 或 format_conversion，通常不需要外部资料。\n"
        "- 编程任务 → task_type=programming；若题目要求特定私有仓库/接口且你无法访问，请将 self_score≤0.3，并置 needs_external_docs=true。\n"
        "- 数学证明 → task_type=math_proof；若要求逐字引用文献页码，则 needs_external_docs=true、needs_verbatim_citations=true。\n"
        "- 逐字引用法律/标准并给出条号/页码/官方编号 → task_type=legal_citation；通常需要外部资料且需要逐字引用。当你无法访问原文时 确保self_score≤0.3。\n"
        "- 依赖最新事实(新闻/价格/赛果/版本发布等) → task_type=time_sensitive；通常需要外部资料，拿不准时 self_score≤0.4。\n"
        "- 其余 → task_type=open_domain。严禁输出任何非JSON字符。"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_prompt},
                  {"role":"user","content":question}],
        response_format={
            "type":"json_schema",
            "json_schema":{
                "name":"GatePlan",
                "schema":{
                    "type":"object",
                    "properties":{
                        "task_type":{"type":"string","enum":TASK_ENUM},
                        "needs_external_docs":{"type":"boolean"},
                        "needs_verbatim_citations":{"type":"boolean"},
                        "self_score":{"type":"number","minimum":0,"maximum":1}
                    },
                    "required":["task_type","needs_external_docs","needs_verbatim_citations","self_score"],
                    "additionalProperties":False
                },
                "strict":True
            }
        },
        temperature=0,
        max_tokens=160,
        extra_body={"usage":{"include":True}},
    )

    raw = (resp.choices[0].message.content or "").strip()
    print(f"[gate][raw]={raw!r}")
    try:
        obj = json.loads(raw) if raw else {}
    except Exception:
        obj = {}
    invalid = (raw == "" or raw == "{}" or not all(k in obj for k in (
        "task_type","needs_external_docs","needs_verbatim_citations","self_score"
    )))
    if invalid:
        print("[gate] invalid/empty structured output → treat as gate failure (NOT low confidence).")
        return None, False

    plan: GatePlan = {
        "task_type": obj["task_type"],
        "needs_external_docs": bool(obj["needs_external_docs"]),
        "needs_verbatim_citations": bool(obj["needs_verbatim_citations"]),
        "self_score": float(obj["self_score"]),
    }
    try:
        u = resp.usage
        print(f"[gate][usage] prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
    except Exception:
        pass
    return plan, True

def answer(client: OpenAI, model: str, question: str, role_hint: Optional[str]=None):
    sys_prompt = role_hint or "你是中文问答助手，回答要结构清晰、分点精炼；若无法满足前置约束，请仅输出“置信度不足”。"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_prompt},
                  {"role":"user","content":question}],
        temperature=0.2,
        max_tokens=1200,
        extra_body={"usage":{"include":True}, "reasoning":{"exclude":True}},
    )
    ch = resp.choices[0]
    msg = ch.message
    text = (msg.content or "").strip() or (getattr(msg,"reasoning","") or getattr(msg,"refusal","") or "[EMPTY]").strip()
    print("\n=== 最终回答 ===\n" + text)
    try:
        u = resp.usage
        print(f"\n[answer][finish_reason] {ch.finish_reason}")
        print(f"[answer][usage] prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
    except Exception:
        pass

def route_and_answer(client: OpenAI, model: str, question: str):
    plan, valid = gate_plan(client, model, question)
    if not valid:
        print("[route] Gate unavailable → fallback to answer() directly.")
        return answer(client, model, question)

    t  = plan["task_type"]
    ed = plan["needs_external_docs"]
    vb = plan["needs_verbatim_citations"]
    sc = plan["self_score"]

    # —— 关键修正：法律逐字引用/需外部来源 → 不信自评，强制拒答或升级 ——
    if t == "legal_citation" or ed or vb:
        # 这里不要相信 self_score；因为 ISO 42001 为付费全文（无法核对逐字），OJ 虽公开但需精确条号/页码核验：
        # https://www.iso.org/store.html  /  https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng
        legal_system_prompt = (
            "你是合规顾问。你必须仅依据“用户允许的原始官方文本”逐字核对并给出精确条号/页码/编号。"
            "若你无法直接核对原文或无法确保逐字准确，请只输出“置信度不足”，不得编造或推断。"
        )
        if not FALLBACK_MODEL:
            print("[ABSTAIN] 法律/逐字引用任务：无备用模型 → 输出‘置信度不足’。")
            return print("\n=== 最终回答 ===\n置信度不足")
        # 升级到更强模型（若你配置了）
        print(f"[route] Escalate to fallback model: {FALLBACK_MODEL}")
        return answer(client, FALLBACK_MODEL, question, role_hint=legal_system_prompt)

    # 编程/算术/格式化/数学：中等阈值
    if t in ("arithmetic","format_conversion","programming","math_proof"):
        if sc < 0.60:
            print(f"[ABSTAIN] 自评不足（score={sc:.2f} < 0.60）。")
            return print("\n=== 最终回答 ===\n置信度不足")
        return answer(client, model, question)

    # 开放问答：尊重自评
    if t == "open_domain":
        if sc < 0.50:
            print(f"[ABSTAIN] 开放式问题且自评不足（score={sc:.2f} < 0.50）。")
            return print("\n=== 最终回答 ===\n置信度不足")
        return answer(client, model, question)

    # 未识别类型：保守
    if sc < 0.60:
        print(f"[ABSTAIN] 未识别任务类型且自评不足（score={sc:.2f} < 0.60）。")
        return print("\n=== 最终回答 ===\n置信度不足")
    return answer(client, model, question)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q","--question", type=str, default="把大象放进冰箱需要几步")
    ap.add_argument("-m","--model", type=str, default=DEF_MODEL)
    args = ap.parse_args()

    if API_KEY.startswith("sk-or-xxxx"):
        print("[WARN] 请先在脚本顶部替换 API_KEY。")
        sys.exit(1)

    client = build_client()
    route_and_answer(client, args.model, args.question)

if __name__ == "__main__":
    main()
