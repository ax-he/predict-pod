# q2o_predictor.py (DeepSeek R1 7B 优先 & 可选长度提示覆盖)
# 目标：输入自然语言“问题” -> 自动估算输入 token S、问题类型、输出 token O 的 p50/p90
# 更新要点：
# - 优先使用 HuggingFace AutoTokenizer 加载 DeepSeek R1 7B 的 tokenizer（可用 --tok 或 Q2O_TOKENIZER 指定）。
# - count() 支持可选把对话按 chat template 包装后再计数（--with_chat_template）。
# - 可选解析“100字左右”这类长度提示，直接覆盖 O 的 p50/p90 预算（默认开启，--no_len_hint 可关闭）。
# - 仍支持：tiktoken(cl100k_base) 作为次级回退；再无则字符近似。

import os
import re
import math
import json
import argparse
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

# 可选依赖：tiktoken、transformers、scikit-learn
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

try:
    from transformers import AutoTokenizer
    _HAS_HF = True
except Exception:
    _HAS_HF = False

try:
    from sklearn.linear_model import LinearRegression, QuantileRegressor
    from sklearn.preprocessing import OneHotEncoder
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# --------------------------
# 0) 实用工具：解析“xx字”
# --------------------------
def parse_len_hint(question: str) -> Optional[int]:
    """
    从“100字”“约120字”“大概80字”这类提示里抽取阿拉伯数字。
    中文通常可近似 1字≈1 token（并非严格等价，但足够用于预算）。
    """
    if not question:
        return None
    m = re.search(r"(\d+)\s*字", question)
    return int(m.group(1)) if m else None


# --------------------------
# 1) 估算输入 token 数 S
# --------------------------
class TokenCounter:
    def __init__(self,
                 prefer_hf_model: Optional[str] = None,
                 use_chat_template: bool = False):
        """
        prefer_hf_model: 显式指定 HF tokenizer 名（优先级最高）
        use_chat_template: 若 True，按 chat 模板包装后计数（更贴近你线上 Chat 模式）
        """
        self.enc = None
        self.hf_tok = None
        self.model_name = None
        self.use_chat_template = use_chat_template

        # 1) 优先：尝试加载 DeepSeek R1 7B 或用户指定的 HF tokenizer
        if _HAS_HF:
            candidates = []
            if prefer_hf_model:
                candidates.append(prefer_hf_model)
            # 环境变量兜底
            env1 = os.getenv("Q2O_TOKENIZER")
            env2 = os.getenv("HF_TOKENIZER")
            for x in (env1, env2):
                if x:
                    candidates.append(x)
            # 常见 DeepSeek 名称兜底（顺序从更精确到更宽泛）
            candidates.extend([
                "deepseek-ai/DeepSeek-R1-7B",   # 若本地已拉取对应仓库，优先使用
                "deepseek-ai/DeepSeek-R1",      # 同一系列 tokenizer 通常兼容
                "deepseek-ai/DeepSeek-V2-7B",
                "deepseek-ai/DeepSeek-V2",
            ])
            # 逐个尝试
            for name in candidates:
                try:
                    self.hf_tok = AutoTokenizer.from_pretrained(name, use_fast=True)
                    self.model_name = name
                    break
                except Exception:
                    continue

        # 2) 次选：tiktoken(cl100k_base)
        if self.hf_tok is None and _HAS_TIKTOKEN:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.enc = None

        # 3) 末选：随便来个 HF tokenizer（gpt2）
        if self.hf_tok is None and self.enc is None and _HAS_HF:
            try:
                self.hf_tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
                self.model_name = "gpt2"
            except Exception:
                self.hf_tok = None

    def count(self, text: str) -> int:
        text = text or ""
        # HF tokenizer 分支
        if self.hf_tok is not None:
            try:
                if self.use_chat_template:
                    # 把输入当作 user 消息，通过 chat template 包装后计数，更贴近 Chat 推理真实入口
                    messages = [{"role": "user", "content": text}]
                    prompt = self.hf_tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    return len(self.hf_tok(prompt, add_special_tokens=False).input_ids)
                # 纯文本计数（不加 BOS/EOS）
                return len(self.hf_tok.encode(text, add_special_tokens=False))
            except Exception:
                # 任何失败回退到纯文本/次级方案
                try:
                    return len(self.hf_tok.encode(text, add_special_tokens=False))
                except Exception:
                    pass

        # tiktoken 分支
        if self.enc is not None:
            try:
                return len(self.enc.encode(text))
            except Exception:
                pass

        # 极简退化：按字符估算（混合文本 1 token≈3.5 chars）
        return max(1, int(math.ceil(len(text) / 3.5)))


# --------------------------
# 2) 轻量“问题类型”识别
# --------------------------
TYPE_RULES = [
    ("translate",  r"(^|\b)(translate|翻译|译为|译成|英文|中文)\b"),
    ("summarize",  r"(^|\b)(总结|概括|摘要|总结一下|tl;dr|tldr|简述)\b"),
    ("code",       r"(^|\b)(代码|实现|bug|报错|stack trace|traceback|编写|写个|用.*(python|cpp|java|rust))"),
    ("math",       r"(^|\b)(计算|求值|方程|证明|derive|积分|微分|log|概率|统计|公式)"),
    ("qa_explain", r"(^|\b)(为什么|原理|对比|比较|区别|如何.*解释|讲解|分析)"),
    ("qa_short",   r".*"),  # 兜底
]

def detect_task_type(question: str) -> str:
    q = (question or "").lower().strip()
    for t, pat in TYPE_RULES:
        if re.search(pat, q, flags=re.I):
            return t
    return "qa_short"


# --------------------------
# 3) 输出长度预测器（历史数据优先；否则用内置先验）
# --------------------------
class LengthPredictor:
    def __init__(self,
                 csv_path: Optional[str] = None,
                 tasks_for_prior: Optional[Dict[str, Dict[str, float]]] = None):
        self.ohe: Optional['OneHotEncoder'] = None
        self.reg_mean = None
        self.reg_q50  = None
        self.reg_q90  = None
        self.has_model = False
        self.tasks_prior = tasks_for_prior or self._default_priors()

        if csv_path and os.path.exists(csv_path) and _HAS_SK:
            df = pd.read_csv(csv_path)
            if "task" not in df.columns:
                df["task"] = "qa_short"
            df = df[["S", "O", "task"]].dropna()
            self._fit_models(df)
            self.has_model = True

    def _default_priors(self) -> Dict[str, Dict[str, float]]:
        # 可按你历史日志微调
        return {
            "translate": {"p50_a": 1.00, "p50_b": 0,  "p90_a": 1.20, "p90_b": 10},
            "summarize": {"p50_a": 0.30, "p50_b": 30, "p90_a": 0.45, "p90_b": 50},
            "code":      {"p50_a": 0.65, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "math":      {"p50_a": 0.55, "p50_b": 30, "p90_a": 0.80, "p90_b": 60},
            "qa_explain":{"p50_a": 0.60, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "qa_short":  {"p50_a": 0.35, "p50_b": 40, "p90_a": 0.55, "p90_b": 80},
        }

    def _build_features(self, df: pd.DataFrame):
        S = df["S"].to_numpy().reshape(-1, 1)
        logS = np.log1p(df["S"].to_numpy()).reshape(-1, 1)
        task = df["task"].fillna("qa_short").to_numpy().reshape(-1, 1)

        # 兼容新版/旧版 scikit-learn
        kw = {"handle_unknown": "ignore"}
        try:
            self.ohe = OneHotEncoder(sparse_output=False, **kw)  # >=1.2
        except TypeError:
            self.ohe = OneHotEncoder(sparse=False, **kw)         # <1.2

        task_oh = self.ohe.fit_transform(task)
        X = np.hstack([np.ones_like(S), S, logS, task_oh])
        y = df["O"].to_numpy()
        return X, y

    def _x_one(self, S: int, task: str):
        S = max(1, int(S))
        base = np.array([[1.0, float(S), math.log1p(S)]])
        task_oh = self.ohe.transform(np.array([[task]])) if self.ohe is not None else np.zeros((1,1))
        return np.hstack([base, task_oh])

    def _fit_models(self, df: pd.DataFrame):
        X, y = self._build_features(df)
        self.reg_mean = LinearRegression()
        self.reg_q50  = QuantileRegressor(quantile=0.5, alpha=1.0, fit_intercept=False, solver="highs")
        self.reg_q90  = QuantileRegressor(quantile=0.9, alpha=0.1, fit_intercept=False, solver="highs")

        self.reg_mean.fit(X, y)
        self.reg_q50.fit(X, y)
        self.reg_q90.fit(X, y)

    def predict(self, S: int, task: str) -> Dict[str, float]:
        if self.has_model and self.ohe is not None:
            x = self._x_one(S, task)
            o_mean = float(self.reg_mean.predict(x)[0])
            o_p50  = float(self.reg_q50.predict(x)[0])
            o_p90  = float(self.reg_q90.predict(x)[0])
        else:
            pri = self.tasks_prior.get(task, self.tasks_prior["qa_short"])
            o_p50 = pri["p50_a"] * S + pri["p50_b"]
            o_p90 = pri["p90_a"] * S + pri["p90_b"]
            o_mean = 0.5 * (o_p50 + o_p90)

        # 合理裁剪
        o_mean = float(np.clip(o_mean, 1, 32768))
        o_p50  = float(np.clip(o_p50,  1, 32768))
        o_p90  = float(np.clip(o_p90,  1, 32768))
        return {"O_mean": o_mean, "O_p50": o_p50, "O_p90": o_p90}


# --------------------------
# 4) 将三步串起来：文本 -> (S, task, O_hat)
# --------------------------
def estimate_from_question(question: str,
                           history_csv: Optional[str] = None,
                           prefer_hf_model: Optional[str] = None,
                           use_chat_template: bool = False,
                           respect_len_hint: bool = True) -> Dict[str, object]:
    tc = TokenCounter(prefer_hf_model=prefer_hf_model,
                      use_chat_template=use_chat_template)
    S = tc.count(question)
    task = detect_task_type(question)
    predictor = LengthPredictor(csv_path=history_csv)
    pred = predictor.predict(S, task)

    # 若命中“xx字”提示，则按字数直接覆盖 O 的 p50/p90 预算（默认开启）
    if respect_len_hint:
        L = parse_len_hint(question)
        if L:
            tgt = max(1, int(L))
            pred["O_p50"] = float(tgt)
            pred["O_p90"] = float(int(round(tgt * 1.2)))  # 给个 20% 的 p90 余量
            pred["O_mean"] = 0.5 * (pred["O_p50"] + pred["O_p90"])

    return {
        "S": S,
        "task": task,
        "tokenizer": tc.model_name or ("tiktoken/cl100k_base" if tc.enc else "approx"),
        **pred
    }


# --------------------------
# 5) CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=str, required=True, help="输入的问题文本")
    ap.add_argument("--hist", type=str, default="", help="可选：历史日志CSV(列: S,O,task)")
    ap.add_argument("--tok", type=str, default="", help="可选：显式指定HF tokenizer名，如 deepseek-ai/DeepSeek-R1-7B")
    ap.add_argument("--with_chat_template", action="store_true",
                    help="按chat模板包装后计数，更贴近Chat场景")
    ap.add_argument("--no_len_hint", action="store_true",
                    help="关闭“xx字”长度提示对 O 的覆盖")

    ap.add_argument("--show_json", action="store_true", help="以 JSON 形式输出")
    args = ap.parse_args()

    out = estimate_from_question(
        question=args.q,
        history_csv=(args.hist if args.hist else None),
        prefer_hf_model=(args.tok if args.tok else None),
        use_chat_template=args.with_chat_template,
        respect_len_hint=(not args.no_len_hint),
    )

    if args.show_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"Tokenizer : {out['tokenizer']}")
        print(f"S (tokens): {out['S']}")
        print(f"task type : {out['task']}")
        print(f"O_mean    : {out['O_mean']:.1f}")
        print(f"O_p50     : {out['O_p50']:.1f}")
        print(f"O_p90     : {out['O_p90']:.1f}")


if __name__ == "__main__":
    main()
