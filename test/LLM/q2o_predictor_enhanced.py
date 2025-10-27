#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q2o_predictor_enhanced.py — Scoring-based CJK-safe TaskTypeDetector + local DeepSeek tokenizer
"""

import os
import re
import math
import json
import argparse
import logging
from typing import Tuple, Dict, Optional, List
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("q2o")

# -------- optional deps --------
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False
    logger.warning("tiktoken 不可用，将尝试其他 tokenizer")

try:
    from transformers import AutoTokenizer
    _HAS_HF = True
except Exception:
    _HAS_HF = False
    logger.warning("transformers 不可用，无法使用 HF tokenizer")

try:
    from sklearn.linear_model import LinearRegression, QuantileRegressor
    from sklearn.preprocessing import OneHotEncoder
    _HAS_SK = True
except Exception:
    _HAS_SK = False
    logger.warning("scikit-learn 不可用，将仅使用先验（不训练回归）")


# --------------------------
# dataclass
# --------------------------
@dataclass
class PredictionResult:
    S: int
    task: str
    tokenizer: str
    O_mean: float
    O_p50: float
    O_p90: float
    confidence: float
    used_len_hint: bool = False
    len_hint_value: Optional[int] = None


# --------------------------
# 1) Length hint
# --------------------------
class LengthHintParser:
    @staticmethod
    def parse_len_hint(question: str) -> Optional[Tuple[int, str]]:
        if not question:
            return None
        ql = question.lower()

        patterns = [
            # 中文
            (r"(\d+)\s*字", "字", 1.0),
            (r"(\d+)\s*词", "词", 1.0),
            (r"(\d+)\s*句", "句", 15.0),
            (r"(\d+)\s*段", "段", 100.0),
            (r"(\d+)\s*行", "行", 20.0),
            (r"(\d+)\s*页", "页", 500.0),
            # 英文
            (r"(\d+)\s*words?", "words", 0.75),
            (r"(\d+)\s*sentences?", "sentences", 15.0),
            (r"(\d+)\s*paragraphs?", "paragraphs", 100.0),
            (r"(\d+)\s*lines?", "lines", 20.0),
            (r"(\d+)\s*pages?", "pages", 500.0),
            # 范围与模糊
            (r"(\d+)\s*到\s*(\d+)\s*字", "字范围", 1.0),
            (r"(\d+)\s*-\s*(\d+)\s*words?", "word范围", 0.75),
            (r"(?:约|大概|大约|左右)\s*(\d+)\s*字", "约字", 1.0),
            (r"(?:about|approximately|around)\s*(\d+)\s*words?", "约words", 0.75),
        ]

        for pat, unit, coef in patterns:
            m = re.search(pat, ql, flags=re.I)
            if not m:
                continue
            try:
                if m.lastindex and m.lastindex >= 2:
                    start, end = int(m.group(1)), int(m.group(2))
                    avg = (start + end) / 2.0
                    return int(round(avg * coef)), f"{unit}({start}-{end})"
                else:
                    num = int(m.group(1))
                    return int(round(num * coef)), unit
            except Exception:
                continue
        return None

    @staticmethod
    def has_length_constraint(question: str) -> bool:
        return bool(re.search(r"(字|词|句|段|行|页|words?|sentences?|paragraphs?|lines?|pages?|约|大概|左右|about|approximately|around)",
                              question or "", flags=re.I))


# --------------------------
# 2) TokenCounter
# --------------------------
class TokenCounter:
    def __init__(self, prefer_hf_model: Optional[str] = None, use_chat_template: bool = False):
        self.enc = None
        self.hf_tok = None
        self.model_name = None
        self.use_chat_template = use_chat_template
        self.errors: List[str] = []

        # A) 本地目录/文件优先（仅本地，不联网）
        if prefer_hf_model:
            p = Path(prefer_hf_model)
            if p.is_file():
                p = p.parent
            if p.is_dir() and _HAS_HF:
                try:
                    logger.info(f"从本地目录加载 tokenizer: {p}")
                    self.hf_tok = AutoTokenizer.from_pretrained(str(p), use_fast=True, local_files_only=True)
                    self.model_name = str(p.resolve())
                    return
                except Exception as e:
                    logger.warning(f"本地目录加载失败，继续候选：{e}")

        # B) 远程候选（兜底，需要网络）
        if _HAS_HF and self.hf_tok is None:
            for name in self._get_tokenizer_candidates():
                try:
                    logger.info(f"尝试加载tokenizer: {name}")
                    self.hf_tok = AutoTokenizer.from_pretrained(name, use_fast=True)
                    self.model_name = name
                    logger.info(f"成功加载tokenizer: {name}")
                    break
                except Exception as e:
                    logger.warning(f"加载 {name} 失败：{e}")
                    continue

        # C) tiktoken
        if self.hf_tok is None and _HAS_TIKTOKEN:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
                logger.info("使用 tiktoken(cl100k_base)")
            except Exception as e:
                self.enc = None
                self.errors.append(f"tiktoken 失败: {e}")

        # D) gpt2 回退
        if self.hf_tok is None and self.enc is None and _HAS_HF:
            try:
                self.hf_tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
                self.model_name = "gpt2"
                logger.info("回退到 gpt2 tokenizer")
            except Exception as e:
                self.hf_tok = None
                self.errors.append(f"gpt2 失败: {e}")

        if self.hf_tok is None and self.enc is None:
            logger.warning("所有 tokenizer 都失败，将使用字符近似法")

    def _get_tokenizer_candidates(self) -> List[str]:
        cands: List[str] = []
        for env in ["Q2O_TOKENIZER", "HF_TOKENIZER", "TOKENIZER_NAME"]:
            v = os.getenv(env)
            if v:
                cands.append(v)
        cands.extend([
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V2-7B",
            "deepseek-ai/DeepSeek-V2",
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "deepseek-ai/DeepSeek-LLM-7B-Chat",
        ])
        return cands

    def count(self, text: str) -> int:
        if not isinstance(text, str) or not text.strip():
            return 1
        text = text.strip()
        errors = []

        if self.hf_tok is not None:
            try:
                if self.use_chat_template:
                    messages = [{"role": "user", "content": text}]
                    # 只统计输入侧（不在尾部加 assistant 起始提示）
                    prompt = self.hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    tokens = self.hf_tok(prompt, add_special_tokens=False).input_ids
                else:
                    tokens = self.hf_tok.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                errors.append(f"HF 分支失败：{e}")
                try:
                    tokens = self.hf_tok.encode(text, add_special_tokens=False)
                    return len(tokens)
                except Exception as e2:
                    errors.append(f"HF 纯文本回退失败：{e2}")

        if self.enc is not None:
            try:
                return len(self.enc.encode(text))
            except Exception as e:
                errors.append(f"tiktoken 失败：{e}")

        if errors:
            logger.warning("Tokenizer errors: " + "; ".join(errors))

        char_count = len(text)
        token_estimate = max(1, int(math.ceil(char_count / 3.5)))
        logger.info(f"字符近似法: {char_count} chars -> ~{token_estimate} tokens")
        return token_estimate


# --------------------------
# 3) TaskTypeDetector — scoring + conflict resolution (CJK-safe)
# --------------------------
class TaskTypeDetector:
    def __init__(self):
        # 优先级用于同分时的决策：解释 > 代码 > 总结 > 翻译 > 数学 > 创作 > 短问
        self.priority = ["qa_explain", "code", "summarize", "translate", "math", "creative", "qa_short"]

        # 预编译常用正则（中文不加 \b；英文用 ASCII 边界）
        self.rx = {
            "ascii_word_python": re.compile(r"(?<![A-Za-z0-9_])(python|java|cpp|c\+\+|c#|rust|go|javascript|typescript|js|ts|html|css|sql|pytorch|torch|numpy|pandas|regex|shell|bash|docker|kubernetes|git)(?![A-Za-z0-9_])", re.I),
            "code_shape": re.compile(r"(def |class |import |return |function )", re.I),
            "code_zh": re.compile(r"(代码|编程|实现|函数|方法|算法|调试|报错|异常|栈跟踪|堆栈|脚本|接口|库|框架|类|对象|模块|依赖|缓存|链表|哈希|字典|映射|递归|并发|多线程|协程|LRU|LRU缓存)"),
            "math_strong": re.compile(r"(导数|积分|微分|方程|概率分布|矩阵|向量|期望|方差|证明|公式|泰勒|傅里叶|拉普拉斯|最大似然|贝叶斯|极限|梯度|范数|内积|外积)"),
            "math_formula": re.compile(r"(\d+\s*[\+\-\*/]\s*\d+|\d+\s*\^\s*\d+|[≈≃≅≡≠≤≥∞∑∏√∫∂∆π]|\\frac|\\sum|\\int|\\alpha|\\beta|\\gamma)"),
            "math_weak": re.compile(r"(计算|概率|统计|期望|方差)"),
            "explain": re.compile(r"(为什么|为何|如何|怎么|原理|机制|原因|解释|区别|不同|对比|比较|difference|compare|contrast|explain|why|how)", re.I),
            "summarize": re.compile(r"(总结|概括|摘要|概要|简述|tl;dr|tldr|summarize|summary|主要观点|核心内容|大意)", re.I),
            "translate": re.compile(r"(翻译|译为|译成|翻译成|translate|translation)", re.I),
            "creative": re.compile(r"(写一篇|创作|故事|小说|诗歌|文章|essay|article|story|poem|想象|假如|如果|假设|开头|结尾|情节|角色)", re.I),
        }

    @staticmethod
    def _approx_units(raw: str) -> int:
        latin_words = re.findall(r"[A-Za-z0-9_]+", raw or "")
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", raw or "")
        return len(latin_words) + len(cjk_chars)

    def detect(self, question: str) -> str:
        if not question:
            return "qa_short"
        q = question or ""

        scores = {
            "code": 0.0, "math": 0.0, "translate": 0.0, "summarize": 0.0,
            "qa_explain": 0.0, "creative": 0.0, "qa_short": 0.0
        }

        # —— 加分：代码 —— #
        if self.rx["ascii_word_python"].search(q):
            scores["code"] += 2.5
        if self.rx["code_shape"].search(q):
            scores["code"] += 2.0
        if self.rx["code_zh"].search(q):
            scores["code"] += 2.0

        # —— 加分：数学 —— #
        if self.rx["math_strong"].search(q):
            scores["math"] += 3.0
        if self.rx["math_formula"].search(q):
            scores["math"] += 3.0
        # “计算/概率/统计”等弱特征只给低分，避免单词误触发
        if self.rx["math_weak"].search(q):
            scores["math"] += 0.8

        # —— 加分：解释/总结/翻译/创作 —— #
        if self.rx["explain"].search(q):
            scores["qa_explain"] += 3.2
        if self.rx["summarize"].search(q):
            scores["summarize"] += 2.5
        if self.rx["translate"].search(q):
            scores["translate"] += 2.5
        if self.rx["creative"].search(q):
            scores["creative"] += 2.0

        # —— 短问题降权 —— #
        units = self._approx_units(q)
        if units <= 8:
            scores["qa_short"] += 1.0

        # —— 冲突消解 —— #
        # 若出现解释型词而数学仅由弱特征触发（没有强公式/强数学词），则解释优先、数学减半
        has_explain = scores["qa_explain"] > 0
        has_math_strong = bool(self.rx["math_strong"].search(q) or self.rx["math_formula"].search(q))
        if has_explain and not has_math_strong and scores["math"] > 0:
            scores["math"] *= 0.5
            scores["qa_explain"] += 0.8  # 轻微加权，让解释更占优

        # 同理：解释 vs 代码（例如“请解释这段 Python 代码”）
        if has_explain and scores["code"] > 0:
            scores["qa_explain"] += 0.4

        # —— 选最大分；同分用优先级 —— #
        best = max(scores.items(), key=lambda kv: (kv[1], -self.priority.index(kv[0])))
        return best[0]


# --------------------------
# 4) Length predictor
# --------------------------
class LengthPredictor:
    def __init__(self, csv_path: Optional[str] = None, tasks_for_prior: Optional[Dict[str, Dict[str, float]]] = None):
        self.ohe: Optional['OneHotEncoder'] = None
        self.reg_mean = None
        self.reg_q50 = None
        self.reg_q90 = None
        self.has_model = False
        self.tasks_prior = tasks_for_prior or self._default_priors()

        if csv_path and os.path.exists(csv_path) and _HAS_SK:
            try:
                df = pd.read_csv(csv_path)
                required = {"S", "O"}
                if not required.issubset(df.columns):
                    raise ValueError(f"历史CSV缺少列，至少需要 {required}")
                df = df.copy()
                for c in ["S", "O"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                if "task" not in df.columns:
                    df["task"] = "qa_short"
                df["task"] = df["task"].fillna("qa_short").astype(str)
                df = df[["S", "O", "task"]].dropna()
                if len(df) > 10:
                    self._fit_models(df)
                    self.has_model = True
                    logger.info(f"基于历史数据训练回归模型，样本数: {len(df)}")
                else:
                    logger.warning("历史数据不足，使用先验")
            except Exception as e:
                logger.error(f"加载历史数据失败，使用先验：{e}")

    def _default_priors(self) -> Dict[str, Dict[str, float]]:
        return {
            "translate": {"p50_a": 1.00, "p50_b": 0,  "p90_a": 1.20, "p90_b": 10},
            "summarize": {"p50_a": 0.30, "p50_b": 30, "p90_a": 0.45, "p90_b": 50},
            "code":      {"p50_a": 0.65, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "math":      {"p50_a": 0.55, "p50_b": 30, "p90_a": 0.80, "p90_b": 60},
            "qa_explain":{"p50_a": 0.60, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "qa_short":  {"p50_a": 0.35, "p50_b": 40, "p90_a": 0.55, "p90_b": 80},
            "creative":  {"p50_a": 0.80, "p50_b": 100, "p90_a": 1.20, "p90_b": 200},
        }

    def _build_features(self, df: pd.DataFrame):
        S = df["S"].to_numpy().reshape(-1, 1)
        logS = np.log1p(df["S"].to_numpy()).reshape(-1, 1)
        task = df["task"].fillna("qa_short").to_numpy().reshape(-1, 1)
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        task_oh = self.ohe.fit_transform(task)
        X = np.hstack([np.ones_like(S), S, logS, task_oh])
        y = df["O"].to_numpy()
        return X, y

    def _x_one(self, S: int, task: str):
        S = max(1, min(int(S), 100000))
        base = np.array([[1.0, float(S), math.log1p(S)]])
        if self.ohe is not None:
            try:
                task_oh = self.ohe.transform(np.array([[task]]))
            except Exception:
                task_oh = np.zeros((1, self.ohe.transform(np.array([["qa_short"]])).shape[1]))
        else:
            task_oh = np.zeros((1, 1))
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
        if S <= 0:
            return {"O_mean": 10.0, "O_p50": 5.0, "O_p90": 20.0}
        S = max(1, min(S, 100000))
        if self.has_model and self.ohe is not None:
            try:
                x = self._x_one(S, task)
                o_mean = float(self.reg_mean.predict(x)[0])
                o_p50  = float(self.reg_q50.predict(x)[0])
                o_p90  = float(self.reg_q90.predict(x)[0])
            except Exception as e:
                logger.warning(f"回归预测失败，使用先验：{e}")
                pri = self.tasks_prior.get(task, self.tasks_prior["qa_short"])
                o_p50 = pri["p50_a"] * S + pri["p50_b"]
                o_p90 = pri["p90_a"] * S + pri["p90_b"]
                o_mean = 0.5 * (o_p50 + o_p90)
        else:
            pri = self.tasks_prior.get(task, self.tasks_prior["qa_short"])
            o_p50 = pri["p50_a"] * S + pri["p50_b"]
            o_p90 = pri["p90_a"] * S + pri["p90_b"]
            o_mean = 0.5 * (o_p50 + o_p90)

        o_mean = float(np.clip(o_mean, 1, 32768))
        o_p50  = float(np.clip(o_p50,  1, 32768))
        o_p90  = float(np.clip(o_p90,  1, 32768))
        return {"O_mean": o_mean, "O_p50": o_p50, "O_p90": o_p90}


# --------------------------
# 5) Glue
# --------------------------
class Q2OPredictor:
    def __init__(self, history_csv: Optional[str] = None, prefer_hf_model: Optional[str] = None,
                 use_chat_template: bool = False, respect_len_hint: bool = True):
        self.token_counter = TokenCounter(prefer_hf_model=prefer_hf_model, use_chat_template=use_chat_template)
        self.task_detector = TaskTypeDetector()
        self.length_predictor = LengthPredictor(csv_path=history_csv)
        self.len_parser = LengthHintParser()
        self.respect_len_hint = respect_len_hint
        logger.info(f"Q2O 初始化完成：tokenizer={self.token_counter.model_name or 'fallback'}, "
                    f"use_chat_template={use_chat_template}")

    @lru_cache(maxsize=1000)
    def predict_single(self, question: str) -> PredictionResult:
        S = self.token_counter.count(question)
        task = self.task_detector.detect(question)
        pred = self.length_predictor.predict(S, task)

        used_len_hint, len_hint_value = False, None
        confidence = 0.6
        if self.respect_len_hint:
            hint = self.len_parser.parse_len_hint(question)
            if hint:
                token_est, unit = hint
                len_hint_value = token_est
                pred["O_p50"] = float(token_est)
                pred["O_p90"] = float(int(round(token_est * 1.2)))
                pred["O_mean"] = 0.5 * (pred["O_p50"] + pred["O_p90"])
                used_len_hint = True
                confidence = 0.8
            elif self.len_parser.has_length_constraint(question):
                confidence = 0.7

        return PredictionResult(
            S=S,
            task=task,
            tokenizer=self.token_counter.model_name or ("tiktoken/cl100k_base" if self.token_counter.enc else "approx"),
            O_mean=pred["O_mean"],
            O_p50=pred["O_p50"],
            O_p90=pred["O_p90"],
            confidence=confidence,
            used_len_hint=used_len_hint,
            len_hint_value=len_hint_value
        )

    def predict_batch(self, questions: List[str]) -> List[PredictionResult]:
        return [self.predict_single(q) for q in questions]

    def clear_cache(self):
        self.predict_single.cache_clear()


def estimate_from_question(question: str, history_csv: Optional[str] = None,
                           prefer_hf_model: Optional[str] = None, use_chat_template: bool = False,
                           respect_len_hint: bool = True) -> Dict[str, object]:
    predictor = Q2OPredictor(history_csv=history_csv, prefer_hf_model=prefer_hf_model,
                             use_chat_template=use_chat_template, respect_len_hint=respect_len_hint)
    return predictor.predict_single(question).__dict__


def main():
    ap = argparse.ArgumentParser(description="输入->S/O估算器（CJK-safe 任务识别 + 本地分词器优先）")
    ap.add_argument("--q", type=str, help="输入的问题文本（单条模式）")
    ap.add_argument("--hist", type=str, default="", help="可选：历史CSV(列: S,O,task)")
    ap.add_argument("--tok", type=str, default="", help="分词器本地目录/文件 或 远程repo名（推荐本地目录）")
    ap.add_argument("--with_chat_template", action="store_true", help="用chat模板计数（仅统计输入侧）")
    ap.add_argument("--no_len_hint", action="store_true", help="关闭长度提示解析")
    ap.add_argument("--batch", action="store_true", help="批量模式：从 --input_file 读取每行一个问题")
    ap.add_argument("--input_file", type=str, help="批量输入文件路径")
    ap.add_argument("--output_file", type=str, help="批量输出JSON路径")
    ap.add_argument("--show_json", action="store_true", help="JSON输出")
    ap.add_argument("--verbose", action="store_true", help="详细日志")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.batch and args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                qs = [ln.strip() for ln in f if ln.strip()]
            predictor = Q2OPredictor(history_csv=(args.hist or None),
                                     prefer_hf_model=(args.tok or None),
                                     use_chat_template=args.with_chat_template,
                                     respect_len_hint=(not args.no_len_hint))
            results = predictor.predict_batch(qs)
            out_items = [{"question": q, **r.__dict__} for q, r in zip(qs, results)]
            if args.output_file:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(out_items, f, ensure_ascii=False, indent=2)
                print(f"批量预测完成：保存到 {args.output_file}")
            else:
                for it in out_items:
                    print(json.dumps(it, ensure_ascii=False))
        except Exception as e:
            logger.error(f"批量失败：{e}")
        return

    if not args.q:
        ap.error("单条模式必须提供 --q，或使用 --batch --input_file")

    out = estimate_from_question(question=args.q,
                                 history_csv=(args.hist or None),
                                 prefer_hf_model=(args.tok or None),
                                 use_chat_template=args.with_chat_template,
                                 respect_len_hint=(not args.no_len_hint))
    if args.show_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"Tokenizer     : {out['tokenizer']}")
        print(f"S (tokens)    : {out['S']}")
        print(f"task type     : {out['task']}")
        print(f"O_mean        : {out['O_mean']:.1f}")
        print(f"O_p50         : {out['O_p50']:.1f}")
        print(f"O_p90         : {out['O_p90']:.1f}")
        print(f"Confidence    : {out['confidence']:.2f}")
        if out['used_len_hint']:
            print(f"Length hint   : {out['len_hint_value']} tokens")
        print(f"Used len hint : {out['used_len_hint']}")


if __name__ == "__main__":
    main()
