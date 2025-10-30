# -*- coding: utf-8 -*-
"""
q2o_predictor_multi.py
目标：同一问题在多个模型上同时估算：
    - 输入 token 数 S（基于各自 tokenizer）
    - 任务类型（解释优先）
    - 输出 token 的 O_mean / O_p50 / O_p90（结合先验 + 可选生成上限剪裁）

本版改动要点：
1) 仅从 {cfg_dir}/config.json 读取可用的生成上限字段（max_new_tokens / max_length）。
2) 不再调用 GenerationConfig.from_pretrained()，避免默认 20 token 截断。
3) 若 config.json 未显式提供上述字段，则不做任何上限剪裁。
"""

import os
import re
import math
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("q2o")

# ---------- 可选依赖（只用于 tokenizer 计数；不加载模型权重） ----------
try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )
    _HAS_HF = True
except Exception as e:
    _HAS_HF = False
    log.warning("transformers 未安装或导入失败，将退化为字符近似计数: %s", e)


# ---------- 数据类 ----------
@dataclass
class ModelBundle:
    alias: str
    tok_dir: str                 # 包含 tokenizer.json / tokenizer_config.json 的目录
    cfg_dir: Optional[str] = None  # 包含 config.json 的目录（可与 tok_dir 相同或不同）
    params_b: Optional[float] = None  # 可选：参数规模（B），用于轻度调节啰嗦度


@dataclass
class Prediction:
    S: int
    task: str
    tokenizer: str
    O_mean: float
    O_p50: float
    O_p90: float
    confidence: float
    used_len_hint: bool
    len_hint_value: Optional[int]


# ---------- 1) 长度提示解析 ----------
class LengthHint:
    @staticmethod
    def parse(question: str) -> Optional[int]:
        if not question:
            return None
        q = question.lower()

        # 单值
        rules = [
            (r"(\d+)\s*字", 1.0),
            (r"(\d+)\s*词", 1.0),
            (r"(\d+)\s*words?", 1.0),
            (r"(?:^|[^a-z])(\\d+)\s*sentences?(?:$|[^a-z])", 15.0),
            (r"(\d+)\s*句", 15.0),
            (r"(?:^|[^a-z])(\d+)\s*paragraphs?(?:$|[^a-z])", 100.0),
            (r"(\d+)\s*段", 100.0),
            (r"(?:^|[^a-z])(\d+)\s*lines?(?:$|[^a-z])", 20.0),
            (r"(\d+)\s*行", 20.0),
            (r"(?:^|[^a-z])(\d+)\s*pages?(?:$|[^a-z])", 500.0),
            (r"(\d+)\s*页", 500.0),
            (r"(?:约|大概|左右)\s*(\d+)\s*字", 1.0),
            (r"(?:about|approximately|around)\s*(\d+)\s*words?", 1.0),
        ]
        for pat, coef in rules:
            m = re.search(pat, q, re.I)
            if m:
                try:
                    return int(int(m.group(1)) * coef)
                except Exception:
                    pass

        # 范围（取均值）
        range_rules = [
            (r"(\d+)\s*到\s*(\d+)\s*字", 1.0),
            (r"(\d+)\s*-\s*(\d+)\s*words?", 1.0),
        ]
        for pat, coef in range_rules:
            m = re.search(pat, q, re.I)
            if m:
                try:
                    a, b = int(m.group(1)), int(m.group(2))
                    return int(((a + b) / 2) * coef)
                except Exception:
                    pass

        return None

    @staticmethod
    def has_constraint(question: str) -> bool:
        if not question:
            return False
        keys = [
            '字','词','句','段','行','页',
            'word','sentence','paragraph','line','page',
            '约','大概','左右','about','approximately','around'
        ]
        q = question.lower()
        return any(k in q for k in keys)


# ---------- 2) 任务类型检测（解释优先于数学） ----------
class TaskDetector:
    def __init__(self):
        pass

    def detect(self, text: str) -> str:
        if not text:
            return "qa_short"
        t = text.lower().strip()

        # 明确解释类：遇到 why/how/原理/为什么/如何 等 => 直接判为解释
        if re.search(r"(why|how|explain|原理|机制|原因|为什么|为何|如何|解释|讲解)", t):
            # 若包含强数学公式痕迹再让位（如显著算式）
            if re.search(r"\d+\s*[\+\-\*/]\s*\d+|\d+\s*\^\s*\d+|[\=\≈~]\s*\d+", t):
                return "math"
            return "qa_explain"

        # code
        if re.search(r"(code|程序|编程|实现|函数|算法|python|java|cpp|c\+\+|javascript|js|def |class |import )", t):
            return "code"

        # math（表达式/积分/概率等）
        if re.search(r"(计算|数学|公式|方程|函数|导数|积分|概率|统计|几何|代数|solve|equation|derivative|integral|probability)", t):
            return "math"

        # translate
        if re.search(r"(翻译|译为|译成|英文|中文|日语|法语|德语|translate|translation)", t):
            return "translate"

        # summarize
        if re.search(r"(总结|概括|摘要|概要|简述|tl;dr|tldr|summarize|summary)", t):
            return "summarize"

        # creative
        if re.search(r"(写一篇|创作|故事|小说|诗歌|essay|article|story|poem|想象|假如|如果|假设|imagine)", t):
            return "creative"

        # 短问句兜底
        if len(t.split()) <= 8:
            return "qa_short"
        return "qa_explain"


# ---------- 3) tokenizer 计数 ----------
class TokenCounter:
    def __init__(self, tok_dir: str):
        self.tok_dir = tok_dir
        self.name = tok_dir
        self.tokenizer = None

        if not _HAS_HF:
            log.warning("transformers 不可用，退化为字符近似计数")
            return

        # 优先：目录里存在 tokenizer.json → 用 PreTrainedTokenizerFast(tokenizer_file=...)
        tok_json = os.path.join(tok_dir, "tokenizer.json")
        if os.path.isfile(tok_json):
            try:
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_json)
                self.name = tok_dir
                return
            except Exception as e:
                log.warning("通过 tokenizer.json 加载失败：%s", e)

        # 次选：AutoTokenizer.from_pretrained(tok_dir)（完全离线，读取本地文件）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=True)
            self.name = tok_dir
        except Exception as e:
            log.warning("AutoTokenizer 加载失败：%s；将退化为字符近似", e)
            self.tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 1
        text = text.strip()
        if not text:
            return 1

        if self.tokenizer is not None:
            try:
                # 不加 special tokens，贴近“纯内容”计数
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                return len(ids)
            except Exception as e:
                log.warning("tokenizer.encode 失败，改为字符近似：%s", e)

        # 退化：中英混合估算 1 token ≈ 3.5 chars
        return max(1, int(math.ceil(len(text) / 3.5)))


# ---------- 4) 仅读取 config.json 的上限 ----------
class GenConfigReader:
    """
    仅从 cfg_dir/config.json 读取生成相关上限（若存在）：
      - max_new_tokens：直接作为“新增 token 上限”
      - max_length：总长度上限（输入+输出），此处仅做保守剪裁（不估算输入 S 与其之和）

    若文件不存在或未包含上述字段：不做任何上限剪裁。
    注意：像 max_position_embeddings / max_seq_len 代表上下文窗口，不等价于生成上限，此处忽略。
    """
    def __init__(self, cfg_dir: Optional[str]):
        self.cfg_dir = cfg_dir
        self.max_new_tokens = None
        self.max_length = None

        if not cfg_dir:
            return

        cfg_path = os.path.join(cfg_dir, "config.json")
        if not os.path.isfile(cfg_path):
            return

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # 仅读取显式提供的生成上限
            if isinstance(cfg.get("max_new_tokens"), int):
                self.max_new_tokens = int(cfg["max_new_tokens"])
            if isinstance(cfg.get("max_length"), int):
                self.max_length = int(cfg["max_length"])
        except Exception as e:
            log.info("读取 config.json 失败（忽略上限）：%s", e)

    def apply_caps(self, o_val: float) -> float:
        cap = None
        if isinstance(self.max_new_tokens, int):
            cap = self.max_new_tokens
        elif isinstance(self.max_length, int):
            cap = self.max_length
        if cap:
            return float(max(1, min(int(round(o_val)), cap)))
        return float(max(1, int(round(o_val))))


# ---------- 5) 输出长度预测（含先验 & 可选模型规模微调） ----------
class LengthPredictor:
    def __init__(self, params_b: Optional[float] = None, gen_reader: Optional[GenConfigReader] = None):
        self.params_b = params_b
        self.gen_reader = gen_reader

        # 经验先验（按任务类型），可根据真实日志再标定
        self.priors = {
            "translate": {"p50_a": 1.00, "p50_b": 0,  "p90_a": 1.20, "p90_b": 10},
            "summarize": {"p50_a": 0.30, "p50_b": 30, "p90_a": 0.45, "p90_b": 50},
            "code":      {"p50_a": 0.65, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "math":      {"p50_a": 0.55, "p50_b": 30, "p90_a": 0.80, "p90_b": 60},
            "qa_explain":{"p50_a": 0.60, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "qa_short":  {"p50_a": 0.35, "p50_b": 40, "p90_a": 0.55, "p90_b": 80},
            "creative":  {"p50_a": 0.80, "p50_b": 100,"p90_a": 1.20, "p90_b": 200},
        }

    def _model_verbosity_factor(self) -> float:
        """根据模型规模做一个*很轻*的啰嗦度调整（可按需替换/关闭）"""
        if not self.params_b or self.params_b <= 0:
            return 1.0
        import math as _m
        f = 1.0 + 0.12 * _m.log10(max(1e-9, self.params_b / 8.0))
        return float(min(1.30, max(0.85, f)))

    def predict(self, S: int, task: str, len_hint_tokens: Optional[int]) -> Tuple[float,float,float,bool]:
        task = task if task in self.priors else "qa_short"
        pri = self.priors[task]

        o_p50 = pri["p50_a"] * S + pri["p50_b"]
        o_p90 = pri["p90_a"] * S + pri["p90_b"]
        o_mean = 0.5 * (o_p50 + o_p90)

        # 模型规模微调
        f = self._model_verbosity_factor()
        o_p50 *= f
        o_p90 *= f
        o_mean *= f

        used_hint = False
        # 长度提示优先覆盖（并给 20% 余量作为 p90）
        if isinstance(len_hint_tokens, int) and len_hint_tokens > 0:
            used_hint = True
            o_p50 = float(len_hint_tokens)
            o_p90 = float(int(round(len_hint_tokens * 1.2)))
            o_mean = 0.5 * (o_p50 + o_p90)

        # 仅基于 config.json 的上限剪裁（如果明确提供）
        if self.gen_reader:
            o_p50 = self.gen_reader.apply_caps(o_p50)
            o_p90 = self.gen_reader.apply_caps(o_p90)
            o_mean = self.gen_reader.apply_caps(o_mean)

        # 合理裁剪
        def clip(x): return float(np.clip(x, 1, 32768))
        return clip(o_mean), clip(o_p50), clip(o_p90), used_hint


# ---------- 6) 主执行 ----------
def run_for_bundle(question: str, mb: ModelBundle) -> Prediction:
    # S: 按各自 tokenizer 计数
    tc = TokenCounter(mb.tok_dir)
    S = tc.count(question)

    # task: 解释优先
    task = TaskDetector().detect(question)

    # 解析长度提示
    hint = LengthHint.parse(question)
    conf = 0.8 if hint else (0.7 if LengthHint.has_constraint(question) else 0.6)

    # 只读 config.json（若存在），不调用 GenerationConfig
    gen_reader = GenConfigReader(cfg_dir=mb.cfg_dir)

    # 输出长度预测
    lp = LengthPredictor(params_b=mb.params_b, gen_reader=gen_reader)
    O_mean, O_p50, O_p90, used_hint = lp.predict(S, task, hint)

    return Prediction(
        S=S,
        task=task,
        tokenizer=tc.name,
        O_mean=O_mean,
        O_p50=O_p50,
        O_p90=O_p90,
        confidence=conf,
        used_len_hint=used_hint,
        len_hint_value=hint
    )


def parse_bundles(args) -> List[ModelBundle]:
    """
    两种指定方式：
    A) 旧式单模型（兼容）：
       --tok /path/to/tokenizer  [--cfg /path/to/config] [--params_b 8]
    B) 多模型（推荐）：
       多次传入 --bundle，格式：
       alias:/path/to/tokenizer[:/path/to/config][:paramsB]
       例如：
       --bundle qwen8b:/home/haga/models/deepseek-r1-qwen-8b/tokenizer:/home/haga/models/deepseek-r1-qwen-8b/config:8 \
       --bundle chimera671b:/home/haga/models/deepseek-r1t-chimera-671b/tokenizer:/home/haga/models/deepseek-r1t-chimera-671b/config:671
    """
    bundles: List[ModelBundle] = []
    if args.bundle:
        for b in args.bundle:
            parts = b.split(":")
            if len(parts) < 2:
                raise ValueError(f"--bundle 格式错误：{b}")
            alias = parts[0]
            tok_dir = parts[1]
            cfg_dir = parts[2] if len(parts) >= 3 and parts[2] else None
            params_b = float(parts[3]) if len(parts) >= 4 and parts[3] else None
            bundles.append(ModelBundle(alias=alias, tok_dir=tok_dir, cfg_dir=cfg_dir, params_b=params_b))
        return bundles

    # 单模型兼容
    if args.tok:
        bundles.append(ModelBundle(
            alias=os.path.basename(os.path.abspath(args.tok)),
            tok_dir=args.tok,
            cfg_dir=args.cfg,
            params_b=args.params_b
        ))
    return bundles


def main():
    ap = argparse.ArgumentParser(description="多模型 S/O token 估算器（仅读 config.json 上限）")
    ap.add_argument("--q", type=str, required=True, help="输入的问题文本")

    # 单模型（兼容）
    ap.add_argument("--tok", type=str, help="单模型：tokenizer 目录")
    ap.add_argument("--cfg", type=str, help="单模型：config 目录（包含 config.json）")
    ap.add_argument("--params_b", type=float, help="单模型：参数规模（B），影响啰嗦度微调")

    # 多模型（推荐）
    ap.add_argument("--bundle", action="append",
                    help="多模型：alias:/tok_dir[:/cfg_dir][:paramsB]，可重复传入多次")

    ap.add_argument("--show_json", action="store_true", help="以 JSON 打印结果")

    args = ap.parse_args()
    bundles = parse_bundles(args)
    if not bundles:
        raise SystemExit("请通过 --tok 或 --bundle 指定至少一个模型。")

    results: Dict[str, Dict] = {}
    for mb in bundles:
        pred = run_for_bundle(args.q, mb)
        results[mb.alias] = {
            "tokenizer": pred.tokenizer,
            "S": pred.S,
            "task": pred.task,
            "O_mean": round(pred.O_mean, 1),
            "O_p50": round(pred.O_p50, 1),
            "O_p90": round(pred.O_p90, 1),
            "confidence": round(pred.confidence, 2),
            "used_len_hint": pred.used_len_hint,
            "len_hint_value": pred.len_hint_value
        }

    if args.show_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for alias, r in results.items():
            print(f"\n=== [{alias}] ===")
            print(f"Tokenizer   : {r['tokenizer']}")
            print(f"S (tokens)  : {r['S']}")
            print(f"task type   : {r['task']}")
            print(f"O_mean      : {r['O_mean']}")
            print(f"O_p50       : {r['O_p50']}")
            print(f"O_p90       : {r['O_p90']}")
            print(f"Confidence  : {r['confidence']}")
            if r["used_len_hint"]:
                print(f"Length hint : {r['len_hint_value']} tokens")


if __name__ == "__main__":
    main()
