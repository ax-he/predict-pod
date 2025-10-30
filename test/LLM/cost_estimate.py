# -*- coding: utf-8 -*-
"""
q2o_cost_estimator.py
同时对多个模型估算：
1) 输入 token S（各自 tokenizer 计数，离线本地加载）
2) 输出 token O（任务类型先验 + 可选长度提示覆盖，默认用 p50，可选 p90）
3) 推理时延（TTFT/TPOT/总时长）与显存占用（权重/KV/合计）

依赖（本地离线可用）：
- transformers（仅用于本地 tokenizer.json 的加载；若缺失则退化为字符长度近似计数）

示例见文件顶部注释。
"""

import os
import re
import math
import json
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("q2o_cost")

# ---------------- Optional deps ----------------
try:
    from transformers import PreTrainedTokenizerFast, AutoTokenizer, PretrainedConfig
    _HAS_HF = True
except Exception as e:
    _HAS_HF = False
    log.warning("transformers 不可用，仅能用字符近似计数: %s", e)

# ---------------- Dataclasses ----------------
@dataclass
class ModelBundle:
    alias: str
    tok_dir: str                   # 需包含 tokenizer.json（或可被 AutoTokenizer 离线加载）
    cfg_dir: Optional[str] = None  # 需包含 config.json（仅读结构参数，不读 generation_config.json）
    params_b: Optional[float] = None     # 参数规模（B），若 cfg 没写 param 数就用它
    weight_bytes: Optional[float] = None # 权重量化字节数：BF16/FP16=2, INT8=1, INT4=0.5

@dataclass
class ModelArch:
    name: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    num_params: float      # 绝对参数个数（非 B）；若未给，从 params_b 推

@dataclass
class HWProfile:
    name: str
    gpu_flops_fp16: float  # 有效 FLOPs/s（FP16/BF16）
    hbm_bw: float          # 显存带宽 Byte/s
    vram_total_gb: float   # 总显存，仅用于提示/对比

@dataclass
class RuntimeOpts:
    kv_bytes: float = 2.0                 # KV 精度字节（BF16/FP16=2；KV 量化 8/4bit -> 1/0.5）
    kv_fragment_overhead: float = 0.05    # PagedAttention 残余碎片率（0~0.1）
    extra_vram_overhead_gb: float = 1.0   # 框架/工作区预留

# ---------------- Length hint ----------------
class LengthHint:
    @staticmethod
    def parse(question: str) -> Optional[int]:
        if not question:
            return None
        q = question.lower()

        single = [
            (r"(\d+)\s*字", 1.0),
            (r"(\d+)\s*词", 1.0),
            (r"(\d+)\s*words?", 1.0),
            (r"(?:约|大概|左右)\s*(\d+)\s*字", 1.0),
            (r"(?:about|approximately|around)\s*(\d+)\s*words?", 1.0),
            # 简单估算：句/段/行/页
            (r"(\d+)\s*(句|sentences?)", 15.0),
            (r"(\d+)\s*(段|paragraphs?)", 100.0),
            (r"(\d+)\s*(行|lines?)", 20.0),
            (r"(\d+)\s*(页|pages?)", 500.0),
        ]
        for pat, coef in single:
            m = re.search(pat, q, re.I)
            if m:
                try:
                    # 英文分组可能在 group(1) 就是数字
                    num = int(m.group(1))
                    return int(num * coef)
                except Exception:
                    pass

        ranges = [
            (r"(\d+)\s*到\s*(\d+)\s*字", 1.0),
            (r"(\d+)\s*-\s*(\d+)\s*words?", 1.0),
        ]
        for pat, coef in ranges:
            m = re.search(pat, q, re.I)
            if m:
                try:
                    a, b = int(m.group(1)), int(m.group(2))
                    return int(((a + b) / 2) * coef)
                except Exception:
                    pass
        return None

# ---------------- Task detector (explain优先于math) ----------------
class TaskDetector:
    def detect(self, text: str) -> str:
        if not text:
            return "qa_short"
        t = text.lower().strip()

        # 明确解释类：why/how/原理/为什么/如何 等 —— 若出现强公式痕迹再让位
        if re.search(r"(why|how|explain|原理|机制|原因|为什么|为何|如何|解释|讲解)", t):
            if re.search(r"\d+\s*[\+\-\*/]\s*\d+|\d+\s*\^\s*\d+|[\=\≈~]\s*\d+", t):
                return "math"
            return "qa_explain"

        # code
        if re.search(r"(code|程序|编程|实现|函数|算法|python|java|cpp|c\+\+|javascript|js|def |class |import )", t):
            return "code"

        # math
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

        return "qa_short" if len(t.split()) <= 8 else "qa_explain"

# ---------------- Tokenizer counter ----------------
class TokenCounter:
    def __init__(self, tok_dir: str):
        self.tok_dir = tok_dir
        self.tokenizer = None
        self.name = tok_dir

        if not _HAS_HF:
            return

        # 优先用 tokenizer.json 直接构造，不会联网
        tj = os.path.join(tok_dir, "tokenizer.json")
        if os.path.isfile(tj):
            try:
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tj)
                self.name = tok_dir
                return
            except Exception as e:
                log.warning("通过 tokenizer.json 加载失败：%s", e)

        # 次选：AutoTokenizer 离线加载（目录下需有完整 tokenizer 资源）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=True, local_files_only=True)
            self.name = tok_dir
        except Exception as e:
            log.warning("AutoTokenizer 加载失败（将使用字符近似）: %s", e)
            self.tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 1
        text = text.strip()
        if not text:
            return 1
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception as e:
                log.warning("tokenizer.encode 失败，改用字符近似：%s", e)
        # 退化：中英混合估计 1 token ≈ 3.5 字符
        return max(1, int(math.ceil(len(text) / 3.5)))

# ---------------- Output length predictor ----------------
class LengthPredictor:
    def __init__(self):
        self.priors = {
            "translate": {"p50_a": 1.00, "p50_b": 0,   "p90_a": 1.20, "p90_b": 10},
            "summarize": {"p50_a": 0.30, "p50_b": 30,  "p90_a": 0.45, "p90_b": 50},
            "code":      {"p50_a": 0.65, "p50_b": 40,  "p90_a": 0.90, "p90_b": 80},
            "math":      {"p50_a": 0.55, "p50_b": 30,  "p90_a": 0.80, "p90_b": 60},
            "qa_explain":{"p50_a": 0.60, "p50_b": 40,  "p90_a": 0.90, "p90_b": 80},
            "qa_short":  {"p50_a": 0.35, "p50_b": 40,  "p90_a": 0.55, "p90_b": 80},
            "creative":  {"p50_a": 0.80, "p50_b": 100, "p90_a": 1.20, "p90_b": 200},
        }

    def predict(self, S: int, task: str, len_hint_tokens: Optional[int], use_p90: bool=False) -> Tuple[float,float,float,bool]:
        task = task if task in self.priors else "qa_short"
        pri = self.priors[task]
        p50 = pri["p50_a"] * S + pri["p50_b"]
        p90 = pri["p90_a"] * S + pri["p90_b"]
        mean = 0.5 * (p50 + p90)

        used_hint = False
        if isinstance(len_hint_tokens, int) and len_hint_tokens > 0:
            used_hint = True
            p50 = float(len_hint_tokens)
            p90 = float(int(round(len_hint_tokens * 1.2)))
            mean = 0.5 * (p50 + p90)

        # 选择用于“上限”的 O
        O_eff = p90 if use_p90 else p50
        return float(mean), float(p50), float(p90), bool(used_hint), float(O_eff)

# ---------------- Read model arch from config.json ----------------
def read_arch(cfg_dir: Optional[str], fallback_name: str, params_b: Optional[float]) -> ModelArch:
    cfg = {}
    if cfg_dir:
        path = os.path.join(cfg_dir, "config.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                log.warning("读取 config.json 失败，将用默认/CLI：%s", e)

    L = int(cfg.get("num_hidden_layers", 32))
    H = int(cfg.get("num_attention_heads", 32))
    D = int(cfg.get("hidden_size", 4096))
    G = int(cfg.get("num_key_value_heads", H))
    name = cfg.get("architectures", [fallback_name])[0] if isinstance(cfg.get("architectures"), list) else fallback_name

    # 参数个数：优先 cfg 里的 “num_params”/“n_parameters”，否则用 CLI 的 B 倍数
    n_params = cfg.get("num_params") or cfg.get("n_parameters")
    if n_params is None and params_b is not None:
        n_params = float(params_b) * 1e9
    if n_params is None:
        # 粗略估：c * L * D^2（c≈3~6），给一个保守 3.5
        n_params = 3.5 * L * (D ** 2)

    return ModelArch(
        name=name, num_hidden_layers=L, hidden_size=D,
        num_attention_heads=H, num_kv_heads=G, num_params=float(n_params)
    )

# ---------------- Cost formulas ----------------
def kv_bytes_per_token(arch: ModelArch, rt: RuntimeOpts) -> float:
    head_dim = arch.hidden_size // arch.num_attention_heads
    kv_heads = arch.num_kv_heads
    # 每 token 的 KV：2 (K,V) * L * (kv_heads * head_dim) * bytes，并考虑碎片率
    base = 2.0 * arch.num_hidden_layers * (kv_heads * head_dim) * rt.kv_bytes
    return base * (1.0 + rt.kv_fragment_overhead)

def vram_kv_gb(arch: ModelArch, rt: RuntimeOpts, B: int, S: int, O: int) -> float:
    per_tok = kv_bytes_per_token(arch, rt)
    total = per_tok * B * (S + O)
    return total / (1024 ** 3)

def vram_weights_gb(n_params: float, weight_bytes: float) -> float:
    return (n_params * weight_bytes) / (1024 ** 3)

def prefill_decode_times(n_params: float, B: int, S: int, O: int, flops: float, bw: float) -> Dict[str, float]:
    """
    采用“推理算术”的简化：
    - 预填充（prefill）近似 compute-bound：T_prefill = (2 * N * B * S) / FLOPs
    - 解码（decode）近似 memory-bound：TPOT = max(2*N*B/FLOPs, 2*N/bw)（取 compute/memory 较大）
    - 总时长 = TTFT + TPOT * O
    """
    # Prefill: compute-bound 近似
    t_comp_prefill = (2.0 * n_params * B * S) / max(1e-9, flops)
    t_mem_prefill  = (2.0 * n_params) / max(1e-9, bw)
    TTFT = max(t_comp_prefill, t_mem_prefill)  # 更保守一些

    # Decode: 逐 token，取 compute/memory 更大的那个
    t_comp_decode = (2.0 * n_params * B) / max(1e-9, flops)
    t_mem_decode  = (2.0 * n_params) / max(1e-9, bw)
    TPOT = max(t_comp_decode, t_mem_decode)

    total_model = TTFT + TPOT * O
    return {
        "TTFT_s": TTFT,
        "TPOT_s": TPOT,
        "Total_model_s": total_model,
        "tokens_per_s": (1.0 / TPOT) if TPOT > 0 else float("inf"),
        "t_compute_prefill_s": t_comp_prefill,
        "t_memory_prefill_s": t_mem_prefill,
        "t_compute_decode_s": t_comp_decode,
        "t_memory_decode_s": t_mem_decode,
    }

# ---------------- HW Library ----------------
HW_LIB: Dict[str, HWProfile] = {
    "RTX3060":  HWProfile(name="RTX 3060 12GB", gpu_flops_fp16=12.7e12,  hbm_bw=360e9,   vram_total_gb=12.0),
    "A10080GB": HWProfile(name="16 * A100 80GB",     gpu_flops_fp16=4.992e15,   hbm_bw=3.2624e13,  vram_total_gb=1280.0),
}

# ---------------- Bundle parsing ----------------
def parse_bundles(args) -> List[ModelBundle]:
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
            weight_bytes = float(parts[4]) if len(parts) >= 5 and parts[4] else None
            bundles.append(ModelBundle(alias=alias, tok_dir=tok_dir, cfg_dir=cfg_dir,
                                       params_b=params_b, weight_bytes=weight_bytes))
    else:
        raise SystemExit("请至少提供一个 --bundle")
    return bundles

def parse_hw_map(args) -> Dict[str, HWProfile]:
    """--hw 形如 alias:RTX3060，可多次"""
    m: Dict[str, HWProfile] = {}
    if args.hw:
        for item in args.hw:
            k, v = item.split(":")
            if v not in HW_LIB:
                raise SystemExit(f"未知硬件标识 {v}，可选：{list(HW_LIB.keys())}")
            m[k] = HW_LIB[v]
    return m

# ---------------- Main routine ----------------
def run_for_bundle(question: str, mb: ModelBundle, hw: HWProfile,
                   rt: RuntimeOpts, use_p90: bool=False) -> Dict[str, object]:
    # 1) S
    tc = TokenCounter(mb.tok_dir)
    S = tc.count(question)

    # 2) task + O 预测（含长度提示覆盖）
    task = TaskDetector().detect(question)
    hint = LengthHint.parse(question)
    lp = LengthPredictor()
    O_mean, O_p50, O_p90, used_hint, O_eff = lp.predict(S, task, hint, use_p90=use_p90)

    # 3) 读结构参数 + 权重量化字节
    arch = read_arch(mb.cfg_dir, mb.alias, mb.params_b)
    w_bytes = 2.0 if mb.weight_bytes is None else float(mb.weight_bytes)

    # 4) 显存估算
    kv_gb = vram_kv_gb(arch, rt, B=1, S=S, O=int(round(O_eff)))
    w_gb  = vram_weights_gb(arch.num_params, w_bytes)
    vram_total = kv_gb + w_gb + rt.extra_vram_overhead_gb

    # 5) 时延估算
    times = prefill_decode_times(arch.num_params, B=1, S=S, O=int(round(O_eff)),
                                 flops=hw.gpu_flops_fp16, bw=hw.hbm_bw)

    return {
        "alias": mb.alias,
        "model_arch": arch.name,
        "hardware": hw.name,
        "S_tokens": S,
        "task": task,
        "O_mean": round(O_mean, 1),
        "O_p50": round(O_p50, 1),
        "O_p90": round(O_p90, 1),
        "O_used": int(round(O_eff)),
        "used_len_hint": used_hint,
        "len_hint_value": hint if hint else None,
        "VRAM_weights_GB": round(w_gb, 3),
        "VRAM_KV_GB": round(kv_gb, 3),
        "VRAM_total_est_GB": round(vram_total, 3),
        "TTFT_ms": round(times["TTFT_s"] * 1000, 1),
        "TPOT_ms": round(times["TPOT_s"] * 1000, 2),
        "Total_model_s": round(times["Total_model_s"], 2),
        "Tokens_per_s": round(times["tokens_per_s"], 2),
        "_details": times,
    }

def main():
    ap = argparse.ArgumentParser(description="多模型 S/O + 时延/显存 估算器（离线）")
    ap.add_argument("--q", type=str, required=True, help="输入的问题文本")
    ap.add_argument("--bundle", action="append",
                    help="alias:/tok_dir[:/cfg_dir][:paramsB][:weight_bytes]（可重复）", required=True)
    ap.add_argument("--hw", action="append",
                    help="可选：alias:HWID（HWID=RTX3060|A10080GB），不写则默认按次序绑定")
    ap.add_argument("--use_p90", action="store_true", help="用 p90 作为输出 O（默认 p50）")
    ap.add_argument("--kv_bytes", type=float, default=2.0, help="KV 精度字节数（2=BF16/FP16, 1=INT8, 0.5=INT4）")
    ap.add_argument("--kv_frag", type=float, default=0.05, help="KV 碎片率（0~0.1）")
    ap.add_argument("--extra_vram_gb", type=float, default=1.0, help="额外显存预留（GB）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")

    args = ap.parse_args()
    t0 = time.perf_counter()

    bundles = parse_bundles(args)
    # 绑定硬件：若未指定 --hw，则第1个→3060，其余→A100
    hw_map = parse_hw_map(args)
    default_order = ["RTX3060", "A10080GB"]
    results = []

    for i, mb in enumerate(bundles):
        if mb.alias in hw_map:
            hw = hw_map[mb.alias]
        else:
            hwid = default_order[0] if i == 0 else default_order[1]
            hw = HW_LIB[hwid]
        rt = RuntimeOpts(kv_bytes=args.kv_bytes, kv_fragment_overhead=args.kv_frag,
                         extra_vram_overhead_gb=args.extra_vram_gb)
        res = run_for_bundle(args.q, mb, hw, rt, use_p90=args.use_p90)
        results.append(res)

    t1 = time.perf_counter()

    if args.json:
        print(json.dumps({
            "question": args.q,
            "results": results,
            "program_elapsed_s": round(t1 - t0, 3)
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\nQuestion: {args.q}")
        for r in results:
            print(f"\n=== [{r['alias']}] ===")
            print(f"Model/HW     : {r['model_arch']} @ {r['hardware']}")
            print(f"S (tokens)   : {r['S_tokens']}   | Task: {r['task']}")
            if r["used_len_hint"]:
                print(f"O (hint)     : {r['O_used']}  (p50={r['O_p50']}, p90={r['O_p90']})")
            else:
                print(f"O (used)     : {r['O_used']}  (p50={r['O_p50']}, p90={r['O_p90']})")
            print(f"TTFT / TPOT  : {r['TTFT_ms']} ms / {r['TPOT_ms']} ms")
            print(f"Total time   : {r['Total_model_s']} s   | ~{r['Tokens_per_s']} tok/s")
            print(f"VRAM(Weights): {r['VRAM_weights_GB']} GB   | KV: {r['VRAM_KV_GB']} GB")
            print(f"VRAM Total   : {r['VRAM_total_est_GB']} GB   (HW cap: ~{HW_LIB['RTX3060'].vram_total_gb if '3060' in r['hardware'] else HW_LIB['A10080GB'].vram_total_gb} GB)")
        print(f"\nProgram elapsed: {round(t1 - t0, 3)} s")

if __name__ == "__main__":
    main()
