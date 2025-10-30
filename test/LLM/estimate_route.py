# -*- coding: utf-8 -*-
"""
cost_estimate_routed.py
融合：S/O 估算 + 时延/显存估算 + 门控/路由（OpenRouter, OpenAI 兼容）
- 仅使用本地 tokenizer 与 config.json
- 8B 计算时延 < 671B 时，触发门控：能做→小模型回答；否则→回退大模型
"""

import os, re, math, json, argparse, logging, time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np

# -------- 日志 --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("cost_route")

# -------- 可选依赖（仅需要 transformers 的 tokenizer 功能）--------
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, PretrainedConfig
    _HAS_HF = True
except Exception as e:
    _HAS_HF = False
    log.warning("transformers 未安装/导入失败，将退化为字符近似计数: %s", e)

# -------- OpenRouter（OpenAI 兼容）--------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception as e:
    _HAS_OPENAI = False
    log.warning("openai SDK 未安装：无法实际调用 OpenRouter，仅能 dry-run。")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"  # OpenRouter 为 OpenAI 兼容 API。参见官方文档与示例。 


# =========================
#  数据类
# =========================
@dataclass
class ModelBundle:
    alias: str
    tok_dir: str                 # 包含 tokenizer.json / tokenizer_config.json
    cfg_dir: Optional[str]       # 包含 config.json
    params_b: float              # 模型参数规模（B）
    weight_bytes: float          # 权重字节数（BF16/FP16=2, INT8=1, INT4=0.5）

@dataclass
class HWProfile:
    name: str
    gpu_flops_fp16: float   # 有效 FLOPs/s
    hbm_bw: float           # 显存带宽 Bytes/s
    vram_total_gb: float    # 总显存 GB

@dataclass
class ModelArch:
    name: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    num_params: float
    weight_bytes: float

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

@dataclass
class CostReport:
    TTFT_s: float
    TPOT_s: float
    Total_model_s: float
    tokens_per_s: float
    vram_weights_gb: float
    vram_kv_gb: float
    vram_total_gb: float
    prefill_comp_s: float
    prefill_mem_s: float
    decode_comp_s: float
    decode_mem_s: float
    bottleneck_prefill: str
    bottleneck_decode: str


# =========================
#  1) 长度提示与任务类型
# =========================
class LengthHint:
    @staticmethod
    def parse(question: str) -> Optional[int]:
        if not question: return None
        q = question.lower()
        # 单值
        rules = [
            (r"(\d+)\s*字", 1.0),
            (r"(\d+)\s*词", 1.0),
            (r"(\d+)\s*words?", 1.0),
            (r"(\d+)\s*(句|sentences?)", 15.0),
            (r"(\d+)\s*(段|paragraphs?)", 100.0),
            (r"(\d+)\s*(行|lines?)", 20.0),
            (r"(\d+)\s*(页|pages?)", 500.0),
            (r"(约|大概|左右)\s*(\d+)\s*字", 1.0),
            (r"(about|approximately|around)\s*(\d+)\s*words?", 1.0),
        ]
        for pat, coef in rules:
            m = re.search(pat, q, re.I)
            if m:
                try:
                    val = int(m.group(1) if m.lastindex == 1 else m.group(2))
                    return int(val * coef)
                except: pass
        # 范围
        for pat, coef in [(r"(\d+)\s*到\s*(\d+)\s*字",1.0),(r"(\d+)\s*-\s*(\d+)\s*words?",1.0)]:
            m = re.search(pat, q, re.I)
            if m:
                try:
                    a, b = int(m.group(1)), int(m.group(2))
                    return int(((a+b)/2) * coef)
                except: pass
        return None

    @staticmethod
    def has_constraint(question: str) -> bool:
        if not question: return False
        keys = ['字','词','句','段','行','页','word','sentence','paragraph','line','page','约','大概','左右','about','approximately','around']
        q = question.lower()
        return any(k in q for k in keys)


class TaskDetector:
    """解释优先，避免把“为什么/如何 … 计算”错判为 math。"""
    def detect(self, text: str) -> str:
        if not text: return "qa_short"
        t = text.lower().strip()
        # 解释优先
        if re.search(r"(why|how|explain|原理|机制|原因|为什么|为何|如何|解释|讲解)", t):
            if re.search(r"\d+\s*[\+\-\*/]\s*\d+|\d+\s*\^\s*\d+|[\=\≈~]\s*\d+", t):
                return "math"
            return "qa_explain"
        # code
        if re.search(r"(code|程序|编程|实现|函数|算法|python|java|cpp|c\+\+|javascript|js|def |class |import )", t):
            return "code"
        # math（关键词）
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
        # 兜底
        return "qa_short" if len(t.split()) <= 8 else "qa_explain"


# =========================
#  2) tokenizer 计数
# =========================
class TokenCounter:
    def __init__(self, tok_dir: str):
        self.tok_dir = tok_dir
        self.name = tok_dir
        self.tokenizer = None
        if not _HAS_HF:
            return
        tok_json = os.path.join(tok_dir, "tokenizer.json")
        if os.path.isfile(tok_json):
            try:
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_json)
                self.name = tok_dir
                return
            except Exception as e:
                log.warning("通过 tokenizer.json 加载失败：%s", e)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=True)
            self.name = tok_dir
        except Exception as e:
            log.warning("AutoTokenizer 加载失败：%s；使用字符近似", e)
            self.tokenizer = None

    def count(self, text: str) -> int:
        if not text: return 1
        text = text.strip()
        if not text: return 1
        if self.tokenizer is not None:
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                return len(ids)
            except Exception as e:
                log.warning("tokenizer.encode 失败，转字符近似：%s", e)
        return max(1, int(math.ceil(len(text) / 3.5)))  # 中英混合近似 3.5 chars/token


# =========================
#  3) 从 config.json 读结构
# =========================
def load_arch(cfg_dir: str, params_b: float, weight_bytes: float, alias: str) -> ModelArch:
    """只读 config.json，不推断 generation_config.json"""
    name = alias
    nh, hs, h, g = 32, 4096, 32, 8
    try:
        if _HAS_HF:
            conf = PretrainedConfig.from_pretrained(cfg_dir)
            name = getattr(conf, "architectures", [alias])[0]
            nh = int(getattr(conf, "num_hidden_layers", nh))
            hs = int(getattr(conf, "hidden_size", hs))
            h  = int(getattr(conf, "num_attention_heads", h))
            g  = int(getattr(conf, "num_key_value_heads", getattr(conf, "num_attention_heads", h)))
    except Exception as e:
        log.info("读取 config.json 失败，使用缺省: %s", e)
    return ModelArch(
        name=name,
        num_hidden_layers=nh,
        hidden_size=hs,
        num_attention_heads=h,
        num_kv_heads=g,
        num_params=float(params_b)*1e9,
        weight_bytes=weight_bytes
    )


# =========================
#  4) 输出长度预测（先验 + 长度提示）
# =========================
class LengthPredictor:
    PRIORS = {
        "translate": {"p50_a": 1.00, "p50_b": 0,  "p90_a": 1.20, "p90_b": 10},
        "summarize": {"p50_a": 0.30, "p50_b": 30, "p90_a": 0.45, "p90_b": 50},
        "code":      {"p50_a": 0.65, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
        "math":      {"p50_a": 0.55, "p50_b": 30, "p90_a": 0.80, "p90_b": 60},
        "qa_explain":{"p50_a": 0.60, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
        "qa_short":  {"p50_a": 0.35, "p50_b": 40, "p90_a": 0.55, "p90_b": 80},
        "creative":  {"p50_a": 0.80, "p50_b": 100,"p90_a": 1.20, "p90_b": 200},
    }

    def __init__(self, params_b: float):
        self.params_b = params_b

    def _verbosity_factor(self) -> float:
        if not self.params_b or self.params_b <= 0:
            return 1.0
        f = 1.0 + 0.12 * math.log10(max(1e-9, self.params_b/8.0))
        return float(min(1.30, max(0.85, f)))

    def predict(self, S: int, task: str, len_hint: Optional[int]) -> Tuple[float,float,float,bool]:
        pri = self.PRIORS.get(task, self.PRIORS["qa_short"])
        o_p50 = pri["p50_a"] * S + pri["p50_b"]
        o_p90 = pri["p90_a"] * S + pri["p90_b"]
        o_mean = 0.5*(o_p50+o_p90)
        # 规模微调
        f = self._verbosity_factor()
        o_p50, o_p90, o_mean = o_p50*f, o_p90*f, o_mean*f
        used = False
        if isinstance(len_hint, int) and len_hint > 0:
            used = True
            o_p50 = float(len_hint)
            o_p90 = float(int(round(len_hint*1.2)))
            o_mean = 0.5*(o_p50+o_p90)
        # 裁剪
        def clip(x): return float(np.clip(x, 1, 32768))
        return clip(o_mean), clip(o_p50), clip(o_p90), used


# =========================
#  5) 显存与时延估算
# =========================
def kv_bytes_per_token(arch: ModelArch, kv_bytes: float, frag_over: float) -> float:
    L = arch.num_hidden_layers
    head_dim = arch.hidden_size // arch.num_attention_heads
    kv_heads = arch.num_kv_heads if arch.num_kv_heads else arch.num_attention_heads
    base = 2.0 * L * (kv_heads * head_dim) * kv_bytes  # 2: K & V
    return base * (1.0 + frag_over)

def vram_weights_gb(arch: ModelArch) -> float:
    return (arch.num_params * arch.weight_bytes) / (1024**3)

def vram_kv_gb(arch: ModelArch, B: int, S_eff: int, O_eff: int, kv_bytes: float, frag_over: float) -> float:
    per_tok = kv_bytes_per_token(arch, kv_bytes, frag_over)
    total = per_tok * B * (S_eff + O_eff)
    return total / (1024**3)

def prefill_decode_times(N: float, B: int, S_eff: int, O_eff: int, flops: float, bw: float) -> Tuple[Dict[str,float], str, str]:
    # compute / memory 两路，取较大者
    t_comp_prefill = (2.0 * N * B * S_eff) / flops
    t_mem_prefill  = (2.0 * N) / bw
    TTFT = max(t_comp_prefill, t_mem_prefill)

    t_comp_decode = (2.0 * N * B) / flops
    t_mem_decode  = (2.0 * N) / bw
    TPOT = max(t_comp_decode, t_mem_decode)

    total_model = TTFT + TPOT*O_eff
    bott_prefill = "compute" if t_comp_prefill >= t_mem_prefill else "memory"
    bott_decode  = "compute" if t_comp_decode  >= t_mem_decode  else "memory"
    return {
        "TTFT_s":TTFT,"TPOT_s":TPOT,"Total_model_s":total_model,
        "tokens_per_s": (1.0/TPOT) if TPOT>0 else float("inf"),
        "t_compute_prefill_s":t_comp_prefill,"t_memory_prefill_s":t_mem_prefill,
        "t_compute_decode_s":t_comp_decode,"t_memory_decode_s":t_mem_decode
    }, bott_prefill, bott_decode

def estimate_cost(arch: ModelArch, hw: HWProfile, B: int, S: int, O: int,
                  kv_bytes: float=2.0, kv_frag_over: float=0.05, extra_overhead_gb: float=1.0) -> CostReport:
    kv_gb = vram_kv_gb(arch, B, S, O, kv_bytes, kv_frag_over)
    w_gb  = vram_weights_gb(arch)
    vram_total = kv_gb + w_gb + extra_overhead_gb
    times, bott_p, bott_d = prefill_decode_times(arch.num_params, B, S, O, hw.gpu_flops_fp16, hw.hbm_bw)
    return CostReport(
        TTFT_s=times["TTFT_s"], TPOT_s=times["TPOT_s"], Total_model_s=times["Total_model_s"],
        tokens_per_s=times["tokens_per_s"], vram_weights_gb=w_gb, vram_kv_gb=kv_gb,
        vram_total_gb=vram_total,
        prefill_comp_s=times["t_compute_prefill_s"], prefill_mem_s=times["t_memory_prefill_s"],
        decode_comp_s=times["t_compute_decode_s"], decode_mem_s=times["t_memory_decode_s"],
        bottleneck_prefill=bott_p, bottleneck_decode=bott_d
    )


# =========================
#  6) 门控与回答（OpenRouter）
# =========================
TASK_ENUM = ["arithmetic","format_conversion","programming","math_proof","legal_citation","time_sensitive","open_domain"]

def build_or_client(api_key: Optional[str]):
    if not _HAS_OPENAI or not api_key:
        return None
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

def gate_plan(client, model_small: str, question: str):
    """返回 (plan_dict, valid_bool)。valid=False 代表 gate 失败（不当作低分）。"""
    if client is None:
        return None, False
    sys_prompt = (
        "你没有联网/外部资料访问能力。"
        "只返回一个 JSON 对象，字段："
        f'{{"task_type":"{"|".join(TASK_ENUM)}",'
        '"needs_external_docs":true/false,'
        '"needs_verbatim_citations":true/false,'
        '"self_score":0..1}}。'
        "判定规则："
        "- 纯算术/进制转换/格式化 → arithmetic/format_conversion；通常不需要外部资料。\n"
        "- 编程 → programming；若依赖私有接口/资料，自评≤0.3，needs_external_docs=true。\n"
        "- 数学证明 → math_proof；若要求逐字引用文献页码，则 needs_external_docs=true。\n"
        "- 逐字引用法律/标准并给出条号/页码/编号 → legal_citation；若无法核对原文，自评≤0.3、needs_external_docs=true。\n"
        "- 依赖最新事实(新闻/价格/赛果/版本发布等) → time_sensitive；通常需要外部资料。\n"
        "- 其余 → open_domain。严禁输出任何非JSON字符。"
    )
    resp = client.chat.completions.create(
        model=model_small,
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":question}],
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
                }, "strict":True
            }
        },
        temperature=0, max_tokens=160,
        extra_body={"usage":{"include":True}}
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(raw) if raw else {}
    except Exception:
        obj = {}
    valid = bool(raw and obj and all(k in obj for k in ("task_type","needs_external_docs","needs_verbatim_citations","self_score")))
    return obj if valid else None, valid

def answer(client, model: str, question: str, role_hint: Optional[str]=None) -> Tuple[str, Dict[str,int], float]:
    """返回 (text, usage_dict, latency_ms)"""
    if client is None:
        return "[DRY RUN] 未配置 OPENROUTER_API_KEY，跳过实际回答。", {"prompt":0,"completion":0,"total":0}, 0.0
    t0 = time.perf_counter()
    sys_prompt = role_hint or "你是中文问答助手，回答结构清晰、分点精炼；若无法满足前置约束，请仅输出“置信度不足”。"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":question}],
        temperature=0.2, max_tokens=1200,
        extra_body={"usage":{"include":True}, "reasoning":{"exclude":True}}
    )
    ms = (time.perf_counter()-t0)*1000.0
    ch = resp.choices[0]
    text = (ch.message.content or "").strip() or "[EMPTY]"
    u = getattr(resp, "usage", None)
    usage = {"prompt":getattr(u,"prompt_tokens",0),"completion":getattr(u,"completion_tokens",0),"total":getattr(u,"total_tokens",0)}
    return text, usage, ms


# =========================
#  7) CLI & 主流程
# =========================
def parse_bundles(args) -> List[ModelBundle]:
    bundles = []
    if not args.bundle or len(args.bundle)==0:
        raise SystemExit("请用 --bundle 至少提供两个模型（小/大）。")
    for b in args.bundle:
        # alias:/tok_dir:/cfg_dir:paramsB:weight_bytes
        parts = b.split(":")
        if len(parts) < 5:
            raise SystemExit(f"--bundle 格式错误：{b}")
        alias, tok, cfg, pb, wbytes = parts[0], parts[1], parts[2], float(parts[3]), float(parts[4])
        bundles.append(ModelBundle(alias=alias, tok_dir=tok, cfg_dir=cfg, params_b=pb, weight_bytes=wbytes))
    if len(bundles) < 2:
        log.warning("只提供了 1 个 bundle，将无法比较大小模型。")
    return bundles

def main():
    ap = argparse.ArgumentParser(description="S/O 估算 + 成本评估 + 门控/路由（OpenRouter）")
    ap.add_argument("--q", type=str, required=True, help="问题文本")
    ap.add_argument("--bundle", action="append",
                    help="alias:/tok_dir:/cfg_dir:paramsB:weight_bytes （可传两次：小/大）")
    ap.add_argument("--model_small", type=str, default="deepseek/deepseek-r1-0528-qwen3-8b:free")
    ap.add_argument("--model_large", type=str, default="tngtech/deepseek-r1t-chimera:free")
    ap.add_argument("--small_name", type=str, default="RTX 3060 12GB")
    ap.add_argument("--large_name", type=str, default="A100 80GB")
    # 默认硬件（可改）
    ap.add_argument("--small_flops", type=float, default=12.7e12)
    ap.add_argument("--small_bw", type=float, default=360e9)
    ap.add_argument("--small_vram", type=float, default=12.0)
    ap.add_argument("--large_flops", type=float, default=312e12)
    ap.add_argument("--large_bw", type=float, default=2.039e12)  # A100 80GB SXM 约 2.039 TB/s
    ap.add_argument("--large_vram", type=float, default=80.0)
    ap.add_argument("--large_ngpus", type=int, default=1, help="大模型 GPU 数（线性放大 FLOPs/带宽/显存）")
    # KV/框架开销
    ap.add_argument("--kv_bytes", type=float, default=2.0)
    ap.add_argument("--kv_frag_over", type=float, default=0.05)
    ap.add_argument("--extra_overhead_gb", type=float, default=1.0)
    # 其它
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true", help="只做估算与路由，不调用 OpenRouter")
    ap.add_argument("--show_json", action="store_true")
    args = ap.parse_args()

    t_prog = time.perf_counter()

    bundles = parse_bundles(args)
    # 取 2 个：按 params_b 小→大 排序
    bundles.sort(key=lambda x: x.params_b)
    small, large = bundles[0], bundles[-1]

    # 硬件对象（大模型按 ngpus 放大）
    small_hw = HWProfile(args.small_name, args.small_flops, args.small_bw, args.small_vram)
    scale = max(1, int(args.large_ngpus))
    large_hw = HWProfile(
        name=f"{scale} * {args.large_name}" if scale>1 else args.large_name,
        gpu_flops_fp16=args.large_flops * scale,
        hbm_bw=args.large_bw * scale,
        vram_total_gb=args.large_vram * scale,
    )

    # 加载 tokenizer & 计数
    tc_small = TokenCounter(small.tok_dir)
    tc_large = TokenCounter(large.tok_dir)
    S_small = tc_small.count(args.q)
    S_large = tc_large.count(args.q)  # 理论上两个 close，但严格按各自 tokenizer

    # 任务与长度提示
    task = TaskDetector().detect(args.q)
    hint = LengthHint.parse(args.q)
    conf_base = 0.8 if hint else (0.7 if LengthHint.has_constraint(args.q) else 0.6)

    # 读取结构 & 预测 O
    arch_small = load_arch(small.cfg_dir, small.params_b, small.weight_bytes, small.alias)
    arch_large = load_arch(large.cfg_dir, large.params_b, large.weight_bytes, large.alias)

    lp_small = LengthPredictor(small.params_b)
    lp_large = LengthPredictor(large.params_b)
    O_mean_s, O_p50_s, O_p90_s, used_hint_s = lp_small.predict(S_small, task, hint)
    O_mean_l, O_p50_l, O_p90_l, used_hint_l = lp_large.predict(S_large, task, hint)

    # 成本估算
    cost_small = estimate_cost(
        arch_small, small_hw, args.batch, S_small, int(round(O_p50_s)),
        kv_bytes=args.kv_bytes, kv_frag_over=args.kv_frag_over, extra_overhead_gb=args.extra_overhead_gb
    )
    cost_large = estimate_cost(
        arch_large, large_hw, args.batch, S_large, int(round(O_p50_l)),
        kv_bytes=args.kv_bytes, kv_frag_over=args.kv_frag_over, extra_overhead_gb=args.extra_overhead_gb
    )

    # 打印两个模型估算
    def pretty(alias, arch, hw, S, O_p50, O_p90, O_mean, cost, tokenizer_name):
        print(f"\n=== [{alias}] ===")
        print(f"Model/HW     : {arch.name} @ {hw.name}")
        print(f"S (tokens)   : {S}   | Task: {task}")
        print(f"O (used)     : {int(round(O_p50))}  (p50={O_p50:.1f}, p90={O_p90:.1f})")
        print(f"TTFT / TPOT  : {cost.TTFT_s*1000:.1f} ms / {cost.TPOT_s*1000:.2f} ms")
        print(f"Total time   : {cost.Total_model_s:.2f} s   | ~{cost.tokens_per_s:.2f} tok/s")
        print(f"VRAM(Weights): {cost.vram_weights_gb:.3f} GB   | KV: {cost.vram_kv_gb:.3f} GB")
        print(f"VRAM Total   : {cost.vram_total_gb:.3f} GB   (HW cap: ~{hw.vram_total_gb:.1f} GB)")
        print(f"Bottleneck   : prefill={cost.bottleneck_prefill}, decode={cost.bottleneck_decode}")
        print(f"Tokenizer    : {tokenizer_name}")

    pretty(small.alias, arch_small, small_hw, S_small, O_p50_s, O_p90_s, O_mean_s, cost_small, tc_small.name)
    pretty(large.alias, arch_large, large_hw, S_large, O_p50_l, O_p90_l, O_mean_l, cost_large, tc_large.name)

    # ---------- 路由逻辑 ----------
    # 仅当小模型总时长更小，且显存可承受时，才触发门控；否则直接用大模型
    small_faster = cost_small.Total_model_s < cost_large.Total_model_s
    small_vram_ok = cost_small.vram_total_gb <= small_hw.vram_total_gb

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    dry = args.dry_run or (not api_key)
    if not api_key:
        log.info("未检测到 OPENROUTER_API_KEY → 自动进入 dry-run（不实际调用 API）")

    chosen_model = args.model_large
    route_msg = "large (no gate)"
    gate_ms = ans_ms = 0.0
    final_text = None
    usage = {"prompt":0,"completion":0,"total":0}

    if small_faster and small_vram_ok:
        # 触发门控
        client = build_or_client(api_key)
        t_gate = time.perf_counter()
        plan, valid = (None, False) if dry else gate_plan(client, args.model_small, args.q)
        gate_ms = (time.perf_counter()-t_gate)*1000.0 if not dry else 0.0

        allow_small = False
        if valid and plan:
            t  = plan["task_type"]
            ed = bool(plan["needs_external_docs"])
            vb = bool(plan["needs_verbatim_citations"])
            sc = float(plan["self_score"])
            # 与你给的硬规则一致
            if t == "legal_citation" or ed or vb:
                allow_small = False
            elif t in ("arithmetic","format_conversion","programming","math_proof"):
                allow_small = (sc >= 0.60)
            elif t == "open_domain":
                allow_small = (sc >= 0.50)
            else:
                allow_small = (sc >= 0.60)
        else:
            # gate 不可用 → 视作失败，回退大模型
            allow_small = False

        if allow_small:
            chosen_model = args.model_small
            route_msg = "small (gate pass)"
        else:
            chosen_model = args.model_large
            route_msg = "large (gate fail or needs external/verbatim)"

        # 最终回答（可 dry_run）
        client = build_or_client(api_key)
        t_ans = time.perf_counter()
        final_text, usage, ans_ms = ("[DRY RUN] 跳过回答。", {"prompt":0,"completion":0,"total":0}, 0.0) if dry \
            else answer(client, chosen_model, args.q)
        ans_ms = (time.perf_counter()-t_ans)*1000.0 if dry else ans_ms
    else:
        # 直接大模型（小模型不更快或显存不足）
        chosen_model = args.model_large
        route_msg = "large (small not faster or VRAM limited)"
        client = build_or_client(api_key)
        t_ans = time.perf_counter()
        final_text, usage, ans_ms = ("[DRY RUN] 跳过回答。", {"prompt":0,"completion":0,"total":0}, 0.0) if dry \
            else answer(client, chosen_model, args.q)
        ans_ms = (time.perf_counter()-t_ans)*1000.0 if dry else ans_ms

    # ---------- 汇总 ----------
    total_prog = (time.perf_counter()-t_prog)
    print("\n=== Routing Decision ===")
    print(f"Chosen model : {chosen_model}  ({route_msg})")
    print(f"Gate ms      : {gate_ms:.1f}")
    print(f"Answer ms    : {ans_ms:.1f}")
    print(f"Program elapsed: {total_prog:.3f} s")

    # 打印最终回答（如有）
    if final_text is not None:
        print("\n=== 最终回答 ===\n" + str(final_text))
        print(f"\n[usage] prompt={usage.get('prompt',0)} completion={usage.get('completion',0)} total={usage.get('total',0)}")

    # 可选 JSON 输出
    if args.show_json:
        out = {
            "small":{
                "alias":small.alias,"arch":arch_small.name,"hw":small_hw.name,
                "S":S_small,"task":task,
                "O_p50":O_p50_s,"O_p90":O_p90_s,"O_mean":O_mean_s,
                "cost":{
                    "TTFT_ms":round(cost_small.TTFT_s*1000,1),
                    "TPOT_ms":round(cost_small.TPOT_s*1000,2),
                    "Total_s":round(cost_small.Total_model_s,2),
                    "tokens_per_s":round(cost_small.tokens_per_s,2),
                    "VRAM_weights_GB":round(cost_small.vram_weights_gb,3),
                    "VRAM_KV_GB":round(cost_small.vram_kv_gb,3),
                    "VRAM_total_GB":round(cost_small.vram_total_gb,3),
                    "bottleneck":{"prefill":cost_small.bottleneck_prefill,"decode":cost_small.bottleneck_decode}
                }
            },
            "large":{
                "alias":large.alias,"arch":arch_large.name,"hw":large_hw.name,
                "S":S_large,"task":task,
                "O_p50":O_p50_l,"O_p90":O_p90_l,"O_mean":O_mean_l,
                "cost":{
                    "TTFT_ms":round(cost_large.TTFT_s*1000,1),
                    "TPOT_ms":round(cost_large.TPOT_s*1000,2),
                    "Total_s":round(cost_large.Total_model_s,2),
                    "tokens_per_s":round(cost_large.tokens_per_s,2),
                    "VRAM_weights_GB":round(cost_large.vram_weights_gb,3),
                    "VRAM_KV_GB":round(cost_large.vram_kv_gb,3),
                    "VRAM_total_GB":round(cost_large.vram_total_gb,3),
                    "bottleneck":{"prefill":cost_large.bottleneck_prefill,"decode":cost_large.bottleneck_decode}
                }
            },
            "routing":{"chosen_model":chosen_model,"route_msg":route_msg,"gate_ms":gate_ms,"answer_ms":ans_ms,"program_elapsed_s":round(total_prog,3)},
            "answer":{"text":final_text,"usage":usage}
        }
        print("\n=== JSON ===")
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
