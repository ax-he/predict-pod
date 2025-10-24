# llm_cost_with_tools.py
# 端侧“提交前”预测：支持 CoT/自一致、多轮 RAG/Agent 对 S/O/时延/显存的影响
# 输出：TTFT/TPOT/总时长、KV/权重/总显存、路由建议（local or edge）

from dataclasses import dataclass
from typing import Dict, Any, List

# -------------------------
# 配置结构
# -------------------------
@dataclass
class ModelConfig:
    name: str
    num_hidden_layers: int      # L
    hidden_size: int            # hidden_size
    num_attention_heads: int    # H
    num_kv_heads: int           # GQA/MQA 时使用；否则与 H 相同
    num_params: float           # N (参数个数)
    weight_bytes: float         # 权重精度字节数（FP16/BF16=2, INT8=1, INT4=0.5）

@dataclass
class RuntimeOpts:
    kv_bytes: float = 2.0                 # KV 精度字节（BF16/FP16=2; KV 量化 8/4bit -> 1/0.5）
    kv_fragment_overhead: float = 0.05    # PagedAttention 后的残余碎片率（0~0.1）
    extra_vram_overhead_gb: float = 1.0   # 框架/工作区等预留

@dataclass
class HWProfile:
    name: str
    gpu_flops_fp16: float        # 有效 FLOPs/s（FP16/BF16）
    hbm_bw: float                # 显存带宽 Bytes/s
    vram_total_gb: float         # 总显存（用于容量判断）

@dataclass
class RequestBase:
    batch: int       # B
    S: int           # 输入 token
    O: int           # 期望输出上限（未含思考/多路线放大前）

# 工具与推理“开关”
@dataclass
class ToolSwitches:
    # 深度思考 / 自一致
    enable_cot: bool = True
    reason_tokens_per_chain: int = 0   # R_reason（单链平均思考 token）
    self_consistency_k: int = 1        # k 条推理路径（=1 表示不开自一致）

    # RAG / Agent（ReAct）
    rag_rounds: int = 0                 # 多轮检索/工具调用次数 r
    rag_ctx_tokens_per_round: List[int] = None  # 每轮 Δctx_i
    rag_tools_time_per_round_s: List[float] = None  # 每轮工具外部时延 T_tools,i（秒）

    def __post_init__(self):
        if self.rag_ctx_tokens_per_round is None:
            self.rag_ctx_tokens_per_round = []
        if self.rag_tools_time_per_round_s is None:
            self.rag_tools_time_per_round_s = []

@dataclass
class SLA:
    max_total_seconds: float
    max_vram_gb: float

# -------------------------
# 公式实现
# -------------------------
def kv_bytes_per_token(cfg: ModelConfig, rt: RuntimeOpts) -> float:
    L = cfg.num_hidden_layers
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_heads = cfg.num_kv_heads if cfg.num_kv_heads else cfg.num_attention_heads
    base = 2.0 * L * (kv_heads * head_dim) * rt.kv_bytes  # 2: K & V
    return base * (1.0 + rt.kv_fragment_overhead)

def vram_weights_gb(cfg: ModelConfig) -> float:
    return (cfg.num_params * cfg.weight_bytes) / (1024**3)

def vram_kv_gb(cfg: ModelConfig, rt: RuntimeOpts, B: int, S_eff: int, O_eff: int) -> float:
    per_tok = kv_bytes_per_token(cfg, rt)
    total = per_tok * B * (S_eff + O_eff)
    return total / (1024**3)

def prefill_decode_times(N: float, B: int, S_eff: int, O_eff: int, flops: float, bw: float) -> Dict[str, float]:
    # TGI 推理算术：每阶段取 compute/memory 的较大者
    t_comp_prefill = (2.0 * N * B * S_eff) / flops
    t_mem_prefill  = (2.0 * N) / bw
    TTFT = max(t_comp_prefill, t_mem_prefill)

    t_comp_decode = (2.0 * N * B) / flops
    t_mem_decode  = (2.0 * N) / bw
    TPOT = max(t_comp_decode, t_mem_decode)

    total_model = TTFT + TPOT * O_eff
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

def apply_tool_effects(base: RequestBase, tools: ToolSwitches) -> Dict[str, Any]:
    """根据 CoT/自一致 和 RAG/Agent 修正 S/O，并计算外部工具时延"""
    B, S, O = base.batch, base.S, base.O

    # ---- RAG/Agent：增加上下文，叠加外部工具时延 ----
    S_prime = S + sum(tools.rag_ctx_tokens_per_round[:tools.rag_rounds]) if tools.rag_rounds > 0 else S
    T_tools = sum(tools.rag_tools_time_per_round_s[:tools.rag_rounds]) if tools.rag_rounds > 0 else 0.0

    # ---- CoT/自一致：放大输出 token ----
    if tools.enable_cot:
        per_chain = O + max(0, tools.reason_tokens_per_chain)
    else:
        per_chain = O
    k = max(1, tools.self_consistency_k)
    O_prime = k * per_chain

    return {"B": B, "S_eff": S_prime, "O_eff": O_prime, "T_tools": T_tools}

def estimate_costs(cfg: ModelConfig, hw: HWProfile, base: RequestBase, rt: RuntimeOpts, tools: ToolSwitches) -> Dict[str, Any]:
    adj = apply_tool_effects(base, tools)
    B, S_eff, O_eff, T_tools = adj["B"], adj["S_eff"], adj["O_eff"], adj["T_tools"]

    kv_gb = vram_kv_gb(cfg, rt, B, S_eff, O_eff)
    w_gb  = vram_weights_gb(cfg)
    vram_total = kv_gb + w_gb + rt.extra_vram_overhead_gb

    times = prefill_decode_times(cfg.num_params, B, S_eff, O_eff, hw.gpu_flops_fp16, hw.hbm_bw)
    total_e2e = T_tools + times["Total_model_s"]   # 工具时延在线上链路之外累加

    return {
        "model": cfg.name,
        "hardware": hw.name,
        "B": B, "S_eff": S_eff, "O_eff": O_eff,
        "T_tools_s": round(T_tools, 3),
        "VRAM_weights_GB": round(w_gb, 3),
        "VRAM_KV_GB": round(kv_gb, 3),
        "VRAM_total_est_GB": round(vram_total, 3),
        "TTFT_ms": round(times["TTFT_s"] * 1000, 1),
        "TPOT_ms": round(times["TPOT_s"] * 1000, 2),
        "Total_model_s": round(times["Total_model_s"], 2),
        "Total_e2e_s": round(total_e2e, 2),
        "Tokens_per_s": round(times["tokens_per_s"], 2),
        "details": times,
    }

def decide_route(cost_local: Dict[str, Any],
                 cost_edge: Dict[str, Any],
                 sla: SLA) -> str:
    local_vram_ok = cost_local["VRAM_total_est_GB"] <= sla.max_vram_gb
    local_time_ok = cost_local["Total_e2e_s"] <= sla.max_total_seconds
    if not local_vram_ok or not local_time_ok:
        return "edge"
    # 二选一（可扩展：能耗/费用权重、敏感数据策略等）
    return "local" if cost_local["Total_e2e_s"] <= cost_edge["Total_e2e_s"] else "edge"

# -------------------------
# 示例：本地 13B（INT4 权重） vs 边缘 32B（BF16 权重）
# -------------------------
if __name__ == "__main__":
    # === 模型 ===
    local_cfg = ModelConfig(
        name="DeepSeek R1 13B (weight-only INT4)",
        num_hidden_layers=48, hidden_size=5120,
        num_attention_heads=40, num_kv_heads=8,
        num_params=13e9, weight_bytes=0.5,
    )
    edge_cfg = ModelConfig(
        name="DeepSeek R1 32B (BF16)",
        num_hidden_layers=64, hidden_size=5120,
        num_attention_heads=40, num_kv_heads=8,
        num_params=32e9, weight_bytes=2.0,
    )

    # === 运行选项 ===
    rt = RuntimeOpts(kv_bytes=2.0, kv_fragment_overhead=0.05, extra_vram_overhead_gb=1.0)

    # === 硬件 ===
    local_hw = HWProfile(name="RTX 3060 12GB", gpu_flops_fp16=12.7e12, hbm_bw=360e9, vram_total_gb=12.0)
    edge_hw  = HWProfile(name="A100 80GB",    gpu_flops_fp16=312e12,  hbm_bw=2.0e12, vram_total_gb=80.0)

    # === 请求 ===
    base = RequestBase(batch=1, S=200, O=200)

    # === 工具开关（修改这里即可测试不同策略） ===
    tools = ToolSwitches(
        enable_cot=True, reason_tokens_per_chain=120, self_consistency_k=1,    # 单链 CoT；k>1 表示自一致
        rag_rounds=1,
        rag_ctx_tokens_per_round=[1500],        # 一轮检索拼接 1500 token
        rag_tools_time_per_round_s=[0.9],       # 工具外部时延 0.9 秒
    )

    # === SLA ===
    sla = SLA(max_total_seconds=4.0, max_vram_gb=10.0)  # 例如：端侧可用显存 10GB，上限 4 秒

    # === 估算 & 路由 ===
    local_cost = estimate_costs(local_cfg, local_hw, base, rt, tools)
    edge_cost  = estimate_costs(edge_cfg,  edge_hw,  base, rt, tools)
    route = decide_route(local_cost, edge_cost, sla)

    # === 打印结果 ===
    cols = [
        ("Route", route),
        ("Local_Total_e2e_s", local_cost["Total_e2e_s"]),
        ("Edge_Total_e2e_s",  edge_cost["Total_e2e_s"]),
        ("Local_VRAM_GB",     local_cost["VRAM_total_est_GB"]),
        ("Edge_VRAM_GB",      edge_cost["VRAM_total_est_GB"]),
        ("Local_TTFT_ms",     local_cost["TTFT_ms"]),
        ("Local_TPOT_ms",     local_cost["TPOT_ms"]),
        ("Edge_TTFT_ms",      edge_cost["TTFT_ms"]),
        ("Edge_TPOT_ms",      edge_cost["TPOT_ms"]),
    ]
    print("=== Decision ===")
    for k, v in cols:
        print(f"{k:>20}: {v}")
