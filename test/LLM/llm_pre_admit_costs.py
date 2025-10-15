# llm_pre_admit_costs.py
# 计算 LLM 推理在“启动前”的资源开销，随提问 token 数 S 变化给出显存/时间/吞吐与本地或边缘的路由建议
# Author: you

from dataclasses import dataclass
from typing import List, Dict, Any
import math

# -------------------------
# 数据结构
# -------------------------
@dataclass
class ModelConfig:
    name: str
    num_hidden_layers: int          # L
    hidden_size: int                # hidden_size
    num_attention_heads: int        # H
    num_kv_heads: int               # 如未使用 GQA，这里与 H 相同
    num_params: float               # N (参数个数), 例如 7e9
    weight_bytes: float             # 权重量化后的每元素字节数（FP16/BF16=2, INT8=1, INT4=0.5）

@dataclass
class RuntimeOpts:
    kv_bytes: float = 2.0           # KV 精度的每元素字节数（默认BF16/FP16=2；8bit=1；4bit=0.5）
    kv_fragment_overhead: float = 0.05  # PagedAttention 之后残余碎片/管理开销（0~0.1常见）
    extra_vram_overhead_gb: float = 1.0 # 运行时额外显存（框架、workspace等）保守预留

@dataclass
class HWProfile:
    name: str
    gpu_flops_fp16: float           # 有效 FLOPs/s（FP16/BF16），如 12.7e12 表示 12.7 TFLOPS
    hbm_bw: float                   # 显存带宽 Bytes/s，如 360e9 = 360 GB/s
    vram_total_gb: float            # 显存总量（用于容量判断）

@dataclass
class Request:
    batch: int                      # B
    input_tokens: int               # S
    output_tokens: int              # O

@dataclass
class SLA:
    max_total_seconds: float        # 允许的总时长上限
    max_vram_gb: float              # 本地允许占用的显存上限（可设为可用显存或某个阈值）

# -------------------------
# 核心计算
# -------------------------
def kv_bytes_per_token(cfg: ModelConfig, rt: RuntimeOpts) -> float:
    """ 每 token 的 KV 字节量（近似公式） """
    L = cfg.num_hidden_layers
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_heads = cfg.num_kv_heads if cfg.num_kv_heads else cfg.num_attention_heads
    base = 2.0 * L * (kv_heads * head_dim) * rt.kv_bytes  # 2: K 和 V
    return base * (1.0 + rt.kv_fragment_overhead)         # 碎片/管理系数

def vram_weights_gb(cfg: ModelConfig) -> float:
    return (cfg.num_params * cfg.weight_bytes) / (1024**3)

def vram_kv_gb(cfg: ModelConfig, rt: RuntimeOpts, req: Request) -> float:
    per_tok = kv_bytes_per_token(cfg, rt)                 # bytes/token
    total = per_tok * req.batch * (req.input_tokens + req.output_tokens)
    return total / (1024**3)

def prefill_decode_times(cfg: ModelConfig, hw: HWProfile, req: Request) -> Dict[str, float]:
    """
    依据 TGI 的推理算术，把 prefill / decode 分开：
    t_compute_prefill  ≈ (2*N*B*S)/FLOPs
    t_memory_prefill   ≈ (2*N)/HBM
    TTFT = max(两者)
    t_compute_decode   ≈ (2*N*B)/FLOPs
    t_memory_decode    ≈ (2*N)/HBM
    TPOT = max(两者)
    """
    N = cfg.num_params
    B, S, O = req.batch, req.input_tokens, req.output_tokens
    flops = hw.gpu_flops_fp16
    bw = hw.hbm_bw

    t_compute_prefill = (2.0 * N * B * S) / flops
    t_memory_prefill  = (2.0 * N) / bw
    ttft = max(t_compute_prefill, t_memory_prefill)

    t_compute_decode = (2.0 * N * B) / flops
    t_memory_decode  = (2.0 * N) / bw
    tpot = max(t_compute_decode, t_memory_decode)

    total = ttft + tpot * O
    return {
        "t_compute_prefill_s": t_compute_prefill,
        "t_memory_prefill_s": t_memory_prefill,
        "TTFT_s": ttft,
        "t_compute_decode_s": t_compute_decode,
        "t_memory_decode_s": t_memory_decode,
        "TPOT_s": tpot,
        "Total_s": total,
        "tokens_per_s": (1.0 / tpot) if tpot > 0 else float("inf"),
    }

def estimate_costs(cfg: ModelConfig, hw: HWProfile, req: Request, rt: RuntimeOpts) -> Dict[str, Any]:
    kv_gb = vram_kv_gb(cfg, rt, req)
    w_gb  = vram_weights_gb(cfg)
    vram_total = kv_gb + w_gb + rt.extra_vram_overhead_gb

    times = prefill_decode_times(cfg, hw, req)
    payload = {
        "model": cfg.name,
        "hardware": hw.name,
        "B": req.batch, "S": req.input_tokens, "O": req.output_tokens,
        "VRAM_weights_GB": round(w_gb, 3),
        "VRAM_KV_GB": round(kv_gb, 3),
        "VRAM_total_est_GB": round(vram_total, 3),
        "TTFT_ms": round(times["TTFT_s"] * 1000, 1),
        "TPOT_ms": round(times["TPOT_s"] * 1000, 2),
        "Total_s": round(times["Total_s"], 2),
        "Tokens_per_s": round(times["tokens_per_s"], 2),
        "details": times,
    }
    return payload

def decide_route(cost_local: Dict[str, Any],
                 cost_edge: Dict[str, Any],
                 sla: SLA) -> str:
    """
    简单决策：
    1) 若本地显存或总时长不达标 -> 路由边缘
    2) 否则比较总时长，谁更快选谁；也可叠加你的成本/能耗权重
    """
    local_vram_ok = cost_local["VRAM_total_est_GB"] <= sla.max_vram_gb
    local_time_ok = cost_local["Total_s"] <= sla.max_total_seconds

    if not local_vram_ok or not local_time_ok:
        return "edge"
    # 二选一（可拓展为 bandit/RL）
    return "local" if cost_local["Total_s"] <= cost_edge["Total_s"] else "edge"

# -------------------------
# 示例：DeepSeek-R1-Distill-Qwen-7B
# -------------------------
if __name__ == "__main__":
    # 模型：7B（Qwen系蒸馏），常见配置：L=28, hidden=3584, H=28, GQA n_kv_heads=4
    cfg = ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-7B (weight-only INT4)",
        num_hidden_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_kv_heads=4,            # GQA
        num_params=7e9,            # 7B
        weight_bytes=0.5,          # INT4 权重量化≈0.5 byte/param
    )

    # 运行时：KV 8/16/4bit 可选，PagedAttention 碎片系数 0~0.1
    rt = RuntimeOpts(
        kv_bytes=2.0,               # KV 用 BF16/FP16=2；若 KV 量化8bit=1，4bit=0.5
        kv_fragment_overhead=0.05,  # PagedAttention 后约5%开销
        extra_vram_overhead_gb=1.0, # 预留1GB框架/工作区
    )

    # 硬件：本地 vs 边缘（示例数值）
    local = HWProfile(
        name="RTX 3060 12GB",
        gpu_flops_fp16=12.7e12,     # 12.7 TFLOPS
        hbm_bw=360e9,               # 360 GB/s (GDDR 带宽)
        vram_total_gb=12.0
    )
    edge = HWProfile(
        name="A100 80GB",
        gpu_flops_fp16=312e12,      # 312 TFLOPS
        hbm_bw=2.0e12,              # 2 TB/s
        vram_total_gb=80.0
    )

    # SLA：例如家庭侧总时长 ≤ 4s，显存不超过“可用” 10GB（留有余量）
    sla = SLA(max_total_seconds=4.0, max_vram_gb=10.0)

    # 扫不同提问 token 数 S；输出表格
    batch = 1
    S_list = [50, 100, 200, 400, 800]   # 可自行修改
    O = 200                              # 期望生成上限

    print(f"{'S':>5} | {'VRAM_local(GB)':>13} | {'Total_local(s)':>13} | "
          f"{'VRAM_edge(GB)':>12} | {'Total_edge(s)':>12} | {'Route':>6}")
    print("-"*75)

    for S in S_list:
        req = Request(batch=batch, input_tokens=S, output_tokens=O)
        local_cost = estimate_costs(cfg, local, req, rt)
        edge_cost  = estimate_costs(cfg, edge,  req, rt)

        route = decide_route(local_cost, edge_cost, sla)
        print(f"{S:5d} | {local_cost['VRAM_total_est_GB']:13.2f} | "
              f"{local_cost['Total_s']:13.2f} | "
              f"{edge_cost['VRAM_total_est_GB']:12.2f} | "
              f"{edge_cost['Total_s']:12.2f} | {route:>6}")

    # 想看某个 S 的详细分解，可取消以下注释
    # debug_S = 200
    # req = Request(batch=1, input_tokens=debug_S, output_tokens=O)
    # detail = estimate_costs(cfg, local, req, rt)
    # from pprint import pprint; pprint(detail)
