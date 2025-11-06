# -*- coding: utf-8 -*-
"""
Backend: Cost Estimate + Gate + Routing + Answer (WebSocket)
- FastAPI + WebSocket
- Uses OpenAI Python SDK (OpenRouter 兼容；或官方 OpenAI，取决于 api_base)
- 把“估算与路由”输出到左框，把“真实回答”输出到右框
"""

import os, json, math, asyncio, traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ========== OpenAI SDK (用于 OpenRouter 或 官方 OpenAI) ==========
from openai import OpenAI

# ========== 可选：HuggingFace tokenizer（本地目录） ==========
try:
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
    _HAS_HF = True
except Exception:
    _HAS_HF = False

# ========== 系统监控库 ==========
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    import pynvml
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

try:
    import wmi
    _HAS_WMI = True
except Exception:
    _HAS_WMI = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://127.0.0.1:18000", "http://localhost:18000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 估算相关结构与公式 =====================
@dataclass
class ModelCfg:
    name: str
    L: int              # 层数
    hidden: int         # hidden_size
    n_head: int         # 注意力头数
    n_kv_head: int      # KV 头数（GQA/MQA）
    n_param: float      # 参数个数
    weight_bytes: float # 权重量化字节：BF16/FP16=2, INT8=1, INT4=0.5

@dataclass
class RuntimeOpts:
    kv_bytes: float = 2.0           # KV 精度字节
    kv_frag: float = 0.05           # PagedAttention 残余碎片率
    extra_vram_gb: float = 1.0      # 框架/工作区预留

@dataclass
class HW:
    name: str
    flops_fp16: float   # 有效 FLOPs/s（FP16/BF16）
    hbm_bw_Bps: float   # 显存带宽 Bytes/s
    vram_gb: float      # 总显存容量（GB）

def _safe_fmt_num(x: float, nd=3) -> str:
    try:
        if x == float("inf") or x != x:
            return "N/A"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def kv_bytes_per_token(cfg: ModelCfg, rt: RuntimeOpts) -> float:
    head_dim = max(1, cfg.hidden // max(1, cfg.n_head))
    kv_heads = cfg.n_kv_head if cfg.n_kv_head else cfg.n_head
    base = 2.0 * cfg.L * (kv_heads * head_dim) * rt.kv_bytes   # *2: K & V
    return base * (1.0 + rt.kv_frag)

def vram_weights_gb(cfg: ModelCfg) -> float:
    return (cfg.n_param * cfg.weight_bytes) / (1024**3)

def vram_kv_gb(cfg: ModelCfg, rt: RuntimeOpts, B: int, S_eff: int, O_eff: int) -> float:
    per_tok = kv_bytes_per_token(cfg, rt)
    total = per_tok * B * (S_eff + O_eff)
    return total / (1024**3)

def prefill_decode_times(N: float, B: int, S_eff: int, O_eff: int, flops: float, bw: float) -> Dict[str, float]:
    # 取 compute/memory 上界
    t_comp_prefill = (2.0 * N * B * S_eff) / max(1e-9, flops)
    t_mem_prefill  = (2.0 * N) / max(1e-9, bw)
    TTFT = max(t_comp_prefill, t_mem_prefill)

    t_comp_decode = (2.0 * N * B) / max(1e-9, flops)
    t_mem_decode  = (2.0 * N) / max(1e-9, bw)
    TPOT = max(t_comp_decode, t_mem_decode)

    total_model = TTFT + TPOT * max(0, O_eff)
    tokens_per_s = 1.0 / TPOT if TPOT > 0 else float("inf")
    return {
        "TTFT_s": TTFT,
        "TPOT_s": TPOT,
        "Total_model_s": total_model,
        "tokens_per_s": tokens_per_s,
        "t_compute_prefill_s": t_comp_prefill,
        "t_memory_prefill_s": t_mem_prefill,
        "t_compute_decode_s": t_comp_decode,
        "t_memory_decode_s": t_mem_decode,
    }

# ===================== tokenizer & 任务识别 & 输出长度 =====================
class TokenCounter:
    def __init__(self, tok_dir: str):
        self.name = tok_dir or "."
        self.tok = None
        if _HAS_HF and tok_dir:
            tok_json = os.path.join(tok_dir, "tokenizer.json")
            if os.path.isfile(tok_json):
                try:
                    self.tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)
                except Exception:
                    self.tok = None
            if self.tok is None:
                try:
                    self.tok = AutoTokenizer.from_pretrained(
                        tok_dir, use_fast=True, trust_remote_code=True, local_files_only=True
                    )
                except Exception:
                    self.tok = None

    def count(self, text: str) -> int:
        if not text:
            return 1
        t = text.strip()
        if not t:
            return 1
        if self.tok is not None:
            try:
                return len(self.tok.encode(t, add_special_tokens=False))
            except Exception:
                pass
        # 退化：中英混合 1 token ≈ 3.5 chars
        return max(1, int(math.ceil(len(t) / 3.5)))

def parse_len_hint(q: str) -> Optional[int]:
    if not q: return None
    import re
    s = q.lower()
    rules = [
        (r"(\d+)\s*字",1.0),(r"(\d+)\s*词",1.0),(r"(\d+)\s*words?",1.0),
        (r"(\d+)\s*句|sentences?",15.0),(r"(\d+)\s*段|paragraphs?",100.0),
        (r"(\d+)\s*行|lines?",20.0),(r"(\d+)\s*页|pages?",500.0),
        (r"(?:约|大概|左右)\s*(\d+)\s*字",1.0),(r"(?:about|approximately|around)\s*(\d+)\s*words?",1.0),
    ]
    for pat, coef in rules:
        m = re.search(pat, s, re.I)
        if m:
            try:
                return int(int(m.group(1))*coef)
            except Exception:
                pass
    for pat, coef in [(r"(\d+)\s*到\s*(\d+)\s*字",1.0),(r"(\d+)\s*-\s*(\d+)\s*words?",1.0)]:
        m = re.search(pat, s, re.I)
        if m:
            try:
                a,b = int(m.group(1)), int(m.group(2))
                return int(((a+b)/2)*coef)
            except Exception:
                pass
    return None

def detect_task(q: str) -> str:
    if not q: return "qa_short"
    import re
    t = q.lower().strip()
    if re.search(r"(why|how|explain|原理|机制|原因|为什么|为何|如何|解释|讲解)", t):
        if re.search(r"\d+\s*[\+\-\*/]\s*\d+|\d+\s*\^\s*\d+|[\=\≈~]\s*\d+", t):
            return "math"
        return "qa_explain"
    if re.search(r"(code|程序|编程|实现|函数|算法|python|java|cpp|c\+\+|javascript|js|def |class |import )", t):
        return "code"
    if re.search(r"(计算|数学|公式|方程|函数|导数|积分|概率|统计|几何|代数|solve|equation|derivative|integral|probability)", t):
        return "math"
    if re.search(r"(翻译|译为|译成|英文|中文|日语|法语|德语|translate|translation)", t):
        return "translate"
    if re.search(r"(总结|概括|摘要|概要|简述|tl;dr|tldr|summarize|summary)", t):
        return "summarize"
    if re.search(r"(写一篇|创作|故事|小说|诗歌|essay|article|story|poem|想象|假如|如果|假设|imagine)", t):
        return "creative"
    return "qa_short" if len(t.split())<=8 else "qa_explain"

PRIORS = {
    "translate": {"p50_a": 1.00, "p50_b": 0,   "p90_a": 1.20, "p90_b": 10},
    "summarize": {"p50_a": 0.30, "p50_b": 30,  "p90_a": 0.45, "p90_b": 50},
    "code":      {"p50_a": 0.65, "p50_b": 40,  "p90_a": 0.90, "p90_b": 80},
    "math":      {"p50_a": 0.55, "p50_b": 30,  "p90_a": 0.80, "p90_b": 60},
    "qa_explain":{"p50_a": 0.60, "p50_b": 40,  "p90_a": 0.90, "p90_b": 80},
    "qa_short":  {"p50_a": 0.35, "p50_b": 40,  "p90_a": 0.55, "p90_b": 80},
    "creative":  {"p50_a": 0.80, "p50_b": 100, "p90_a": 1.20, "p90_b": 200},
}

def o_pred(S:int, task:str, params_B: Optional[float], len_hint: Optional[int]) -> Tuple[float,float,float,bool]:
    pri = PRIORS.get(task, PRIORS["qa_short"])
    p50 = pri["p50_a"]*S + pri["p50_b"]
    p90 = pri["p90_a"]*S + pri["p90_b"]
    mean = 0.5*(p50+p90)
    if params_B and params_B>0:
        f = 1.0 + 0.12*math.log10(max(1e-9, params_B/8.0))
        f = min(1.30, max(0.85, f))
        p50*=f; p90*=f; mean*=f
    used=False
    if isinstance(len_hint,int) and len_hint>0:
        used=True
        p50=float(len_hint)
        p90=float(int(round(len_hint*1.2)))
        mean=0.5*(p50+p90)
    # clip
    def clip(x): 
        try:
            v = int(round(x))
            return float(min(32768, max(1, v)))
        except Exception:
            return float(1)
    return clip(mean), clip(p50), clip(p90), used

# ===================== Gate & Routing =====================
def need_legal_verbatim(q: str) -> bool:
    if not q: return False
    t=q.lower()
    keys=["逐字","页码","原文","引用","原始官方","oj","ojeu","cited in the ojeu","法规","法案","article","annex"]
    return any(k in t for k in keys)

def can_small_model_handle(task:str, q:str) -> bool:
    if need_legal_verbatim(q):
        return False
    # 注意：detect_task返回"code"而不是"programming"
    if task in ("arithmetic","format_conversion","programming","code","math","qa_explain","qa_short","summarize","creative","translate"):
        return True
    return False

# ===================== OpenAI/ OpenRouter 调用 =====================
def build_client(api_key: str, api_base: str) -> OpenAI:
    # OpenRouter 是 OpenAI 兼容接口：base_url="https://openrouter.ai/api/v1"
    # 官方 OpenAI 则 base_url="https://api.openai.com/v1"（也可不传）
    # 设置更长的超时时间（120秒）
    if api_base:
        return OpenAI(api_key=api_key, base_url=api_base, timeout=120.0)
    return OpenAI(api_key=api_key, timeout=120.0)

async def get_answer_text_async(client: OpenAI, model: str, question: str) -> str:
    """
    为避免阻塞事件循环，这里把同步 API 放到线程里执行。
    返回单段文本，然后我们在 WS 里“伪流式”逐块发送，兼容前端 UI。
    """
    def _call():
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"你是中文问答助手；若题目要求逐字引用且你无法核对，请仅输出\"置信度不足\"。"},
                {"role":"user","content":question},
            ],
            temperature=0.2,
            max_tokens=4096,  # 增加生成长度限制
            extra_body={"usage":{"include":True}, "reasoning":{"exclude":True}}
        )
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""
    return await asyncio.to_thread(_call)

async def send_error(ws: WebSocket, msg: str):
    try:
        await ws.send_json({"type":"error","data": msg})
    except Exception:
        pass

async def send_log(ws: WebSocket, text: str):
    # 左框输出
    try:
        await ws.send_json({"type":"log","data": text})
    except Exception:
        pass

async def send_status(ws: WebSocket, text: str):
    try:
        await ws.send_json({"type":"status","data": text})
    except Exception:
        pass

async def send_answer(ws: WebSocket, text: str):
    """
    把整段文本切成小块，逐块推给前端（右框），模拟流式。
    """
    CHUNK = 300
    if not text:
        await ws.send_json({"type":"answer_done"})
        return
    i = 0
    n = len(text)
    while i < n:
        await ws.send_json({"type":"answer_chunk", "data": text[i:i+CHUNK]})
        i += CHUNK
        await asyncio.sleep(0)  # 让出事件循环
    await ws.send_json({"type":"answer_done"})

# ===================== 系统资源监控 =====================
def get_system_stats() -> Dict[str, Any]:
    """获取系统资源使用情况（支持NVIDIA/AMD/Intel GPU）"""
    stats = {
        "cpu_percent": 0.0,
        "disk_usage": 0.0,
        "disk_read_mb": 0.0,
        "disk_write_mb": 0.0,
        "gpu_count": 0,
        "gpus": []
    }
    
    # CPU使用率
    if _HAS_PSUTIL:
        try:
            stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        except Exception:
            pass
        
        # 磁盘使用率（C盘）
        try:
            disk = psutil.disk_usage('C:\\')
            stats["disk_usage"] = disk.percent
        except Exception:
            pass
        
        # 磁盘IO
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                stats["disk_read_mb"] = round(disk_io.read_bytes / (1024**2), 2)
                stats["disk_write_mb"] = round(disk_io.write_bytes / (1024**2), 2)
        except Exception:
            pass
    
    # GPU信息 - 通用方案（通过WMI获取所有GPU）
    gpu_list = []
    nvidia_gpu_map = {}  # 存储NVIDIA GPU的详细信息
    
    # 第一步：通过WMI获取所有GPU的基本信息
    if _HAS_WMI:
        try:
            w = wmi.WMI()
            wmi_gpus = w.Win32_VideoController()
            for idx, gpu in enumerate(wmi_gpus):
                gpu_name = gpu.Name or "Unknown GPU"
                adapter_ram = 0
                try:
                    if gpu.AdapterRAM:
                        adapter_ram = int(gpu.AdapterRAM) / (1024**3)  # 转换为GB
                except Exception:
                    pass
                
                # 判断GPU厂商
                vendor = "Unknown"
                if "NVIDIA" in gpu_name.upper():
                    vendor = "NVIDIA"
                elif "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper():
                    vendor = "AMD"
                elif "INTEL" in gpu_name.upper():
                    vendor = "Intel"
                
                gpu_info = {
                    "id": idx,
                    "name": gpu_name,
                    "vendor": vendor,
                    "mem_total_gb": round(adapter_ram, 2) if adapter_ram > 0 else None,
                    "gpu_util": None,  # WMI不提供利用率
                    "mem_util": None,
                    "mem_used_gb": None,
                    "mem_percent": None,
                    "temperature": None,
                    "driver_version": gpu.DriverVersion or None
                }
                gpu_list.append(gpu_info)
        except Exception as e:
            print(f"WMI GPU检测错误: {e}")
    
    # 第二步：对于NVIDIA GPU，使用pynvml获取详细信息
    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # GPU利用率
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    mem_util = util.memory
                except Exception:
                    gpu_util = None
                    mem_util = None
                
                # 显存信息
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                    mem_total_gb = mem_info.total / (1024**3)
                    mem_percent = (mem_info.used / mem_info.total) * 100
                except Exception:
                    mem_used_gb = None
                    mem_total_gb = None
                    mem_percent = None
                
                # 温度
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = None
                
                nvidia_gpu_map[name] = {
                    "gpu_util": gpu_util,
                    "mem_util": mem_util,
                    "mem_used_gb": round(mem_used_gb, 2) if mem_used_gb else None,
                    "mem_total_gb": round(mem_total_gb, 2) if mem_total_gb else None,
                    "mem_percent": round(mem_percent, 1) if mem_percent else None,
                    "temperature": temp
                }
        except Exception as e:
            print(f"NVIDIA GPU监控错误: {e}")
    
    # 第三步：合并信息
    for gpu in gpu_list:
        if gpu["vendor"] == "NVIDIA" and gpu["name"] in nvidia_gpu_map:
            # 用pynvml的详细信息更新
            nvidia_data = nvidia_gpu_map[gpu["name"]]
            gpu.update(nvidia_data)
    
    stats["gpu_count"] = len(gpu_list)
    stats["gpus"] = gpu_list
    
    return stats

async def system_monitor_task(websocket: WebSocket, stop_event: asyncio.Event):
    """后台任务：定期发送系统资源信息"""
    try:
        while not stop_event.is_set():
            stats = await asyncio.to_thread(get_system_stats)
            try:
                await websocket.send_json({"type": "system_stats", "data": stats})
            except Exception:
                break
            # 每2秒更新一次
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=2.0)
                break  # 如果stop_event被设置，退出循环
            except asyncio.TimeoutError:
                continue  # 超时继续下一次循环
    except Exception as e:
        print(f"系统监控任务错误: {e}")

# ===================== WebSocket 主流程 =====================
@app.websocket("/ws")
async def ws_main(websocket: WebSocket):
    await websocket.accept()
    print("connection open")
    
    # 启动系统监控后台任务
    stop_monitor = asyncio.Event()
    monitor_task = asyncio.create_task(system_monitor_task(websocket, stop_monitor))
    
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except Exception as e:
                await send_error(websocket, f"invalid json: {e}")
                continue

            if (payload.get("action") or "").lower() != "estimate_and_answer":
                await send_error(websocket, f"unknown action: {payload.get('action')}")
                continue

            # ---- 读入前端参数（均有默认值）----
            q = payload.get("question") or ""
            small_model_id = payload.get("small_model") or "deepseek/deepseek-r1-0528-qwen3-8b:free"
            large_model_id = payload.get("large_model") or "tngtech/deepseek-r1t-chimera:free"
            small_tok_dir = payload.get("small_tok_dir") or ""
            large_tok_dir = payload.get("large_tok_dir") or ""
            small_params_B = float(payload.get("small_params_B", 8))
            large_params_B = float(payload.get("large_params_B", 671))
            small_weight_bytes = float(payload.get("small_weight_bytes", 2.0))
            large_weight_bytes = float(payload.get("large_weight_bytes", 2.0))

            # 硬件（含多卡支持）
            small_gpus = int(payload.get("small_gpus", 1))
            large_gpus = int(payload.get("large_gpus", 1))
            
            small_hw = HW(
                name=payload.get("small_hw_name","RTX 3060 12GB"),
                flops_fp16=float(payload.get("small_flops_fp16", 12.7e12)) * small_gpus,
                hbm_bw_Bps=float(payload.get("small_hbm_bw_Bps", 360e9)) * small_gpus,
                vram_gb=float(payload.get("small_vram_gb", 12.0)),
            )
            large_hw = HW(
                name=payload.get("large_hw_name","A100 80GB"),
                flops_fp16=float(payload.get("large_flops_fp16", 312e12)) * large_gpus,
                hbm_bw_Bps=float(payload.get("large_hbm_bw_Bps", 2039e9)) * large_gpus,
                vram_gb=float(payload.get("large_vram_gb", 80.0)) * large_gpus,
            )

            rt = RuntimeOpts(
                kv_bytes=float(payload.get("kv_bytes", 2.0)),
                kv_frag=float(payload.get("kv_frag", 0.05)),
                extra_vram_gb=float(payload.get("extra_vram_overhead_gb", 1.0)),
            )

            api_key = payload.get("api_key") or ""
            api_base = payload.get("api_base") or "https://openrouter.ai/api/v1"

            if not q:
                await send_error(websocket, "question is empty")
                continue

            # ---- 计 S / 估 O ----
            tc_small = TokenCounter(small_tok_dir) if small_tok_dir else TokenCounter(".")
            tc_large = TokenCounter(large_tok_dir) if large_tok_dir else TokenCounter(".")
            S_small = tc_small.count(q)
            S_large = tc_large.count(q)
            task = detect_task(q)
            hint = parse_len_hint(q)

            O_mean_s, O_p50_s, O_p90_s, used_s = o_pred(S_small, task, small_params_B, hint)
            O_mean_l, O_p50_l, O_p90_l, used_l = o_pred(S_large, task, large_params_B, hint)

            small_cfg_m = ModelCfg(
                name=small_model_id, L=int(payload.get("small_L", 32)),
                hidden=int(payload.get("small_hidden", 4096)),
                n_head=int(payload.get("small_n_head", 32)),
                n_kv_head=int(payload.get("small_n_kv_head", 8)),
                n_param=small_params_B * 1e9, weight_bytes=small_weight_bytes
            )
            large_cfg_m = ModelCfg(
                name=large_model_id, L=int(payload.get("large_L", 64)),
                hidden=int(payload.get("large_hidden", 9216)),
                n_head=int(payload.get("large_n_head", 72)),
                n_kv_head=int(payload.get("large_n_kv_head", 8)),
                n_param=large_params_B * 1e9, weight_bytes=large_weight_bytes
            )

            # 时延/显存估算
            t_s = prefill_decode_times(small_cfg_m.n_param, 1, S_small, int(O_p50_s), small_hw.flops_fp16, small_hw.hbm_bw_Bps)
            kv_s = vram_kv_gb(small_cfg_m, rt, 1, S_small, int(O_p50_s))
            w_s  = vram_weights_gb(small_cfg_m)
            vram_s = kv_s + w_s + rt.extra_vram_gb

            t_l = prefill_decode_times(large_cfg_m.n_param, 1, S_large, int(O_p50_l), large_hw.flops_fp16, large_hw.hbm_bw_Bps)
            kv_l = vram_kv_gb(large_cfg_m, rt, 1, S_large, int(O_p50_l))
            w_l  = vram_weights_gb(large_cfg_m)
            vram_l = kv_l + w_l + rt.extra_vram_gb

            # 左侧输出（格式化避免 NaN/Inf）
            text_left = []
            text_left.append(f"Task: {task} | len_hint={hint} used_hint={used_s or used_l}")
            text_left.append(f"-- Small [{small_hw.name}] {small_model_id} (x{small_gpus} GPU)")
            text_left.append(f"S={S_small}  p50={_safe_fmt_num(O_p50_s,1)}  p90={_safe_fmt_num(O_p90_s,1)}")
            text_left.append(f"TTFT={_safe_fmt_num(t_s['TTFT_s']*1000,1)}ms  TPOT={_safe_fmt_num(t_s['TPOT_s']*1000,2)}ms  "
                             f"Total={_safe_fmt_num(t_s['Total_model_s'],2)}s  ~{_safe_fmt_num(t_s['tokens_per_s'],2)} tok/s")
            text_left.append(f"VRAM: weights={_safe_fmt_num(w_s)}GB  kv={_safe_fmt_num(kv_s)}GB  total={_safe_fmt_num(vram_s)}GB  @{small_hw.name}")
            text_left.append("")
            text_left.append(f"-- Large [{large_hw.name}] {large_model_id} (x{large_gpus} GPU)")
            text_left.append(f"S={S_large}  p50={_safe_fmt_num(O_p50_l,1)}  p90={_safe_fmt_num(O_p90_l,1)}")
            text_left.append(f"TTFT={_safe_fmt_num(t_l['TTFT_s']*1000,1)}ms  TPOT={_safe_fmt_num(t_l['TPOT_s']*1000,2)}ms  "
                             f"Total={_safe_fmt_num(t_l['Total_model_s'],2)}s  ~{_safe_fmt_num(t_l['tokens_per_s'],2)} tok/s")
            text_left.append(f"VRAM: weights={_safe_fmt_num(w_l)}GB  kv={_safe_fmt_num(kv_l)}GB  total={_safe_fmt_num(vram_l)}GB  @{large_hw.name}")
            await send_log(websocket, "\n".join(text_left))

            # 路由：小模型更快 且 能完成 → 小；否则大
            small_faster = t_s["Total_model_s"] <= t_l["Total_model_s"]
            small_ok = can_small_model_handle(task, q)
            if small_faster and small_ok:
                routed_model = small_model_id
                routed_hw = small_hw.name
            else:
                routed_model = large_model_id
                routed_hw = large_hw.name

            await send_log(websocket, f"\n\n--> Routed to: {routed_model}  @{routed_hw}\n")
            await send_status(websocket, f"[route] {routed_model}")

            if not api_key:
                await send_error(websocket, "API Key 为空；请在左上填入有效 Key")
                continue

            # 真正回答（线程中执行，避免阻塞）
            try:
                client = build_client(api_key, api_base)
                await send_status(websocket, f"[answer] model={routed_model}")
                # 通知前端开始显示"正在思考"
                await websocket.send_json({"type":"thinking_start"})
                text = await get_answer_text_async(client, routed_model, q)
                if not text:
                    await send_error(websocket, "[answer] 空响应或解析失败")
                else:
                    await send_answer(websocket, text)
            except Exception as e:
                await send_error(websocket, f"[answer] {type(e).__name__}: {e}")

    except WebSocketDisconnect:
        print("connection closed")
    except Exception as e:
        print("server exception:", repr(e))
        traceback.print_exc()
        try:
            await send_error(websocket, f"server exception: {type(e).__name__}: {e}")
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        # 停止系统监控任务
        stop_monitor.set()
        try:
            await asyncio.wait_for(monitor_task, timeout=3.0)
        except asyncio.TimeoutError:
            monitor_task.cancel()
        except Exception:
            pass

if __name__ == "__main__":
    # pip install fastapi uvicorn openai transformers (transformers 可选)
    uvicorn.run(app, host="0.0.0.0", port=8000)
