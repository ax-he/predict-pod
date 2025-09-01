# ~/pred-svc/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import os, math, json, subprocess, shlex

class Kernel(BaseModel):
    kind: str                 # "gemm" | "fft" | "transcode"
    params: dict              # {"m":..,"n":..,"k":..} / {"N":..} / {"path":..}

class PredictIn(BaseModel):
    kernels: list[Kernel]

class PredictOut(BaseModel):
    per_kernel_sec: list[float]
    time_lower_bound_sec: float
    peak_flops_gflops: float
    peak_mem_bandwidth_gbs: float

app = FastAPI()

def peaks():
    # 由 ConfigMap 注入的环境变量提供；未注入时 fallback
    return (
        float(os.getenv("PEAK_FLOPS_GFLOPS", "50")),
        float(os.getenv("PEAK_MEM_BANDWIDTH_GBS", "10"))
    )

def gemm_cost(m,n,k, bpe=4):
    flops = 2.0*m*n*k                       # 经典 2mnk
    bytes_ = bpe*(m*k + k*n + m*n)          # 粗估读写
    return flops, bytes_

def fft_cost(N, bpe=8):
    flops = 5.0 * N * math.log2(max(2, N))  # 常用上界 ~5Nlog2N
    bytes_ = bpe * N
    return flops, bytes_

def probe_media(path:str):
    cmd = f'ffprobe -v error -print_format json -show_format -show_streams {shlex.quote(path)}'
    meta = subprocess.check_output(cmd, shell=True, text=True)
    m = json.loads(meta)
    v = next((s for s in m.get("streams",[]) if s.get("codec_type")=="video"), {})
    w,h = int(v.get("width",0) or 0), int(v.get("height",0) or 0)
    fps_txt = v.get("r_frame_rate","0/1"); num,den = (fps_txt.split("/") + ["1"])[:2]
    fps = (int(num)/int(den)) if int(den) else 0
    dur = float(m.get("format",{}).get("duration",0) or 0)
    codec = v.get("codec_name","unknown")
    return w,h,fps,dur,codec

def transcode_cost(path):
    w,h,fps,dur,codec = probe_media(path)
    work_pixels = w*h*fps*dur
    codec_w = {"h264":1.0, "hevc":1.5, "av1":2.0}.get(codec, 1.2)
    flops  = work_pixels * codec_w
    bytes_ = w*h*3 * fps * dur            # 近似：RGB 3B/px
    return flops, bytes_

def roofline_t(flops, bytes_, peak_gflops, peak_gbs):
    t_compute = flops  / (peak_gflops*1e9)
    t_memory  = bytes_ / (peak_gbs   *1e9)
    return max(t_compute, t_memory)

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    peak_f, peak_b = peaks()
    per = []
    total = 0.0
    for k in inp.kernels:
        if k.kind=="gemm":
            fl, by = gemm_cost(int(k.params["m"]), int(k.params["n"]), int(k.params["k"]))
        elif k.kind=="fft":
            fl, by = fft_cost(int(k.params["N"]))
        elif k.kind=="transcode":
            fl, by = transcode_cost(k.params["path"])
        else:
            raise ValueError("unknown kind")
        t = roofline_t(fl, by, peak_f, peak_b)
        per.append(t); total += t
    return PredictOut(per_kernel_sec=per, time_lower_bound_sec=total,
                      peak_flops_gflops=peak_f, peak_mem_bandwidth_gbs=peak_b)

