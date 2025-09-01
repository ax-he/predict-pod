# ~/pred-svc/listen.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
import math, os, subprocess, json, tempfile, shutil

app = FastAPI(title="pred-svc", version="0.3")

# ---------- 读机器峰值（来自环境变量 / ConfigMap） ----------
def getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

PEAK_FLOPS_GFLOPS = getenv_float("PEAK_FLOPS_GFLOPS", 50.0)     # 例：50 GFLOPS
PEAK_MEM_BANDWIDTH_GBS = getenv_float("PEAK_MEM_BANDWIDTH_GBS", 15.0)  # 例：15 GB/s
LOCAL_TIME_BUDGET_S = getenv_float("LOCAL_TIME_BUDGET_S", 10.0)  # 默认本地预算 10s

PEAK_FLOPS = PEAK_FLOPS_GFLOPS * 1e9         # FLOPs/s
PEAK_BW    = PEAK_MEM_BANDWIDTH_GBS * (1024**3)  # Bytes/s（按GiB/s）

# ---------- 请求/响应模型 ----------
DType = Literal["float16", "float32", "float64", "int8"]

class GemmSpec(BaseModel):
    m: int; n: int; k: int
    dtype: DType = "float32"

class FFT1DSpec(BaseModel):
    n: int
    batches: int = 1
    dtype: DType = "float32"

class TranscodeSpec(BaseModel):
    # 任选其一：直接给媒体路径/URL，或手填元数据
    input: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_s: Optional[float] = None
    codec: Optional[str] = None        # 用于经验权重
    preset: Optional[str] = None       # ultrafast/fast/medium/slow...

class Kernel(BaseModel):
    type: Literal["gemm", "fft1d", "transcode"]
    params: Dict[str, Any]
    flops: float
    bytes: float
    t_lower_bound_s: float

class PredictRequest(BaseModel):
    gemm: Optional[List[GemmSpec]] = None
    fft1d: Optional[List[FFT1DSpec]] = None
    transcode: Optional[List[TranscodeSpec]] = None
    time_budget_s: Optional[float] = None  # 不填用 LOCAL_TIME_BUDGET_S

class Decision(BaseModel):
    allow: bool
    message: str
    reason: str
    policy: Dict[str, Any]

class PredictResponse(BaseModel):
    kernels: List[Kernel]
    sum_flops: float
    sum_bytes: float
    time_lower_bound_s: float
    decision: Decision
    peaks: Dict[str, float]

# ---------- 工具 ----------
DTYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "int8": 1}

def parse_rate(r: Optional[str]) -> float:
    """把 '30000/1001' 或 '30/1' 解析为 float；异常时返回 0"""
    try:
        if not r: return 0.0
        if "/" in r:
            a, b = r.split("/")
            return float(a) / float(b) if float(b) != 0 else 0.0
        return float(r)
    except Exception:
        return 0.0

def gemm_cost(spec: GemmSpec):
    P = DTYPE_BYTES[spec.dtype]; m, n, k = spec.m, spec.n, spec.k
    flops = 2.0 * m * n * k
    bytes_ = P * (k * (m + n) + 2 * m * n)
    t = max(flops / PEAK_FLOPS, bytes_ / PEAK_BW)
    return flops, bytes_, t

def fft1d_cost(spec: FFT1DSpec):
    P = DTYPE_BYTES[spec.dtype]; n = spec.n; batches = spec.batches
    flops_single = 5.0 * n * (math.log(n, 2))
    flops = flops_single * batches
    bytes_per_batch = P * n * 8  # 粗略
    bytes_ = bytes_per_batch * batches
    t = max(flops / PEAK_FLOPS, bytes_ / PEAK_BW)
    return flops, bytes_, t

def weight_from_codec_preset(codec: Optional[str], preset: Optional[str]) -> float:
    codec_w = {None:1.0, "h264":1.0, "hevc":1.8, "h265":1.8, "av1":3.0, "vp9":2.0}
    preset_w = {None:1.0, "ultrafast":0.5, "superfast":0.7, "veryfast":0.8,
                "faster":0.9, "fast":1.0, "medium":1.2, "slow":1.6, "veryslow":2.0}
    key_c = str(codec).lower() if codec else None
    key_p = str(preset).lower() if preset else None
    return codec_w.get(key_c, 1.0) * preset_w.get(key_p, 1.0)

def run_ffprobe_json(input_path: str) -> Dict[str, Any]:
    """ffprobe -v error -print_format json -show_format -show_streams"""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", input_path],
            stderr=subprocess.STDOUT
        )
        return json.loads(out.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400,
            detail=f"ffprobe failed: {e.output.decode('utf-8', errors='ignore')}")

def transcode_cost(spec: TranscodeSpec):
    # 若给了 input，则用 ffprobe 抽取元数据
    w, h, fps, duration = spec.width, spec.height, spec.fps, spec.duration_s
    codec = spec.codec
    if spec.input and (w is None or h is None or fps is None or duration is None or codec is None):
        info = run_ffprobe_json(spec.input)
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not vstreams:
            raise HTTPException(status_code=400, detail="no video stream found by ffprobe")
        vs = vstreams[0]
        w = w or int(vs.get("width", 0))
        h = h or int(vs.get("height", 0))
        fps = fps or parse_rate(vs.get("avg_frame_rate") or vs.get("r_frame_rate"))
        duration = duration or float(info.get("format", {}).get("duration", "0") or 0)
        codec = codec or vs.get("codec_name")

    if not all([w, h, fps, duration]):
        raise HTTPException(status_code=400,
            detail="transcode requires width,height,fps,duration or an input readable by ffprobe")

    pixels = float(w) * float(h) * float(fps) * float(duration)
    weight = weight_from_codec_preset(codec, spec.preset)

    # 占位：用像素量×权重估 FLOPs；Bytes 近似读+写
    flops = 200.0 * pixels * weight     # 之后可用回归替换
    bytes_ = 3.0 * pixels
    t = max(flops / PEAK_FLOPS, bytes_ / PEAK_BW)
    return flops, bytes_, t

def decide_and_pack(kernels: List[Kernel], time_budget_s: Optional[float]) -> PredictResponse:
    if not kernels:
        raise HTTPException(status_code=400, detail="no kernels provided")

    sum_flops = sum(k.flops for k in kernels)
    sum_bytes = sum(k.bytes for k in kernels)
    time_lower_bound = sum(max(k.flops / PEAK_FLOPS, k.bytes / PEAK_BW) for k in kernels)

    budget = time_budget_s if time_budget_s is not None else LOCAL_TIME_BUDGET_S
    req_flops_rate = sum_flops / budget
    req_bw_rate = sum_bytes / budget

    over_compute = req_flops_rate > PEAK_FLOPS
    over_bw = req_bw_rate > PEAK_BW

    if over_compute or over_bw:
        allow = False
        message = "no, go to edge"
        reason = "required compute or bandwidth exceeds local peak for the time budget"
    else:
        allow = True
        message = "yes, do it right now"
        reason = "fits within local peaks for the time budget"

    decision = Decision(
        allow=allow,
        message=message,
        reason=reason,
        policy={
            "time_budget_s": budget,
            "required_flops_per_s": req_flops_rate,
            "required_bw_Bps": req_bw_rate,
            "peaks": {"FLOPs_per_s": PEAK_FLOPS, "BW_Bps": PEAK_BW},
        },
    )

    return PredictResponse(
        kernels=kernels,
        sum_flops=sum_flops,
        sum_bytes=sum_bytes,
        time_lower_bound_s=time_lower_bound,
        decision=decision,
        peaks={"peak_flops_gflops": PEAK_FLOPS_GFLOPS, "peak_mem_bandwidth_gbs": PEAK_MEM_BANDWIDTH_GBS},
    )

# ---------- 健康检查 ----------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "peak_flops_gflops": PEAK_FLOPS_GFLOPS,
        "peak_mem_bandwidth_gbs": PEAK_MEM_BANDWIDTH_GBS,
        "local_time_budget_s": LOCAL_TIME_BUDGET_S
    }

# ---------- 通用 JSON 预测 ----------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    kernels: List[Kernel] = []

    if req.gemm:
        for g in req.gemm:
            fl, by, t = gemm_cost(g)
            kernels.append(Kernel(type="gemm", params=g.model_dump(), flops=fl, bytes=by, t_lower_bound_s=t))

    if req.fft1d:
        for f in req.fft1d:
            fl, by, t = fft1d_cost(f)
            kernels.append(Kernel(type="fft1d", params=f.model_dump(), flops=fl, bytes=by, t_lower_bound_s=t))

    if req.transcode:
        for tr in req.transcode:
            fl, by, t = transcode_cost(tr)
            kernels.append(Kernel(type="transcode", params=tr.model_dump(), flops=fl, bytes=by, t_lower_bound_s=t))

    return decide_and_pack(kernels, req.time_budget_s)

# ---------- 新增：上传视频 → 自动 ffprobe → 决策 ----------
@app.post("/predict/transcode/upload", response_model=PredictResponse)
async def predict_transcode_upload(
    time_budget_s: float = Form(...),
    preset: Optional[str] = Form(None),          # 可选：用户声明编码预设
    file: UploadFile = File(...),                # 视频文件
):
    # FastAPI 处理表单/文件上传需要 python-multipart 支持
    # 参考官方文档：https://fastapi.tiangolo.com/tutorial/request-forms-and-files/
    # 把上传内容落到容器 /tmp，避免一次性读入内存
    suffix = os.path.splitext(file.filename or "")[1]
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmpf = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=suffix or ".bin")
    tmp_path = tmpf.name
    try:
        with tmpf as f:
            # 以流式方式拷贝
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB/chunk
                if not chunk:
                    break
                f.write(chunk)

        # 用 ffprobe 抽取元数据
        info = run_ffprobe_json(tmp_path)
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not vstreams:
            raise HTTPException(status_code=400, detail="no video stream found by ffprobe")

        vs = vstreams[0]
        width  = int(vs.get("width", 0))
        height = int(vs.get("height", 0))
        fps    = parse_rate(vs.get("avg_frame_rate") or vs.get("r_frame_rate"))
        duration_s = float(info.get("format", {}).get("duration", "0") or 0)
        codec = vs.get("codec_name")

        # 构造一次 transcode kernel
        spec = TranscodeSpec(
            width=width, height=height, fps=fps, duration_s=duration_s,
            codec=codec, preset=preset
        )
        flops, bytes_, t = transcode_cost(spec)
        kernel = Kernel(
            type="transcode",
            params={"width":width, "height":height, "fps":fps, "duration_s":duration_s,
                    "codec":codec, "preset":preset},
            flops=flops, bytes=bytes_, t_lower_bound_s=t
        )
        return decide_and_pack([kernel], time_budget_s)

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
