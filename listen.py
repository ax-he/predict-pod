# ~/pred-svc/listen.py
# 9.12 v0.8 test
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
import math, os, subprocess, json, tempfile, re, uuid, time

app = FastAPI(title="pred-svc", version="0.7")

# ---------- 读机器峰值（来自环境变量 / ConfigMap） ----------
def getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

PEAK_FLOPS_GFLOPS = getenv_float("PEAK_FLOPS_GFLOPS", 50.0)      # 例：50 GFLOPS
PEAK_MEM_BANDWIDTH_GBS = getenv_float("PEAK_MEM_BANDWIDTH_GBS", 15.0)  # 例：15 GB/s
LOCAL_TIME_BUDGET_S = getenv_float("LOCAL_TIME_BUDGET_S", 10.0)  # 默认本地预算 10s

PEAK_FLOPS = PEAK_FLOPS_GFLOPS * 1e9              # FLOPs/s
PEAK_BW    = PEAK_MEM_BANDWIDTH_GBS * (1024**3)   # Bytes/s（按GiB/s）

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
    input: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_s: Optional[float] = None
    codec: Optional[str] = None
    preset: Optional[str] = None

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

    flops = 200.0 * pixels * weight     # 粗略；可用回归替换
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
    preset: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    suffix = os.path.splitext(file.filename or "")[1]
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmpf = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=suffix or ".bin")
    tmp_path = tmpf.name
    try:
        with tmpf as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB/chunk
                if not chunk:
                    break
                f.write(chunk)

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


# =====================================================================
# ===============  从 Kubernetes Pod 自动解析 GEMM/FFT  ==============
# =====================================================================

def _load_k8s_clients():
    try:
        from kubernetes import client, config
    except Exception:
        raise HTTPException(status_code=500, detail="kubernetes client is not installed in image")
    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to load k8s config: {e}")
    v1 = client.CoreV1Api()
    batch = client.BatchV1Api()
    return client, v1, batch

def _get_container_from_pod(v1, namespace: str, pod: str, container: Optional[str]):
    p = v1.read_namespaced_pod(pod, namespace)
    if not p or not p.spec or not p.spec.containers:
        raise HTTPException(status_code=404, detail="pod or containers not found")
    if container:
        for c in p.spec.containers:
            if c.name == container:
                return p, c
        raise HTTPException(status_code=404, detail=f"container '{container}' not found in pod")
    return p, p.spec.containers[0]

def _resolve_env_dict(v1, namespace: str, pod, container) -> Dict[str, str]:
    """解析容器环境变量：env + envFrom(configMapRef)。为安全起见不读取 Secret。"""
    env: Dict[str, str] = {}

    if getattr(container, "env", None):
        for e in container.env:
            if getattr(e, "value", None) is not None:
                env[e.name] = e.value
            else:
                src = getattr(e, "value_from", None)
                cmr = getattr(src, "config_map_key_ref", None) if src else None
                if cmr and cmr.name and cmr.key:
                    try:
                        cm = v1.read_namespaced_config_map(cmr.name, namespace)
                        data = cm.data or {}
                        if cmr.key in data:
                            env[e.name] = data[cmr.key]
                    except Exception:
                        pass

    if getattr(container, "env_from", None):
        for ef in container.env_from:
            cmref = getattr(ef, "config_map_ref", None)
            if cmref and cmref.name:
                try:
                    cm = v1.read_namespaced_config_map(cmref.name, namespace)
                    data = cm.data or {}
                    for k, v in data.items():
                        env[k] = v
                except Exception:
                    pass
    return env

def _collect_args(container) -> List[str]:
    args: List[str] = []
    if getattr(container, "command", None):
        args.extend(container.command)
    if getattr(container, "args", None):
        args.extend(container.args)
    return args

def _read_flag(args: List[str], names: List[str]) -> Optional[str]:
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None

def _get_from_env(env: Dict[str, str], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in env and env[k]:
            return env[k]
    return None

def _parse_dtype(val: Optional[str]) -> DType:
    if not val: return "float32"
    s = str(val).lower()
    if s in ("fp16","float16","f16"): return "float16"
    if s in ("fp32","float32","f32"): return "float32"
    if s in ("fp64","float64","f64"): return "float64"
    if s in ("int8","i8"): return "int8"
    return "float32"

def _extract_gemm_from_container(v1, namespace: str, pod, container) -> GemmSpec:
    args = _collect_args(container)
    env = _resolve_env_dict(v1, namespace, pod, container)

    m = _read_flag(args, ["--m","-m"]) or _get_from_env(env, ["M","GEMM_M"])
    n = _read_flag(args, ["--n","-n"]) or _get_from_env(env, ["N","GEMM_N"])
    k = _read_flag(args, ["--k","-k"]) or _get_from_env(env, ["K","GEMM_K"])
    dt = _read_flag(args, ["--dtype"])   or _get_from_env(env, ["DTYPE","GEMM_DTYPE"])

    if not (m and n and k):
        raise HTTPException(status_code=400, detail="cannot infer GEMM m/n/k from args/env/configmap")

    try:
        return GemmSpec(m=int(m), n=int(n), k=int(k), dtype=_parse_dtype(dt))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid GEMM m/n/k/dtype format")

def _extract_fft_from_container(v1, namespace: str, pod, container) -> FFT1DSpec:
    args = _collect_args(container)
    env = _resolve_env_dict(v1, namespace, pod, container)

    n  = _read_flag(args, ["--n"]) or _get_from_env(env, ["N","FFT_N"])
    bt = _read_flag(args, ["--batches"]) or _get_from_env(env, ["BATCHES","FFT_BATCHES"])
    dt = _read_flag(args, ["--dtype"])   or _get_from_env(env, ["DTYPE","FFT_DTYPE"])

    if not n:
        raise HTTPException(status_code=400, detail="cannot infer FFT n from args/env/configmap")

    try:
        batches = int(bt) if bt else 1
        return FFT1DSpec(n=int(n), batches=batches, dtype=_parse_dtype(dt))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid FFT n/batches/dtype format")

@app.post("/predict/gemm/from_pod", response_model=PredictResponse)
def predict_gemm_from_pod(
    namespace: str = Form("default"),
    pod: str = Form(...),
    time_budget_s: float = Form(...),
    container: Optional[str] = Form(None),
):
    client, v1, _ = _load_k8s_clients()
    p, c = _get_container_from_pod(v1, namespace, pod, container)
    spec = _extract_gemm_from_container(v1, namespace, p, c)
    fl, by, t = gemm_cost(spec)
    kernel = Kernel(
        type="gemm",
        params=spec.model_dump(),
        flops=fl, bytes=by, t_lower_bound_s=t
    )
    return decide_and_pack([kernel], time_budget_s)

@app.post("/predict/fft1d/from_pod", response_model=PredictResponse)
def predict_fft_from_pod(
    namespace: str = Form("default"),
    pod: str = Form(...),
    time_budget_s: float = Form(...),
    container: Optional[str] = Form(None),
):
    client, v1, _ = _load_k8s_clients()
    p, c = _get_container_from_pod(v1, namespace, pod, container)
    spec = _extract_fft_from_container(v1, namespace, p, c)
    fl, by, t = fft1d_cost(spec)
    kernel = Kernel(
        type="fft1d",
        params=spec.model_dump(),
        flops=fl, bytes=by, t_lower_bound_s=t
    )
    return decide_and_pack([kernel], time_budget_s)


# ========= 通过 K8s Job + LD_PRELOAD 探针从 .c 源码抓 GEMM =========

BLAS_PROBE_IMAGE = os.getenv("BLAS_PROBE_IMAGE", "blas-probe:0.3")
JOB_NAMESPACE = os.getenv("JOB_NAMESPACE", "default")

# 兼容老版 [max] 行
GEMM_LOG_RE = re.compile(
    r"\[max\]\s+\w+\s+M=(\d+)\s+N=(\d+)\s+K=(\d+)\s+FLOPs=(\d+)",
    re.IGNORECASE
)
# 新版 TOTAL/MAX 行
GEMM_TOTAL_LOG_RE = re.compile(r"\[probe\].*TOTAL:.*FLOPs=([0-9.]+).*Bytes=([0-9.]+)", re.IGNORECASE)
GEMM_MAX_LOG_RE   = re.compile(r"\[probe\].*MAX:.*M=(\d+)\s+N=(\d+)\s+K=(\d+).*FLOPs=([0-9.]+).*Bytes=([0-9.]+)", re.IGNORECASE)

def _k8s_clients():
    try:
        from kubernetes import client, config
    except Exception:
        raise HTTPException(status_code=500, detail="kubernetes client not installed")
    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"load kube config failed: {e}")
    return client, client.CoreV1Api(), client.BatchV1Api()

def _await_job_done(v1, batch, namespace: str, job_name: str, timeout_s: int = 90) -> str:
    """等待 Job 完成；返回唯一 Pod 名称；失败/超时抛异常。"""
    t0 = time.time()
    pod_name = None
    while time.time() - t0 < timeout_s:
        pods = v1.list_namespaced_pod(namespace, label_selector=f"job-name={job_name}", limit=1)
        if pods.items:
            pod = pods.items[0]
            pod_name = pod.metadata.name
            if pod.status.phase in ("Succeeded", "Failed"):
                break
        time.sleep(1.0)
    if not pod_name:
        raise HTTPException(status_code=504, detail="job pod not found in time")

    while time.time() - t0 < timeout_s:
        j = batch.read_namespaced_job_status(job_name, namespace)
        s = j.status
        if (s.succeeded and s.succeeded > 0) or (s.failed and s.failed > 0):
            return pod_name
        time.sleep(1.0)
    raise HTTPException(status_code=504, detail="job not completed in time")

def _cleanup_job_and_cm(batch, v1, namespace: str, job_name: str, cm_name: str):
    if os.getenv("KEEP_PROBE_JOB", "0") == "1":
        return
    try:
        from kubernetes import client as k8s_client
        propagation = k8s_client.V1DeleteOptions(propagation_policy="Background")
        batch.delete_namespaced_job(name=job_name, namespace=namespace, body=propagation)
    except Exception:
        pass
    try:
        v1.delete_namespaced_config_map(cm_name, namespace)
    except Exception:
        pass

def _create_job_with_cm_and_run(src_c_text: str, time_limit_s: int = 60, namespace: str = JOB_NAMESPACE) -> tuple[int, int, int]:
    """创建 ConfigMap + Job，编译并 LD_PRELOAD 运行，读日志解析 M/N/K（兼容 TOTAL/MAX/旧版输出）。"""
    if len(src_c_text.encode("utf-8")) > 900 * 1024:
        raise HTTPException(status_code=413, detail="C source too large for ConfigMap (~1MiB limit)")

    client, v1, batch = _k8s_clients()

    uniq = uuid.uuid4().hex[:8]
    cm_name = f"csrc-{uniq}"
    job_name = f"probe-{uniq}"

    # 1) 把源码放进 ConfigMap
    from kubernetes.client import V1ObjectMeta, V1ConfigMap
    cm = V1ConfigMap(metadata=V1ObjectMeta(name=cm_name), data={"user.c": src_c_text})
    v1.create_namespaced_config_map(namespace, cm)

    # 2) 创建 Job：容器内编译 + 运行 + 打印日志
    from kubernetes.client import (
        V1Job, V1JobSpec, V1PodTemplateSpec, V1PodSpec, V1Container,
        V1VolumeMount, V1Volume, V1ConfigMapVolumeSource, V1ObjectMeta
    )

    shell = r"""
set -euo pipefail
cp /code/user.c /work/user.c

echo "[probe-job] === compile phase ==="
INC_OB="/usr/include/x86_64-linux-gnu/openblas-pthread"
LIB_OB="/usr/lib/x86_64-linux-gnu/openblas-pthread"

set +e
PKG_CFLAGS="$(pkg-config --cflags openblas 2>/dev/null)"
PKG_LIBS="$(pkg-config --libs openblas 2>/dev/null)"
set -e

COMPILE_OK=0
if [ -n "${PKG_LIBS:-}" ]; then
  echo "[probe-job][try pkg-config] gcc -O2 $PKG_CFLAGS -o /work/a.out /work/user.c $PKG_LIBS -lm -lpthread -Wl,--no-as-needed"
  if gcc -O2 $PKG_CFLAGS -o /work/a.out /work/user.c $PKG_LIBS -lm -lpthread -Wl,--no-as-needed; then
    COMPILE_OK=1
  fi
fi

if [ $COMPILE_OK -eq 0 ]; then
  echo "[probe-job][try explicit] gcc -O2 -I$INC_OB -L$LIB_OB -o /work/a.out /work/user.c -lopenblas -lm -lpthread -Wl,--no-as-needed"
  if gcc -O2 -I"$INC_OB" -L"$LIB_OB" -o /work/a.out /work/user.c -lopenblas -lm -lpthread -Wl,--no-as-needed; then
    COMPILE_OK=1
  fi
fi

if [ $COMPILE_OK -eq 0 ]; then
  echo "[probe-job][fallback] gcc -O2 -o /work/a.out /work/user.c -lcblas -lblas -lm -lpthread -Wl,--no-as-needed"
  if gcc -O2 -o /work/a.out /work/user.c -lcblas -lblas -lm -lpthread -Wl,--no-as-needed; then
    COMPILE_OK=1
  fi
fi

if [ $COMPILE_OK -eq 0 ]; then
  echo "[probe-job] ERROR: failed to compile with OpenBLAS/CBLAS" >&2
  exit 86
fi

echo "[probe-job] ldd /work/a.out:"
ldd /work/a.out || true

echo "[probe-job] === run phase (LD_PRELOAD) ==="
export LD_PRELOAD=/opt/probe/lib/libblasprobe.so
export PROBE_DRY_RUN=1
export LOG_DIR=/work
/work/a.out || true

[ -f /work/max.log ] && cat /work/max.log || true
"""

    cmd = ["/bin/sh", "-lc", shell]

    container = V1Container(
        name="runner",
        image=BLAS_PROBE_IMAGE,
        command=cmd[:2],
        args=[cmd[2]],
        volume_mounts=[
            V1VolumeMount(name="code", mount_path="/code"),
            V1VolumeMount(name="work", mount_path="/work"),
        ],
    )

    pod_spec = V1PodSpec(
        restart_policy="Never",
        containers=[container],
        volumes=[
            V1Volume(name="code", config_map=V1ConfigMapVolumeSource(name=cm_name, items=[{"key": "user.c", "path": "user.c"}])),
            V1Volume(name="work", empty_dir={}),
        ],
    )

    job = V1Job(
        metadata=V1ObjectMeta(name=job_name),
        spec=V1JobSpec(
            ttl_seconds_after_finished=60,
            backoff_limit=0,
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(labels={"app": "blas-probe", "job-name": job_name}),
                spec=pod_spec,
            ),
        ),
    )

    batch.create_namespaced_job(namespace, job)

    try:
        pod_name = _await_job_done(v1, batch, namespace, job_name, timeout_s=time_limit_s)
        logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
    finally:
        _cleanup_job_and_cm(batch, v1, namespace, job_name, cm_name)

    # 兼容三种日志格式：TOTAL / MAX / 旧版 [max]
    if logs is None:
        logs = ""

    m_total = GEMM_TOTAL_LOG_RE.search(logs)
    m_max = GEMM_MAX_LOG_RE.search(logs)
    m_legacy = GEMM_LOG_RE.search(logs)

    if m_total:
        FLOPs = float(m_total.group(1))
        # 用等效 K 归一化个“近似 GEMM”，只为后续算 FLOPs/Bytes（不影响最终决策）
        K = max(int(round((FLOPs / 2.0) ** (1.0 / 3.0))), 1)
        return K, K, K
    if m_max:
        M, N, K = int(m_max.group(1)), int(m_max.group(2)), int(m_max.group(3))
        return M, N, K
    if m_legacy:
        return int(m_legacy.group(1)), int(m_legacy.group(2)), int(m_legacy.group(3))

    raise HTTPException(status_code=422, detail=f"failed to parse GEMM from logs:\n{logs[:800] if logs else 'NO LOGS'}")

@app.post("/predict/gemm/from_c_upload", response_model=PredictResponse)
async def predict_gemm_from_c_upload(
    time_budget_s: float = Form(...),
    file: UploadFile = File(...),
    namespace: str = Form(JOB_NAMESPACE),
):
    # 读上传 C 源码（FastAPI 处理 multipart 需 python-multipart 支持）
    try:
        src = (await file.read()).decode("utf-8", errors="ignore")
    finally:
        await file.close()

    # 调度临时 Job 获取 M/N/K
    M, N, K = _create_job_with_cm_and_run(src, time_limit_s=90, namespace=namespace)

    spec = GemmSpec(m=M, n=N, k=K, dtype="float32")  # 也可扩展从源码探测 dtype
    fl, by, t = gemm_cost(spec)
    kernel = Kernel(type="gemm", params=spec.model_dump(), flops=fl, bytes=by, t_lower_bound_s=t)

    resp = decide_and_pack([kernel], time_budget_s)
    out = resp.model_dump()
    out["inferred"] = {"M": M, "N": N, "K": K, "source": "LD_PRELOAD via blas-probe job"}
    return out
