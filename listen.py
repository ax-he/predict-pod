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


# =====================================================================
# ===============  新增：从 Kubernetes Pod 自动解析 GEMM/FFT  ==========
# =====================================================================

# 懒加载 K8s 客户端（容器里需 pip 安装 kubernetes；RBAC 需 get/list pod、configmaps）
def _load_k8s_clients():
    try:
        from kubernetes import client, config
    except Exception:
        raise HTTPException(status_code=500, detail="kubernetes client is not installed in image")
    try:
        # in-cluster 优先（ServiceAccount 注入的方式）
        config.load_incluster_config()
    except Exception:
        # 回退到本地 kubeconfig（便于本地调试）
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

    # 直接 env=...
    if getattr(container, "env", None):
        for e in container.env:
            if getattr(e, "value", None) is not None:
                env[e.name] = e.value
            else:
                # configMapKeyRef
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
                # Secret 不读取

    # envFrom(configMapRef)
    if getattr(container, "env_from", None):
        for ef in container.env_from:
            cmref = getattr(ef, "config_map_ref", None)
            if cmref and cmref.name:
                try:
                    cm = v1.read_namespaced_config_map(cmref.name, namespace)
                    data = cm.data or {}
                    # 注意：无 prefix 处理。如有需要可读取 ef.prefix
                    for k, v in data.items():
                        env[k] = v
                except Exception:
                    pass
            # Secret 不读取
    return env

def _collect_args(container) -> List[str]:
    args: List[str] = []
    if getattr(container, "command", None):
        args.extend(container.command)
    if getattr(container, "args", None):
        args.extend(container.args)
    return args

def _read_flag(args: List[str], names: List[str]) -> Optional[str]:
    # 支持 --m 1024 / --m=1024 / -m 1024 / -m=1024
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


# ========= 新增：上传 .c 源码 → 调用 blas-probe Job 抓取 GEMM 参数 =========
import base64, uuid, time, re
from fastapi import BackgroundTasks

BLAS_PROBE_IMAGE = os.getenv("BLAS_PROBE_IMAGE", "blas-probe:0.1")
JOB_NAMESPACE = os.getenv("JOB_NAMESPACE", "default")

GEMM_LOG_RE = re.compile(
    r"\[max\]\s+\w+\s+M=(\d+)\s+N=(\d+)\s+K=(\d+)\s+FLOPs=(\d+)",
    re.IGNORECASE
)

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
    # 等 pod 出来
    while time.time() - t0 < timeout_s:
        pods = v1.list_namespaced_pod(
            namespace,
            label_selector=f"job-name={job_name}",
            limit=1
        )
        if pods.items:
            pod = pods.items[0]
            pod_name = pod.metadata.name
            # 等 pod 进入结束态（Succeeded/Failed）
            if pod.status.phase in ("Succeeded", "Failed"):
                break
        time.sleep(1.0)

    if not pod_name:
        raise HTTPException(status_code=504, detail="job pod not found in time")

    # 再确认 job 状态
    while time.time() - t0 < timeout_s:
        j = batch.read_namespaced_job_status(job_name, namespace)
        s = j.status
        if (s.succeeded and s.succeeded > 0) or (s.failed and s.failed > 0):
            return pod_name
        time.sleep(1.0)

    raise HTTPException(status_code=504, detail="job not completed in time")

def _cleanup_job_and_cm(batch, v1, namespace: str, job_name: str, cm_name: str):
    # 删除 Job（连带其 pods）
    try:
        from kubernetes import client as k8s_client
        propagation = k8s_client.V1DeleteOptions(propagation_policy="Background")
        batch.delete_namespaced_job(name=job_name, namespace=namespace, body=propagation)
    except Exception:
        pass
    # 删除 ConfigMap
    try:
        v1.delete_namespaced_config_map(cm_name, namespace)
    except Exception:
        pass

def _create_job_with_cm_and_run(src_c_text: str, time_limit_s: int = 60, namespace: str = JOB_NAMESPACE) -> tuple[int,int,int]:
    """创建 CM + Job，编译并 LD_PRELOAD 运行，读日志解析 M/N/K。"""
    if len(src_c_text.encode("utf-8")) > 900*1024:
        # ConfigMap 约 1MiB 上限，留点余量（官方提示: 单个对象数据建议 <= 1MiB）
        # https://kubernetes.io/docs/concepts/configuration/configmap/
        raise HTTPException(status_code=413, detail="C source too large for ConfigMap (~1MiB limit)")

    client, v1, batch = _k8s_clients()

    uniq = uuid.uuid4().hex[:8]
    cm_name = f"csrc-{uniq}"
    job_name = f"probe-{uniq}"

    # 1) 创建 ConfigMap
    from kubernetes.client import V1ObjectMeta, V1ConfigMap
    cm = V1ConfigMap(
        metadata=V1ObjectMeta(name=cm_name),
        data={"user.c": src_c_text}
    )
    v1.create_namespaced_config_map(namespace, cm)

    # 2) 创建 Job（用 blas-probe:0.1）
    #    要求镜像里存在：
    #      - /opt/probe/lib/libblasprobe.so  (LD_PRELOAD)
    #      - gcc + OpenBLAS headers & lib (能 gcc ... -lopenblas)
    from kubernetes.client import (
        V1Job, V1JobSpec, V1PodTemplateSpec, V1PodSpec, V1Container,
        V1VolumeMount, V1Volume, V1ConfigMapVolumeSource, V1ObjectMeta
    )

    cmd = [
        "/bin/sh", "-lc",
        # 编译 -> 运行（允许用户程序异常退出，因此尾部加 `|| true`），
        # 然后把 /work/max.log 打出来，便于 read_namespaced_pod_log 获取
        r"""
set -e
cp /code/user.c /work/user.c
# 你也可以在这里替换为 clang，或添加 -I/-L 指向 OpenBLAS 位置
gcc -O2 -o /work/a.out /work/user.c -lopenblas
export LOG_DIR=/work
export LD_PRELOAD=/opt/probe/lib/libblasprobe.so
/work/a.out || true
if [ -f /work/max.log ]; then
  cat /work/max.log
else
  echo "[max] none"
fi
"""
    ]

    container = V1Container(
        name="runner",
        image=BLAS_PROBE_IMAGE,
        command=cmd[:2],
        args=[cmd[2]],
        env=[],
        volume_mounts=[
            V1VolumeMount(name="code", mount_path="/code"),
            V1VolumeMount(name="work", mount_path="/work")
        ]
    )

    pod_spec = V1PodSpec(
        restart_policy="Never",
        containers=[container],
        volumes=[
            V1Volume(
                name="code",
                config_map=V1ConfigMapVolumeSource(name=cm_name, items=[{"key":"user.c","path":"user.c"}])
            ),
            V1Volume(
                name="work",
                empty_dir={}
            )
        ]
    )

    job = V1Job(
        metadata=V1ObjectMeta(name=job_name),
        spec=V1JobSpec(
            ttl_seconds_after_finished=60,
            backoff_limit=0,
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(labels={"app":"blas-probe","job-name":job_name}),
                spec=pod_spec
            )
        )
    )

    batch.create_namespaced_job(namespace, job)

    try:
        pod_name = _await_job_done(v1, batch, namespace, job_name, timeout_s=time_limit_s)
        logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
    finally:
        _cleanup_job_and_cm(batch, v1, namespace, job_name, cm_name)

    m = GEMM_LOG_RE.search(logs or "")
    if not m:
        raise HTTPException(status_code=422, detail=f"failed to parse GEMM params from logs:\n{logs[:800] if logs else 'NO LOGS'}")

    M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return M, N, K

@app.post("/predict/gemm/from_c_upload", response_model=PredictResponse)
async def predict_gemm_from_c_upload(
    time_budget_s: float = Form(...),
    file: UploadFile = File(...),
    namespace: str = Form(JOB_NAMESPACE),
):
    # 读上传 C 源码
    try:
        src = (await file.read()).decode("utf-8", errors="ignore")
    finally:
        await file.close()

    # 调度临时 Job 获取 M/N/K
    M, N, K = _create_job_with_cm_and_run(src, time_limit_s=90, namespace=namespace)

    spec = GemmSpec(m=M, n=N, k=K, dtype="float32")  # 也可扩展从源码中探测 dtype
    fl, by, t = gemm_cost(spec)
    kernel = Kernel(type="gemm", params=spec.model_dump(), flops=fl, bytes=by, t_lower_bound_s=t)

    resp = decide_and_pack([kernel], time_budget_s)
    # 附带抓到的参数，便于排错
    out = resp.model_dump()
    out["inferred"] = {"M": M, "N": N, "K": K, "source": "LD_PRELOAD via blas-probe job"}
    return out
