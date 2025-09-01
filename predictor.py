#!/usr/bin/env python3
# ~/pred-svc/predictor.py
import json, math, os, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

def env_gflops():  # GFLOPs -> FLOPs/s
    gf = float(os.environ.get("PEAK_FLOPS_GFLOPS", "50.0"))
    return gf * 1e9

def env_gbs():     # GB/s -> B/s
    gbs = float(os.environ.get("PEAK_MEM_BANDWIDTH_GBS", "15.0"))
    return gbs * 1e9

def elem_bytes(dtype:str)->int:
    return {"fp16":2,"float16":2,"bf16":2,"fp32":4,"float32":4,"fp64":8,"float64":8}.get(dtype.lower(),4)

def gemm_cost(m,n,k,dtype):
    P = elem_bytes(dtype)
    flops = 2.0*m*n*k                               # 2MNK （GEMM 经典计数）:contentReference[oaicite:1]{index=1}
    bytes_ = P * (k*(m+n) + 2*m*n)                  # 近似 A,B 读一次 + C 读写一次 :contentReference[oaicite:2]{index=2}
    return flops, bytes_

def fft1d_cost(N,dtype):
    # 采用常见上界 ~5*N*log2(N) （复数 FFT 的经验上界）；bytes 做一个保守多遍读写近似
    P = elem_bytes(dtype)
    flops = 5.0 * N * math.log2(max(N,1))           # :contentReference[oaicite:3]{index=3}
    bytes_ = P * N * max(1, math.log2(max(N,1)))    # 粗近似（多层蝶形访存）
    return flops, bytes_

def ffprobe_json(input_path, timeout=5):
    # 只要 JSON，屏蔽 stderr，避免混入错误信息。:contentReference[oaicite:4]{index=4}
    cmd = [
        "ffprobe","-v","error",
        "-print_format","json",
        "-show_format","-show_streams",
        input_path
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    return json.loads(out.stdout.decode("utf-8"))

def fps_of(stream):
    # 取 avg_frame_rate 或 r_frame_rate（形如 "30000/1001"）
    val = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        num, den = val.split("/")
        num, den = float(num), float(den)
        return 0.0 if den==0 else num/den
    except Exception:
        return 0.0

def transcode_cost(input_path, codec_hint=None):
    meta = ffprobe_json(input_path)                  # 获取分辨率/时长/尺寸等 :contentReference[oaicite:5]{index=5}
    vstreams = [s for s in meta.get("streams",[]) if s.get("codec_type")=="video"]
    if not vstreams:
        raise RuntimeError("no video stream")
    vs = vstreams[0]
    w = int(vs.get("width",0)); h = int(vs.get("height",0))
    fps = fps_of(vs)
    dur = float(meta.get("format",{}).get("duration", 0.0))
    size_bytes = float(meta.get("format",{}).get("size", 0.0))  # 输入字节数

    pixels = w*h*fps*dur
    # 粗略“每像素 ops”系数（可用环境变量再校准）
    weights = {
        "h264":1.0, "avc":1.0,
        "hevc":1.6, "h265":1.6,
        "vp9":1.3, "av1":3.0
    }
    if codec_hint: codec_hint = codec_hint.lower()
    alpha = float(os.environ.get("XCODE_OPS_PER_PIXEL","50"))   # 缺省 50 ops/px，需线下校准
    if codec_hint in weights: alpha *= weights[codec_hint]
    flops = alpha * pixels
    # bytes 取“读输入 + 写输出”近似，这里仅用输入 size 作为下界（可按目标码率再加上输出）
    bytes_ = size_bytes * 1.5
    return flops, bytes_, {"width":w,"height":h,"fps":fps,"duration":dur,"input_bytes":size_bytes,"ops_per_pixel":alpha}

def roofline_time(flops, bytes_):
    peak_f = env_gflops()
    peak_b = env_gbs()
    t_compute = flops / max(peak_f,1e-9)
    t_memory  = bytes_ / max(peak_b,1e-9)
    return max(t_compute, t_memory), t_compute, t_memory

class Handler(BaseHTTPRequestHandler):
    def _json(self, code:int, obj):
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

    def do_POST(self):
        if self.path != "/predict":
            return self._json(404, {"error":"not found"})
        ln = int(self.headers.get("Content-Length","0"))
        body = json.loads(self.rfile.read(ln).decode("utf-8")) if ln>0 else {}
        kernels = []
        total_time = 0.0
        for k in body.get("kernels", []):
            t = k.get("type","").lower()
            if t=="gemm":
                flops, bytes_ = gemm_cost(int(k["m"]), int(k["n"]), int(k["k"]), k.get("dtype","fp32"))
                t_lb, tc, tm = roofline_time(flops, bytes_)
                kernels.append({"type":"gemm","flops":flops,"bytes":bytes_,"t_compute":tc,"t_memory":tm,"t_lower_bound":t_lb})
                total_time += t_lb
            elif t=="fft1d":
                flops, bytes_ = fft1d_cost(int(k["N"]), k.get("dtype","fp32"))
                t_lb, tc, tm = roofline_time(flops, bytes_)
                kernels.append({"type":"fft1d","flops":flops,"bytes":bytes_,"t_compute":tc,"t_memory":tm,"t_lower_bound":t_lb})
                total_time += t_lb
            elif t=="transcode":
                flops, bytes_, extra = transcode_cost(k["input_path"], k.get("codec_hint"))
                t_lb, tc, tm = roofline_time(flops, bytes_)
                e = {"type":"transcode","flops":flops,"bytes":bytes_,"t_compute":tc,"t_memory":tm,"t_lower_bound":t_lb}
                e.update(extra)
                kernels.append(e)
                total_time += t_lb
            else:
                kernels.append({"type":t,"error":"unknown kernel type"})
        self._json(200, {
            "peak_flops_gflops": float(os.environ.get("PEAK_FLOPS_GFLOPS","50.0")),
            "peak_mem_bandwidth_gbs": float(os.environ.get("PEAK_MEM_BANDWIDTH_GBS","15.0")),
            "kernels": kernels,
            "time_lower_bound_sum_s": total_time
        })

if __name__=="__main__":
    port = int(os.environ.get("PORT","8000"))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
