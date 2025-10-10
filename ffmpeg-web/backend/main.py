#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, json, shlex, time, datetime, pathlib, subprocess, tempfile, shutil, sys
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

POPENC = dict(text=True, encoding="utf-8", errors="replace")

# ---------------- Config ----------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAFETY_DEFAULT = 1.25
SAMPLE_SECS_DEFAULT = 10
MID_SAMPLE = True

# ---------------- Resolve FFmpeg Binaries (cross-platform, Windows-friendly) ----------------
def _resolve_ff_bins():
    """
    Resolve full paths for ffmpeg and ffprobe with the following priority:
    1) env FFMPEG_BIN / FFPROBE_BIN
    2) env FFMPEG_DIR (join ffmpeg(.exe) / ffprobe(.exe))
    3) Windows default hinted path: C:\\Users\\10760\\Downloads\\ffmpeg\\bin
    4) shutil.which on PATH
    Also prepend chosen bin dir to PATH (helps DLL resolution on Windows).
    """
    is_win = os.name == "nt"

    # 1) explicit bins
    ffm = os.environ.get("FFMPEG_BIN")
    ffp = os.environ.get("FFPROBE_BIN")
    if ffm and ffp and pathlib.Path(ffm).exists() and pathlib.Path(ffp).exists():
        bin_dir = str(pathlib.Path(ffm).resolve().parent)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return ffm, ffp

    # 2) FFMPEG_DIR
    ffdir = os.environ.get("FFMPEG_DIR")
    if ffdir:
        ffdir_p = pathlib.Path(ffdir)
        cand_ffm = ffdir_p / ("ffmpeg.exe" if is_win else "ffmpeg")
        cand_ffp = ffdir_p / ("ffprobe.exe" if is_win else "ffprobe")
        if cand_ffm.exists() and cand_ffp.exists():
            os.environ["PATH"] = str(ffdir_p) + os.pathsep + os.environ.get("PATH", "")
            return str(cand_ffm), str(cand_ffp)

    # 3) Windows default path (as you provided)
    if is_win:
        default_dir = pathlib.Path(r"C:\Users\10760\Downloads\ffmpeg\bin")
        cand_ffm = default_dir / "ffmpeg.exe"
        cand_ffp = default_dir / "ffprobe.exe"
        if cand_ffm.exists() and cand_ffp.exists():
            os.environ["PATH"] = str(default_dir) + os.pathsep + os.environ.get("PATH", "")
            return str(cand_ffm), str(cand_ffp)

    # 4) which on PATH
    cand_ffm = shutil.which("ffmpeg.exe" if is_win else "ffmpeg")
    cand_ffp = shutil.which("ffprobe.exe" if is_win else "ffprobe")
    return cand_ffm, cand_ffp

FFMPEG, FFPROBE = _resolve_ff_bins()

def _ensure_bins():
    if not FFMPEG or not FFPROBE:
        raise RuntimeError(
            "FFmpeg/ffprobe not found. Set FFMPEG_DIR or FFMPEG_BIN/FFPROBE_BIN, "
            "or add them to PATH. On Windows you can set FFMPEG_DIR=C:\\Users\\10760\\Downloads\\ffmpeg\\bin"
        )

# ---------------- Utils ----------------
def run_cmd_capture(cmd_list):
    p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **POPENC)
    out, err = p.communicate()
    return p.returncode, out, err

def parse_speed_from_ffmpeg(stderr: str) -> Optional[float]:
    speed = None
    for line in stderr.splitlines():
        if "speed=" in line:
            try:
                seg = line.split("speed=")[-1]
                val = seg.split("x")[0].strip()
                speed = float(val)
            except Exception:
                pass
    return speed

def pretty_time(t_seconds: float) -> str:
    t = int(round(t_seconds))
    h = t // 3600; m = (t % 3600) // 60; s = t % 60
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------- Core: ffprobe / ops ----------------
def ffprobe_info(path: str) -> Dict[str, Any]:
    _ensure_bins()
    cmd = [
        FFPROBE,"-v","error","-select_streams","v:0",
        "-show_entries","stream=codec_name,width,height,avg_frame_rate,pix_fmt,bit_rate",
        "-show_entries","format=duration,size,format_name", "-of","json", path
    ]
    code, out, err = run_cmd_capture(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip()}")
    j = json.loads(out)
    v = j["streams"][0]; f = j["format"]
    afr = v.get("avg_frame_rate", "0/1")
    if "/" in afr:
        a,b = afr.split("/")
        fps = (float(a)/float(b)) if float(b) != 0 else 0.0
    else:
        fps = float(afr or 0)
    return {
        "codec": v.get("codec_name",""),
        "pix_fmt": v.get("pix_fmt",""),
        "bit_rate": int(v.get("bit_rate") or 0),
        "w": int(v.get("width") or 0),
        "h": int(v.get("height") or 0),
        "fps": fps,
        "duration": float(f.get("duration") or 0),
        "size": int(f.get("size") or 0),
        "format": f.get("format_name","")
    }

OPS = [
    "1) H.264 CRF Transcode",
    "1) H.265 CRF Transcode (Slow)",
    "1) Remux (No Re-encode)",
    "2) Fast Trim (no re-encode, keyframe aligned)",
    "2) Precise Trim (re-encode)",
    "3) Scale+FPS (CRF quality)",
    "3) Two-pass Target Bitrate",
    "4) Overlay Watermark (PNG at 10,10)",
    "4) Denoise (hqdn3d) + Transcode",
    "5) Loudness Normalize (EBU R128 loudnorm)",
    "6) Soft Subtitles (mov_text)",
    "6) Hard Subtitles (burn-in)",
    "7) Thumbnail Grid (fps=1, 5x4)",
    "7) GIF (fps=10, width=480)",
    "8) HDR->SDR Tonemap (zscale+tonemap)",
    "9) HLS VOD Segments",
    "9) RTMP Live Stream",
    "10) Screen/Device Capture (note)",
]

def build_pipeline_args(op: str, P: Dict[str, str]):
    vf, vcodec, aargs = [], [], []
    crf = P.get("crf") or "23"
    preset = P.get("preset") or "medium"
    tw = P.get("target_w") or ""
    th = P.get("target_h") or ""
    fps_out = P.get("fps_out") or ""
    bitrate_k = P.get("bitrate_k") or ""
    subs_path = P.get("subs") or ""
    hls_time = P.get("hls_time") or "6"

    def scale_clause():
        if tw and th: return f"scale={tw}:{th}"
        if tw and not th: return f"scale={tw}:-2"
        if th and not tw: return f"scale=-2:{th}"
        return ""

    if op == "1) H.264 CRF Transcode":
        sc=scale_clause(); vf += [sc] if sc else []
        if fps_out: vf.append(f"fps={fps_out}")
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "1) H.265 CRF Transcode (Slow)":
        sc=scale_clause(); vf += [sc] if sc else []
        if fps_out: vf.append(f"fps={fps_out}")
        vcodec = ["-c:v","libx265","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "1) Remux (No Re-encode)":
        vcodec = ["-c","copy"]
    elif op == "2) Fast Trim (no re-encode, keyframe aligned)":
        vcodec = ["-c","copy"]
    elif op == "2) Precise Trim (re-encode)":
        sc=scale_clause(); vf += [sc] if sc else []
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "3) Scale+FPS (CRF quality)":
        sc=scale_clause(); vf += [sc] if sc else []
        if fps_out: vf.append(f"fps={fps_out}")
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "3) Two-pass Target Bitrate":
        sc=scale_clause(); vf += [sc] if sc else []
        if fps_out: vf.append(f"fps={fps_out}")
        br = bitrate_k or "4000"
        vcodec = ["-c:v","libx264","-b:v",f"{br}k","-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "4) Overlay Watermark (PNG at 10,10)":
        vf.append("overlay=10:10")
        sc=scale_clause(); vf += [sc] if sc else []
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "4) Denoise (hqdn3d) + Transcode":
        vf.append("hqdn3d"); sc=scale_clause(); vf += [sc] if sc else []
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "5) Loudness Normalize (EBU R128 loudnorm)":
        aargs = ["-af","loudnorm=I=-16:LRA=11:TP=-1.5"]; vcodec=["-c:v","copy"]
    elif op == "6) Soft Subtitles (mov_text)":
        vcodec = ["-c:v","copy","-c:a","copy","-c:s","mov_text"]
    elif op == "6) Hard Subtitles (burn-in)":
        if subs_path: vf.append(f"subtitles={shlex.quote(subs_path)}")
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "7) Thumbnail Grid (fps=1, 5x4)":
        vf.append("fps=1,scale=320:-1,tile=5x4")
    elif op == "7) GIF (fps=10, width=480)":
        vf.append("fps=10,scale=480:-1:flags=lanczos"); vcodec=["-c:v","gif"]
    elif op == "8) HDR->SDR Tonemap (zscale+tonemap)":
        vf.append("zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709:m=bt709:r=tv")
        vcodec=["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "9) HLS VOD Segments":
        vcodec=["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "9) RTMP Live Stream":
        vcodec=["-c:v","libx264","-crf",crf,"-preset",preset]; aargs=["-c:a","aac","-b:a","128k"]
    elif op == "10) Screen/Device Capture (note)":
        vcodec=["-c:v","libx264","-crf",crf,"-preset",preset]

    vf_arg = ["-vf", ",".join(vf)] if vf else []
    return vf_arg + vcodec + aargs

def run_sample(input_path: str, pipeline_args, duration_full: float, sample_secs: int, op_name: str):
    _ensure_bins()
    cmd = [FFMPEG,"-y"]
    if MID_SAMPLE and duration_full > sample_secs * 3 and not op_name.startswith("2) Fast Trim"):
        start = max(0.0, duration_full * 0.2); cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-t", str(sample_secs), "-i", input_path] + pipeline_args + ["-f","null","-","-v","quiet","-stats"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **POPENC)
    _, err = p.communicate()
    sp = parse_speed_from_ffmpeg(err or "")
    return (sp if (sp and sp>0) else 1.0), " ".join(shlex.quote(str(x)) for x in cmd)

def estimate_total_seconds(duration: float, speed: float, safety: float, op_name: str) -> float:
    t = duration / max(speed, 0.1) * safety
    if op_name == "3) Two-pass Target Bitrate":
        t *= 2.0
    return t

def choose_output_path(input_path: str, op: str) -> str:
    stem = pathlib.Path(input_path).stem
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if op == "1) Remux (No Re-encode)":
        ext = ".mp4"
    elif op == "7) GIF (fps=10, width=480)":
        ext = ".gif"
    elif op == "7) Thumbnail Grid (fps=1, 5x4)":
        ext = ".jpg"
    elif op == "9) HLS VOD Segments":
        ext = ".m3u8"
    else:
        ext = ".mp4"
    return str(OUTPUT_DIR / f"{stem}_out_{ts}{ext}")

def build_real_run_cmd(input_path: str, output_path: str, op: str, P: Dict[str,str]):
    _ensure_bins()
    base = [FFMPEG,"-y"]
    if op.startswith("2) Fast Trim") and P.get("ss"): base += ["-ss", P["ss"]]
    if op.startswith("2) Fast Trim") and P.get("to"): base += ["-to", P["to"]]
    inputs = ["-i", input_path]
    if op == "4) Overlay Watermark (PNG at 10,10)":
        if not P.get("overlay"): raise RuntimeError("Overlay PNG required.")
        inputs = ["-i", input_path, "-i", P["overlay"]]
    if op == "6) Soft Subtitles (mov_text)":
        if not P.get("subs"): raise RuntimeError("Subtitles .srt required.")
        inputs = ["-i", input_path, "-i", P["subs"]]
    if op == "6) Hard Subtitles (burn-in)":
        if not P.get("subs"): raise RuntimeError("Subtitles .srt required for hard subtitles.")
        inputs = ["-i", input_path, "-i", P["subs"]]
    after_inputs = []
    if op.startswith("2) Precise Trim"):
        if P.get("ss"): after_inputs += ["-ss", P["ss"]]
        if P.get("to"): after_inputs += ["-to", P["to"]]
    pipeline = build_pipeline_args(op, P)
    if op == "6) Soft Subtitles (mov_text)":
        pipeline = ["-c:v","copy","-c:a","copy","-c:s","mov_text","-map","0","-map","1:0"]
    if op == "7) Thumbnail Grid (fps=1, 5x4)":
        pipeline += ["-frames:v","1"]
    if op == "9) HLS VOD Segments":
        pipeline += ["-hls_time", P.get("hls_time") or "6", "-hls_playlist_type","vod"]
    return base + inputs + after_inputs + pipeline + [output_path]

# ---------------- FastAPI App ----------------
app = FastAPI(title="FFmpeg ETA Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "ffmpeg_path": FFMPEG,
        "ffprobe_path": FFPROBE,
        "ffmpeg_exists": bool(FFMPEG and pathlib.Path(FFMPEG).exists()),
        "ffprobe_exists": bool(FFPROBE and pathlib.Path(FFPROBE).exists()),
        "platform": sys.platform,
    }

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    # 检查文件类型是否为视频文件
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.3g2', '.mpeg', '.mpg'}
    file_extension = pathlib.Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_extension}。只允许视频文件。")

    suffix = pathlib.Path(file.filename).suffix or ".mp4"
    dst = UPLOAD_DIR / (datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + pathlib.Path(file.filename).name)
    with open(dst, "wb") as f:
        f.write(file.file.read())
    return {"file_id": dst.name, "path": str(dst)}

@app.get("/ffprobe")
def api_ffprobe(file_id: str):
    path = str(UPLOAD_DIR / file_id)
    info = ffprobe_info(path)
    return info

# @app.post("/estimate")
# def api_estimate(
#     file_id: str = Form(...),
#     operation: str = Form(...),
#     params_json: str = Form(...),
# ):
#     path = str(UPLOAD_DIR / file_id)
#     P = json.loads(params_json)
#     info = ffprobe_info(path)
#     pipeline = build_pipeline_args(operation, P)
#     speed, sample_cmd = run_sample(path, pipeline, info["duration"], int(P.get("sample_secs", SAMPLE_SECS_DEFAULT)), operation)
#     eta = estimate_total_seconds(info["duration"], speed, float(P.get("safety", SAFETY_DEFAULT)), operation)
#     return {
#         "ffprobe": {"duration": info["duration"]},
#         "sample_speed_x": round(speed, 3),
#         "sample_cmd": sample_cmd,
#         "eta_seconds": eta,
#         "eta_hms": pretty_time(eta)
#     }


def _ensure_encode_pipeline(op: str, P: dict, pipeline: list) -> list:
    """
    若 pipeline 缺少编码参数，则按 P 或默认值补齐，使采样时真正执行“解码+重编码”。
    仅在纯 Trim 时放行空管线（估时另行处理）。
    """
    if op.startswith("2) Fast Trim") or op.startswith("2) Precise Trim"):
        return pipeline

    has_v = any(k in pipeline for k in ("-c:v", "-vn"))
    has_a = any(k in pipeline for k in ("-c:a", "-an"))
    if has_v and has_a:
        return pipeline

    vcodec   = P.get("vcodec", "libx264")
    crf      = str(P.get("crf", 23))
    preset   = P.get("preset", "medium")
    acodec   = P.get("acodec", "aac")
    a_bitrate= str(P.get("audio_bitrate", "128k"))

    fixed = list(pipeline)
    if not has_v:
        fixed += ["-c:v", vcodec, "-crf", crf, "-preset", preset]
    if not has_a:
        fixed += ["-c:a", acodec, "-b:a", a_bitrate]
    return fixed

@app.post("/estimate")
def api_estimate(
    file_id: str = Form(...),
    operation: str = Form(...),
    params_json: str = Form(...),
):
    path = str(UPLOAD_DIR / file_id)
    P = json.loads(params_json)

    info = ffprobe_info(path)

    # 1) 与 GUI 一致：构造 trim 前缀（采样仍保持 -t 窗口）
    trim_prefix = []
    if operation.startswith("2) Fast Trim") or operation.startswith("2) Precise Trim"):
        if P.get("ss"): trim_prefix += ["-ss", P["ss"]]
        if P.get("to"): trim_prefix += ["-to", P["to"]]

    # 2) 管线参数，并强制补齐编码项
    pipeline = build_pipeline_args(operation, P)
    pipeline = _ensure_encode_pipeline(operation, P, pipeline)

    # 3) 采样命令片段（用于日志/回显；实际执行仍用 run_sample）
    sample_secs = int(P.get("sample_secs", SAMPLE_SECS_DEFAULT))
    if trim_prefix:
        sample_cmd_base = ["ffmpeg"] + trim_prefix + ["-t", str(sample_secs)]
    else:
        sample_cmd_base = ["ffmpeg", "-t", str(sample_secs)]
    sample_cmd_base += ["-i", path]

    if operation == "4) Overlay Watermark (PNG at 10,10)" and P.get("overlay"):
        sample_cmd_base = sample_cmd_base[:-3] + ["-i", P["overlay"]] + sample_cmd_base[-3:]
    if operation == "6) Soft Subtitles (mov_text)" and P.get("subs"):
        sample_cmd_base = sample_cmd_base[:-3] + ["-i", P["subs"]] + sample_cmd_base[-3:]
    if operation == "6) Hard Subtitles (burn-in)" and P.get("subs"):
        sample_cmd_base = sample_cmd_base[:-3] + ["-i", P["subs"]] + sample_cmd_base[-3:]

    full_cmd_preview = sample_cmd_base + pipeline + ["-f", "null", "-", "-v", "quiet", "-stats"]
    sample_cmd_snippet = " ".join(shlex.quote(x) for x in full_cmd_preview[:20]) + \
                         (" ..." if len(full_cmd_preview) > 20 else "")

    # 4) 实际执行采样（Windows 版 run_sample 返回 (speed, cmd_str)）
    speed, sample_cmd = run_sample(
        path, pipeline, info["duration"], sample_secs, operation
    )

    # 5) 估时（如需，可把 IO 惩罚做成可选参数再乘上）
    safety = float(P.get("safety", SAFETY_DEFAULT))

    # 对于剪辑操作，需要计算实际的剪辑时长
    actual_duration = info["duration"]
    if operation.startswith("2) Fast Trim") or operation.startswith("2) Precise Trim"):
        ss = P.get("ss")
        to = P.get("to")
        if ss and to:
            # 解析时间格式 (HH:MM:SS 或 MM:SS 或 SS)
            try:
                if ':' in ss:
                    if ss.count(':') == 2:  # HH:MM:SS
                        ss_seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], ss.split(':')))
                    else:  # MM:SS
                        ss_seconds = sum(x * float(t) for x, t in zip([60, 1], ss.split(':')))
                else:
                    ss_seconds = float(ss)

                if ':' in to:
                    if to.count(':') == 2:  # HH:MM:SS
                        to_seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], to.split(':')))
                    else:  # MM:SS
                        to_seconds = sum(x * float(t) for x, t in zip([60, 1], to.split(':')))
                else:
                    to_seconds = float(to)

                # 计算剪辑的实际时长
                if to_seconds > ss_seconds:
                    actual_duration = to_seconds - ss_seconds
                else:
                    actual_duration = info["duration"] - ss_seconds  # 如果结束时间无效，只剪辑从开始时间到结尾
            except (ValueError, IndexError):
                # 如果时间格式解析失败，使用整个视频时长
                actual_duration = info["duration"]

    eta = estimate_total_seconds(actual_duration, speed, safety, operation)

    return {
        "ffprobe": info,
        # UI 需要这个字段名
        "sample_cmd": sample_cmd,
        # 也返回片段，便于界面展示
        "sample_cmd_snippet": sample_cmd_snippet,
        "sample_speed_x": round(speed, 3),
        "eta_seconds": eta,
        "eta_hms": pretty_time(eta),
        "actual_duration": actual_duration,  # 添加实际剪辑时长
    }

@app.post("/run_local")
def api_run_local(
    file_id: str = Form(...),
    operation: str = Form(...),
    params_json: str = Form(...),
    last_eta_seconds: float = Form(0.0)
):
    path = str(UPLOAD_DIR / file_id)
    P = json.loads(params_json)

    if operation == "9) RTMP Live Stream" and not P.get("rtmp_url"):
        return JSONResponse(status_code=400, content={"error":"RTMP URL required."})

    out_path = choose_output_path(path, operation)
    if operation == "9) HLS VOD Segments":
        folder = pathlib.Path(out_path).with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        out_path = str(folder / "index.m3u8")

    start_ts = now_str(); t0 = time.time()
    cmd = build_real_run_cmd(path, out_path, operation, P)
    if operation == "9) RTMP Live Stream":
        cmd = [FFMPEG,"-re","-i", path] + build_pipeline_args(operation, P) + ["-f","flv", P["rtmp_url"]]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **POPENC)
    out, err = proc.communicate()
    code = proc.returncode

    t1 = time.time(); finish_ts = now_str()
    elapsed = t1 - t0

    cmp_block = None
    if last_eta_seconds and last_eta_seconds > 0:
        diff = elapsed - last_eta_seconds
        pct = (diff / last_eta_seconds) * 100.0
        cmp_block = {
            "eta_seconds": last_eta_seconds,
            "eta_hms": pretty_time(last_eta_seconds),
            "actual_seconds": elapsed,
            "actual_hms": pretty_time(elapsed),
            "diff_seconds": diff,
            "diff_percent": pct
        }
    err = err or ""
    stderr_tail = "\n".join(err.splitlines()[-80:])  # 多拿一些尾部日志更有用
    
    return {
        "start": start_ts,
        "finish": finish_ts,
        "elapsed_seconds": elapsed,
        "elapsed_hms": pretty_time(elapsed),
        "cmd": " ".join(shlex.quote(str(x)) for x in cmd),
        "rc": code,
        "stderr_tail": "\n".join(err.splitlines()[-40:]),
        "output_path": out_path if code == 0 else None,
        "comparison": cmp_block
    }

@app.post("/run_local_stream")
def api_run_local_stream(
    file_id: str = Form(...),
    operation: str = Form(...),
    params_json: str = Form(...),
    last_eta_seconds: float = Form(0.0)
):
    """
    流式返回 ffmpeg 运行日志（实时）。最后会输出一段以 '=== SUMMARY === ' 开头的 JSON 汇总行。
    前端按字节流接收并直接显示。
    """
    path = str(UPLOAD_DIR / file_id)
    P = json.loads(params_json)

    if operation == "9) RTMP Live Stream" and not P.get("rtmp_url"):
        return JSONResponse(status_code=400, content={"error":"RTMP URL required."})

    out_path = choose_output_path(path, operation)
    if operation == "9) HLS VOD Segments":
        folder = pathlib.Path(out_path).with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        out_path = str(folder / "index.m3u8")

    cmd = build_real_run_cmd(path, out_path, operation, P)
    if operation == "9) RTMP Live Stream":
        cmd = [FFMPEG, "-re", "-i", path] + build_pipeline_args(operation, P) + ["-f","flv", P["rtmp_url"]]

    def line_stream():
        start_ts = now_str()
        t0 = time.time()
        # 头部信息
        yield f"=== Run Locally (stream) ===\n"
        yield f"Start: {start_ts}\n"
        yield f"Cmd: {' '.join(shlex.quote(str(x)) for x in cmd)}\n"
        yield "-"*60 + "\n"

        # 运行 ffmpeg 并逐行读取 stderr
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # 行缓冲
                **POPENC,    # text=True, encoding='utf-8', errors='replace'
            )
        except Exception as e:
            yield f"[spawn error] {e}\n"
            # 生成一个汇总行，方便前端结束
            summary = {
                "rc": -1,
                "start": start_ts,
                "finish": now_str(),
                "elapsed_seconds": 0.0,
                "output_path": None,
                "comparison": None,
                "error": str(e),
            }
            yield f"=== SUMMARY === {json.dumps(summary)}\n"
            return

        # ffmpeg 大部分日志走 stderr
        if proc.stderr is not None:
            for line in proc.stderr:
                # 实时吐给前端
                yield line if line.endswith("\n") else (line + "\n")

        # 等待结束
        rc = proc.wait()
        finish_ts = now_str()
        elapsed = time.time() - t0

        # 估时对比
        cmp_block = None
        if last_eta_seconds and last_eta_seconds > 0:
            diff = elapsed - float(last_eta_seconds)
            pct = (diff / float(last_eta_seconds)) * 100.0
            cmp_block = {
                "eta_seconds": float(last_eta_seconds),
                "eta_hms": pretty_time(float(last_eta_seconds)),
                "actual_seconds": elapsed,
                "actual_hms": pretty_time(elapsed),
                "diff_seconds": diff,
                "diff_percent": pct
            }

        # 尾部与汇总
        yield "-"*60 + "\n"
        yield f"Finish: {finish_ts}\n"
        yield f"Elapsed: {pretty_time(elapsed)} ({elapsed:.1f} s)\n"
        if rc == 0:
            yield f"Output: {out_path}\n"
        # 最后一行给一个结构化 JSON，方便前端知道结束与结果
        summary = {
            "rc": rc,
            "start": start_ts,
            "finish": finish_ts,
            "elapsed_seconds": elapsed,
            "output_path": out_path if rc == 0 else None,
            "comparison": cmp_block
        }
        yield f"=== SUMMARY === {json.dumps(summary)}\n"

    # 用 StreamingResponse 持续推送文本
    return StreamingResponse(line_stream(), media_type="text/plain; charset=utf-8")

@app.post("/run_remote_stub")
def api_run_remote_stub(
    file_id: str = Form(...),
    operation: str = Form(...),
    params_json: str = Form(...),
    last_eta_seconds: float = Form(0.0)
):
    P = json.loads(params_json)
    remote_input = "/path/on/remote/input.mp4"
    remote_output = "/path/on/remote/output.mp4"
    try:
        preview_cmd = build_real_run_cmd(remote_input, remote_output, operation, P)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    plan = {
        "upload": [
            "scp <LOCAL_INPUT> REMOTE:REMOTE_DIR/input.mp4",
        ],
        "run": "ssh REMOTE 'cd REMOTE_DIR && " + " ".join(shlex.quote(str(x)) for x in preview_cmd) + "'",
        "download": "scp REMOTE:REMOTE_DIR/output.* <LOCAL_DIR>/"
    }
    if operation == "4) Overlay Watermark (PNG at 10,10)" and P.get("overlay"):
        plan["upload"].append("scp <overlay.png> REMOTE:REMOTE_DIR/overlay.png")
    if operation.startswith("6)") and P.get("subs"):
        plan["upload"].append("scp <subs.srt> REMOTE:REMOTE_DIR/subs.srt")

    ref = None
    if last_eta_seconds and last_eta_seconds > 0:
        ref = {"eta_seconds": last_eta_seconds, "eta_hms": pretty_time(last_eta_seconds)}

    return {"plan": plan, "reference_eta": ref, "note": "Fill REMOTE / REMOTE_DIR later."}

@app.post("/clear_uploads")
def clear_uploads():
    """
    清空上传文件夹中的所有文件
    """
    try:
        deleted_count = 0
        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1

        return {
            "success": True,
            "message": f"已删除 {deleted_count} 个上传文件",
            "deleted_count": deleted_count
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"清空上传文件夹失败: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
