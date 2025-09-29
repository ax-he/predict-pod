#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import shlex
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# >>> NEW: extra imports (append-only)
import time
import datetime
import pathlib
# <<< NEW

# ---------- Defaults ----------
SAFETY = 1.25
SAMPLE_SECS = 10
MID_SAMPLE = True

# ---------- Utilities ----------
def run_cmd_capture(cmd_list):
    try:
        p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate()
        return p.returncode, out, err
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd_list[0]}"

def which(binname):
    return subprocess.call(["which", binname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def parse_speed_from_ffmpeg(stderr_lines):
    speed = None
    for line in stderr_lines.splitlines():
        if "speed=" in line:
            try:
                seg = line.split("speed=")[-1]
                val = seg.split("x")[0].strip()
                speed = float(val)
            except Exception:
                pass
    return speed

def pretty_time(t_seconds):
    t = int(round(t_seconds))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

# >>> NEW: helper to print wall-clock timestamps
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# <<< NEW

# ---------- ffprobe / sampling / estimation ----------
def ffprobe_info(path):
    cmd = [
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=codec_name,width,height,avg_frame_rate,pix_fmt,bit_rate",
        "-show_entries","format=duration,size,format_name",
        "-of","json", path
    ]
    code, out, err = run_cmd_capture(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip()}")
    j = json.loads(out)
    v = j["streams"][0]
    f = j["format"]
    afr = v.get("avg_frame_rate", "0/1")
    if "/" in afr:
        a, b = afr.split("/")
        fps = (float(a) / float(b)) if float(b) != 0 else 0.0
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

# ----- Operation catalog (10 groups, multiple concrete ops) -----
OPS = [
    # 1) Transcode / Remux
    "1) H.264 CRF Transcode",
    "1) H.265 CRF Transcode (Slow)",
    "1) Remux (No Re-encode)",

    # 2) Cut / Trim
    "2) Fast Trim (no re-encode, keyframe aligned)",
    "2) Precise Trim (re-encode)",

    # 3) Scale / FPS / Bitrate
    "3) Scale+FPS (CRF quality)",
    "3) Two-pass Target Bitrate",

    # 4) Video filters
    "4) Overlay Watermark (PNG at 10,10)",
    "4) Denoise (hqdn3d) + Transcode",

    # 5) Audio
    "5) Loudness Normalize (EBU R128 loudnorm)",

    # 6) Subtitles
    "6) Soft Subtitles (mov_text)",
    "6) Hard Subtitles (burn-in)",

    # 7) Thumbnails / GIF
    "7) Thumbnail Grid (fps=1, 5x4)",
    "7) GIF (fps=10, width=480)",

    # 8) HDR / Colorspace
    "8) HDR->SDR Tonemap (zscale+tonemap)",

    # 9) Streaming / HLS
    "9) HLS VOD Segments",
    "9) RTMP Live Stream",

    # 10) Capture
    "10) Screen/Device Capture (note)",
]

OP_NOTES = {
    "1) Remux (No Re-encode)": "ETA ≈ I/O bound; sampling speed is less meaningful.",
    "2) Fast Trim (no re-encode, keyframe aligned)": "Uses -ss before -i and -c copy. Requires start/end times.",
    "2) Precise Trim (re-encode)": "Frame-accurate; slower; requires start/end times.",
    "3) Two-pass Target Bitrate": "ETA ≈ 2 × single-pass time (sampling uses single-pass).",
    "7) Thumbnail Grid (fps=1, 5x4)": "Produces a JPEG grid from frames; ETA similar to decode+filter.",
    "7) GIF (fps=10, width=480)": "GIF encoding is CPU-heavy; sampling approximates.",
    "8) HDR->SDR Tonemap (zscale+tonemap)": "Requires zimg support in ffmpeg build.",
    "9) HLS VOD Segments": "We estimate by encoding speed; not actually writing segments during sampling.",
    "9) RTMP Live Stream": "ETA concept doesn’t apply; sampling encodes to null to gauge throughput.",
    "10) Screen/Device Capture (note)": "Needs a capture device. ETA without a file is not meaningful.",
}

def build_pipeline_args(op, P):
    """
    Build ffmpeg args for the selected operation (excluding input/output file).
    P carries UI params (strings), we guard empties.
    """
    vf = []
    aargs = []
    vcodec = []
    # common params
    crf = P.get("crf") or "23"
    preset = P.get("preset") or "medium"
    tw = P.get("target_w") or ""
    th = P.get("target_h") or ""
    fps_out = P.get("fps_out") or ""
    bitrate_k = P.get("bitrate_k") or ""  # for two-pass
    ss = P.get("ss") or ""   # start
    to = P.get("to") or ""   # end
    overlay_png = P.get("overlay") or ""
    subs_path = P.get("subs") or ""
    rtmp_url = P.get("rtmp_url") or ""
    hls_time = P.get("hls_time") or "6"

    # helpers
    def scale_clause():
        if tw and th:
            return f"scale={tw}:{th}"
        elif tw and not th:
            return f"scale={tw}:-2"
        elif th and not tw:
            return f"scale=-2:{th}"
        return ""

    # ----- per operation -----
    if op == "1) H.264 CRF Transcode":
        sc = scale_clause()
        if sc: vf.append(sc)
        if fps_out: vf.append(f"fps={fps_out}")
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "1) H.265 CRF Transcode (Slow)":
        sc = scale_clause()
        if sc: vf.append(sc)
        if fps_out: vf.append(f"fps={fps_out}")
        vcodec = ["-c:v","libx265","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "1) Remux (No Re-encode)":
        vcodec = ["-c","copy"]

    elif op == "2) Fast Trim (no re-encode, keyframe aligned)":
        # trim is handled by runner (placing -ss/-to), here we just copy
        vcodec = ["-c","copy"]

    elif op == "2) Precise Trim (re-encode)":
        sc = scale_clause()
        if sc: vf.append(sc)
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "3) Scale+FPS (CRF quality)":
        sc = scale_clause()
        if sc: vf.append(sc)
        if fps_out: vf.append(f"fps={fps_out}")
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "3) Two-pass Target Bitrate":
        sc = scale_clause()
        if sc: vf.append(sc)
        if fps_out: vf.append(f"fps={fps_out}")
        # For sampling, we will run single-pass libx264 with target bitrate to approximate complexity
        br = bitrate_k or "4000"
        vcodec = ["-c:v","libx264","-b:v",f"{br}k","-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "4) Overlay Watermark (PNG at 10,10)":
        if overlay_png:
            vf.append(f"overlay=10:10")
        sc = scale_clause()
        if sc: vf.append(sc)
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "4) Denoise (hqdn3d) + Transcode":
        vf.append("hqdn3d")
        sc = scale_clause()
        if sc: vf.append(sc)
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "5) Loudness Normalize (EBU R128 loudnorm)":
        aargs = ["-af","loudnorm=I=-16:LRA=11:TP=-1.5","-c:v","copy"]  # keep video

    elif op == "6) Soft Subtitles (mov_text)":
        # sampling to null won't mux subs; but throughput depends on video copy/encode
        vcodec = ["-c","copy"]  # soft subs: typically copy video
        # real run would add: -i subs.srt -c:s mov_text

    elif op == "6) Hard Subtitles (burn-in)":
        if subs_path:
            vf.append(f"subtitles={shlex.quote(subs_path)}")
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "7) Thumbnail Grid (fps=1, 5x4)":
        vf.append("fps=1,scale=320:-1,tile=5x4")
        vcodec = []  # image output in real run; for sampling we send to null

    elif op == "7) GIF (fps=10, width=480)":
        vf.append("fps=10,scale=480:-1:flags=lanczos")
        vcodec = ["-c:v","gif"]  # for sampling we still write to null

    elif op == "8) HDR->SDR Tonemap (zscale+tonemap)":
        vf.append("zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709:m=bt709:r=tv")
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "9) HLS VOD Segments":
        # Sampling approximates encoding cost; real run would include -hls_time etc.
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "9) RTMP Live Stream":
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]
        aargs = ["-c:a","aac","-b:a","128k"]

    elif op == "10) Screen/Device Capture (note)":
        # Not applicable for file-based sampling
        vcodec = ["-c:v","libx264","-crf",crf,"-preset",preset]

    # compose -vf
    vf_arg = []
    if vf:
        vf_arg = ["-vf", ",".join(vf)]
    return vf_arg + vcodec + aargs

def run_sample(input_path, pipeline_args, duration_full, sample_secs=SAMPLE_SECS, op_name=""):
    # Handle special cases: capture / remux / streaming where ETA is odd but we still try to gauge encode speed
    cmd = ["ffmpeg", "-y"]
    # Trim ops: apply -ss/-to for representativeness (if provided)
    # We only add -ss/-to at sampling time for trim modes.
    # (Real execution should repeat identical flags.)
    if MID_SAMPLE and duration_full > sample_secs * 3 and not op_name.startswith("2) Fast Trim"):
        start = max(0.0, duration_full * 0.2)
        cmd += ["-ss", f"{start:.2f}"]

    # Add short sample window
    cmd += ["-t", str(sample_secs), "-i", input_path] + pipeline_args + ["-f", "null", "-", "-v", "quiet", "-stats"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = p.communicate()
    sp = parse_speed_from_ffmpeg(err)
    return sp if sp and sp > 0 else 1.0

def estimate_total_seconds(duration, speed, safety=SAFETY, op_name=""):
    t = duration / max(speed, 0.1) * safety
    # Two-pass correction
    if op_name == "3) Two-pass Target Bitrate":
        t *= 2.0
    return t

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FFmpeg ETA Helper (10 function groups included)")
        self.geometry("900x640")

        # >>> NEW: to store last ETA for comparison
        self.last_eta_s = None
        # <<< NEW

        # Top: input file
        top = ttk.Frame(self); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="Input video:").grid(row=0, column=0, sticky="w")
        self.in_path = tk.StringVar()
        ttk.Entry(top, textvariable=self.in_path, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(top, text="Browse…", command=self.browse_file).grid(row=0, column=2)

        # Operation
        opsf = ttk.LabelFrame(self, text="Operation & Common Parameters")
        opsf.pack(fill="x", padx=12, pady=8)

        self.op_var = tk.StringVar(value=OPS[0])
        ttk.Label(opsf, text="Operation:").grid(row=0, column=0, sticky="e")
        ttk.Combobox(opsf, textvariable=self.op_var, values=OPS, state="readonly", width=38)\
            .grid(row=0, column=1, sticky="w", padx=6, columnspan=3)

        ttk.Label(opsf, text="CRF:").grid(row=1, column=0, sticky="e")
        self.crf_var = tk.StringVar(value="23")
        ttk.Entry(opsf, textvariable=self.crf_var, width=6).grid(row=1, column=1, sticky="w")

        ttk.Label(opsf, text="preset:").grid(row=1, column=2, sticky="e")
        self.preset_var = tk.StringVar(value="medium")
        ttk.Combobox(opsf, textvariable=self.preset_var,
                     values=["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"],
                     state="readonly", width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(opsf, text="Target Width:").grid(row=2, column=0, sticky="e")
        self.tw_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.tw_var, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(opsf, text="Target Height:").grid(row=2, column=2, sticky="e")
        self.th_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.th_var, width=8).grid(row=2, column=3, sticky="w")

        ttk.Label(opsf, text="Output FPS:").grid(row=3, column=0, sticky="e")
        self.fps_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.fps_var, width=8).grid(row=3, column=1, sticky="w")

        ttk.Label(opsf, text="Bitrate (kbps, 2-pass):").grid(row=3, column=2, sticky="e")
        self.br_var = tk.StringVar(value="4000")
        ttk.Entry(opsf, textvariable=self.br_var, width=10).grid(row=3, column=3, sticky="w")

        ttk.Label(opsf, text="Start -ss (e.g. 00:00:10):").grid(row=4, column=0, sticky="e")
        self.ss_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.ss_var, width=12).grid(row=4, column=1, sticky="w")

        ttk.Label(opsf, text="End -to (e.g. 00:00:20):").grid(row=4, column=2, sticky="e")
        self.to_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.to_var, width=12).grid(row=4, column=3, sticky="w")

        ttk.Label(opsf, text="Overlay PNG:").grid(row=5, column=0, sticky="e")
        self.overlay_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.overlay_var, width=30).grid(row=5, column=1, sticky="w")
        ttk.Button(opsf, text="Pick…", command=self.pick_overlay).grid(row=5, column=2, sticky="w")

        ttk.Label(opsf, text="Subtitles (.srt):").grid(row=6, column=0, sticky="e")
        self.subs_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.subs_var, width=30).grid(row=6, column=1, sticky="w")
        ttk.Button(opsf, text="Pick…", command=self.pick_subs).grid(row=6, column=2, sticky="w")

        ttk.Label(opsf, text="RTMP URL:").grid(row=7, column=0, sticky="e")
        self.rtmp_var = tk.StringVar()
        ttk.Entry(opsf, textvariable=self.rtmp_var, width=40).grid(row=7, column=1, columnspan=3, sticky="we")

        ttk.Label(opsf, text="HLS segment time (s):").grid(row=8, column=0, sticky="e")
        self.hls_time_var = tk.StringVar(value="6")
        ttk.Entry(opsf, textvariable=self.hls_time_var, width=6).grid(row=8, column=1, sticky="w")

        # Estimation params
        estf = ttk.LabelFrame(self, text="Estimation Parameters")
        estf.pack(fill="x", padx=12, pady=8)
        ttk.Label(estf, text="Safety factor:").grid(row=0, column=0, sticky="e")
        self.safety_var = tk.StringVar(value=str(SAFETY))
        ttk.Entry(estf, textvariable=self.safety_var, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(estf, text="Sample seconds:").grid(row=0, column=2, sticky="e")
        self.sample_var = tk.StringVar(value=str(SAMPLE_SECS))
        ttk.Entry(estf, textvariable=self.sample_var, width=6).grid(row=0, column=3, sticky="w")

        # Buttons
        btnf = ttk.Frame(self); btnf.pack(fill="x", padx=12, pady=4)
        ttk.Button(btnf, text="(1) Probe (ffprobe)", command=self.on_probe).pack(side="left", padx=4)
        ttk.Button(btnf, text="(2) Sample & Estimate", command=self.on_estimate).pack(side="left", padx=4)
        # >>> NEW: two extra action buttons
        ttk.Button(btnf, text="(3) Run Locally", command=self.on_run_local).pack(side="left", padx=4)
        ttk.Button(btnf, text="(4) Upload & Run (remote)", command=self.on_run_remote_stub).pack(side="left", padx=4)
        # <<< NEW
        ttk.Button(btnf, text="Clear Output", command=lambda: self.txt.delete("1.0","end")).pack(side="left", padx=4)

        # Output
        outf = ttk.LabelFrame(self, text="Output")
        outf.pack(fill="both", expand=True, padx=12, pady=8)
        self.txt = tk.Text(outf, height=16, wrap="word")
        self.txt.pack(fill="both", expand=True)

        # Self-check
        self.after(200, self.self_check)

    # ---- helpers ----
    def log(self, s):
        self.txt.insert("end", s.rstrip()+"\n")
        self.txt.see("end")

    def browse_file(self):
        path = filedialog.askopenfilename(title="Select video file")
        if path:
            self.in_path.set(path)

    def pick_overlay(self):
        path = filedialog.askopenfilename(title="Pick PNG Overlay", filetypes=[("PNG","*.png"),("All files","*.*")])
        if path:
            self.overlay_var.set(path)

    def pick_subs(self):
        path = filedialog.askopenfilename(title="Pick Subtitles (.srt)", filetypes=[("SRT","*.srt"),("All files","*.*")])
        if path:
            self.subs_var.set(path)

    def self_check(self):
        missing = []
        if not which("ffprobe"): missing.append("ffprobe")
        if not which("ffmpeg"):  missing.append("ffmpeg")
        if missing:
            messagebox.showwarning(
                "Missing dependencies",
                f"Not found: {', '.join(missing)}.\nPlease install in WSL:\n  sudo apt install -y ffmpeg"
            )
        else:
            self.log("ffmpeg / ffprobe detected. Ready.")
            self.log("Tip: set SAMPLE_SECS to 20–30s for more stable ETAs on long/complex jobs.")

    def gather_params(self):
        def nz(s): 
            s = (s or "").strip()
            return s if s else ""
        try:
            safety = float(self.safety_var.get())
        except:
            safety = SAFETY
        try:
            sample_secs = int(self.sample_var.get())
        except:
            sample_secs = SAMPLE_SECS
        return {
            "crf": nz(self.crf_var.get()),
            "preset": nz(self.preset_var.get()),
            "target_w": nz(self.tw_var.get()),
            "target_h": nz(self.th_var.get()),
            "fps_out": nz(self.fps_var.get()),
            "bitrate_k": nz(self.br_var.get()),
            "ss": nz(self.ss_var.get()),
            "to": nz(self.to_var.get()),
            "overlay": nz(self.overlay_var.get()),
            "subs": nz(self.subs_var.get()),
            "rtmp_url": nz(self.rtmp_var.get()),
            "hls_time": nz(self.hls_time_var.get()),
            "safety": safety,
            "sample_secs": sample_secs
        }

    # ---- actions ----
    def on_probe(self):
        path = self.in_path.get().strip()
        if not path:
            messagebox.showinfo("Info", "Please select an input video file first.")
            return
        def work():
            try:
                info = ffprobe_info(path)
                self.log("=== ffprobe info ===")
                self.log(json.dumps(info, indent=2, ensure_ascii=False))
            except Exception as e:
                self.log(f"[probe error] {e}")
        threading.Thread(target=work, daemon=True).start()

    def on_estimate(self):
        path = self.in_path.get().strip()
        if not path:
            # For capture/streaming ETA there is no input; but we still require a file for sampling-based ETA.
            messagebox.showinfo("Info", "Please select an input video file first (sampling-based ETA).")
            return

        op = self.op_var.get()
        P = self.gather_params()

        # quick op note
        if op in OP_NOTES:
            self.log(f"[Note] {OP_NOTES[op]}")

        def work():
            try:
                info = ffprobe_info(path)
                self.log("=== ffprobe info (for estimation) ===")
                self.log(json.dumps(info, indent=2, ensure_ascii=False))

                # Trim flags: only used for trim ops in real run; for sampling we keep short -t window.
                trim_prefix = []
                if op.startswith("2) Fast Trim") or op.startswith("2) Precise Trim"):
                    if P["ss"]: trim_prefix += ["-ss", P["ss"]]
                    if P["to"]: trim_prefix += ["-to", P["to"]]

                pipeline = build_pipeline_args(op, P)

                # For overlay/hard-subs sampling, we still must add overlay/subs inputs if needed.
                # To keep the sampling simple, we assume overlay/subs are readable paths.
                sample_cmd = ["ffmpeg", "-t", str(P["sample_secs"])]
                if trim_prefix: sample_cmd = ["ffmpeg"] + trim_prefix + ["-t", str(P["sample_secs"])]
                sample_cmd += ["-i", path]

                if op == "4) Overlay Watermark (PNG at 10,10)" and P["overlay"]:
                    sample_cmd = sample_cmd[:-3] + ["-i", P["overlay"]] + sample_cmd[-3:]
                if op == "6) Soft Subtitles (mov_text)" and P["subs"]:
                    sample_cmd = sample_cmd[:-3] + ["-i", P["subs"]] + sample_cmd[-3:]

                full_cmd = sample_cmd + pipeline + ["-f","null","-","-v","quiet","-stats"]
                self.log("Sample command snippet:")
                self.log("  " + " ".join(shlex.quote(x) for x in full_cmd[:20]) + (" ..." if len(full_cmd) > 20 else ""))

                # Actually run sampling (single pass). For two-pass we’ll multiply by 2 later.
                sp = run_sample(path, pipeline, info["duration"], P["sample_secs"], op_name=op)
                t_total = estimate_total_seconds(info["duration"], sp, P["safety"], op_name=op)

                # >>> NEW: remember ETA for comparison after real run
                self.last_eta_s = t_total
                # <<< NEW

                self.log(f"Sample speed ≈ {sp:.2f}x (real-time multiplier)")
                self.log(f"Video duration ≈ {pretty_time(info['duration'])}")
                self.log(f"Safety factor = {P['safety']}")
                self.log(f"Estimated local total time ≈ {pretty_time(t_total)}  (≈ {t_total:.1f} s)")
                self.log("-"*60)

            except Exception as e:
                self.log(f"[estimate error] {e}")

        threading.Thread(target=work, daemon=True).start()

    # >>> NEW: choose output path by op
    def _choose_output_path(self, input_path, op):
        stem = pathlib.Path(input_path).stem
        parent = pathlib.Path(input_path).parent
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
        out = parent / f"{stem}_out_{ts}{ext}"
        return str(out)
    # <<< NEW

    # >>> NEW: build real ffmpeg command for actual run
    def _build_real_run_cmd(self, input_path, output_path, op, P):
        base = ["ffmpeg","-y"]

        # Fast trim: -ss/-to before -i
        if op.startswith("2) Fast Trim") and P["ss"]:
            base += ["-ss", P["ss"]]
        if op.startswith("2) Fast Trim") and P["to"]:
            base += ["-to", P["to"]]

        inputs = ["-i", input_path]

        # Extra inputs
        if op == "4) Overlay Watermark (PNG at 10,10)":
            if not P["overlay"]:
                raise RuntimeError("Overlay PNG required for overlay operation.")
            inputs = ["-i", input_path, "-i", P["overlay"]]
        if op == "6) Soft Subtitles (mov_text)":
            if not P["subs"]:
                raise RuntimeError("Subtitles .srt required for soft subtitles.")
            inputs = ["-i", input_path, "-i", P["subs"]]

        after_inputs = []
        if op.startswith("2) Precise Trim"):
            if P["ss"]: after_inputs += ["-ss", P["ss"]]
            if P["to"]: after_inputs += ["-to", P["to"]]

        pipeline = build_pipeline_args(op, P)

        # Special mapping
        if op == "6) Soft Subtitles (mov_text)":
            pipeline = ["-c:v","copy","-c:a","copy","-c:s","mov_text","-map","0","-map","1:0"]

        if op == "7) Thumbnail Grid (fps=1, 5x4)":
            pipeline += ["-frames:v","1"]

        if op == "9) HLS VOD Segments":
            pipeline += ["-hls_time", P["hls_time"] or "6", "-hls_playlist_type","vod"]

        cmd = base + inputs + after_inputs + pipeline + [output_path]
        return cmd
    # <<< NEW

    # >>> NEW: run locally and compare with ETA
    def on_run_local(self):
        input_path = self.in_path.get().strip()
        if not input_path:
            return messagebox.showinfo("Info","Please select an input video file first.")
        op = self.op_var.get()
        P = self.gather_params()

        if op == "9) RTMP Live Stream" and not P["rtmp_url"]:
            return messagebox.showinfo("Info","Please fill RTMP URL before running live stream.")
        if op.startswith("10) "):
            return messagebox.showinfo("Info","Screen/Device capture is not supported in this file-based runner.")

        out_path = self._choose_output_path(input_path, op)
        # For HLS create folder
        if op == "9) HLS VOD Segments":
            folder = pathlib.Path(out_path).with_suffix("")
            folder.mkdir(parents=True, exist_ok=True)
            out_path = str(folder / "index.m3u8")

        def work():
            try:
                self.log(f"=== Run Locally: {op} ===")
                self.log(f"Start time: {now_str()}")
                t0 = time.time()

                cmd = self._build_real_run_cmd(input_path, out_path, op, P)
                if op == "9) RTMP Live Stream":
                    # stream to RTMP
                    cmd = ["ffmpeg","-re","-i", input_path] + build_pipeline_args(op, P) + ["-f","flv", P["rtmp_url"]]

                self.log("Command:")
                self.log("  " + " ".join(shlex.quote(x) for x in cmd[:40]) + (" ..." if len(cmd) > 40 else ""))

                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                while True:
                    line = proc.stderr.readline()
                    if not line:
                        break
                    if "time=" in line or "speed=" in line:
                        self.log(line.strip())
                code = proc.wait()

                t1 = time.time()
                elapsed = t1 - t0
                if code != 0:
                    self.log(f"[run error] ffmpeg exited with code {code}")
                else:
                    self.log(f"Output file: {out_path}")

                self.log(f"Finish time: {now_str()}")
                self.log(f"Elapsed: {pretty_time(elapsed)}  (≈ {elapsed:.1f} s)")

                if self.last_eta_s is not None:
                    diff = elapsed - self.last_eta_s
                    pct = (diff / self.last_eta_s) * 100.0
                    self.log(f"ETA comparison: estimated {pretty_time(self.last_eta_s)}  → actual {pretty_time(elapsed)}  ({diff:+.1f}s, {pct:+.1f}%)")

                self.log("-"*60)
            except Exception as e:
                self.log(f"[run error] {e}")

        threading.Thread(target=work, daemon=True).start()
    # <<< NEW

    # >>> NEW: remote stub (prints SCP/SSH plan only)
    def on_run_remote_stub(self):
        input_path = self.in_path.get().strip()
        if not input_path:
            return messagebox.showinfo("Info","Please select an input video file first.")
        op = self.op_var.get()
        P = self.gather_params()

        # Configuration (fill in later)
        REMOTE = ""      # e.g., "user@server"
        REMOTE_DIR = ""  # e.g., "/tmp/ffjobs"

        def work():
            try:
                self.log("=== Upload & Run (remote) — STUB ===")
                self.log(f"Plan start: {now_str()}")
                # Build a remote-side command preview
                try:
                    preview_cmd = self._build_real_run_cmd("/path/on/remote/input.mp4",
                                                           "/path/on/remote/output.mp4", op, P)
                except Exception as e:
                    self.log(f"[plan error] {e}")
                    return

                self.log("Remote server not configured yet. Typical steps when ready:")
                self.log("  1) Upload input:")
                self.log("     scp <LOCAL_INPUT>  REMOTE:REMOTE_DIR/input.mp4")
                if op == "4) Overlay Watermark (PNG at 10,10)" and P['overlay']:
                    self.log("     scp <overlay.png> REMOTE:REMOTE_DIR/overlay.png")
                if op.startswith("6)") and P['subs']:
                    self.log("     scp <subs.srt>    REMOTE:REMOTE_DIR/subs.srt")
                self.log("  2) Execute remotely:")
                self.log("     ssh REMOTE 'cd REMOTE_DIR && " +
                         " ".join(shlex.quote(x) for x in preview_cmd) + "'")
                self.log("  3) Download result:")
                self.log("     scp REMOTE:REMOTE_DIR/output.*  <LOCAL_DIR>/")
                self.log("Fill REMOTE / REMOTE_DIR later.")

                self.log(f"Plan finish: {now_str()}")
                if self.last_eta_s is not None:
                    self.log(f"[Reference] Last local ETA: {pretty_time(self.last_eta_s)}")
                self.log("-"*60)
            except Exception as e:
                self.log(f"[remote plan error] {e}")

        threading.Thread(target=work, daemon=True).start()
    # <<< NEW

if __name__ == "__main__":
    App().mainloop()
