#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSL2 友好的 perf-uprobe 资源计数 & 预测脚本
- 自动为 libcrypto/libopenblas/libc 添加 uprobes，并抓取实参寄存器字段
- 运行目标程序并记录样本
- 解析 perf script 输出，汇总字节数/浮点量
- 按给定机器参数估算时间（Roofline 粗粒度）

用法：
  sudo ./perf_predict.py -- ./test_app [args...]

可调环境变量：
  CPU_HZ          默认 3.5e9（若能从 /proc/cpuinfo 解析则用解析值）
  PEAK_GFLOPS     默认 50.906
  MEM_BW_BPS      默认 14_530_000_000  (≈14.53 GB/s)
  CPE_SHA256      SHA-256 cycles/byte  默认 12.0  （可按你机器校准）
  CPE_CMAC_AES128 AES-128 CMAC cyc/B   默认 2.0   （AES-NI 场景；无 AES-NI 可设 7~12）
"""
import json, os, re, shlex, subprocess, sys, tempfile

# -------- machine knobs --------
def cpu_hz_guess():
    # 粗略解析 /proc/cpuinfo
    try:
        txt = subprocess.check_output(["bash","-lc","awk -F: '/cpu MHz/ {mhz=$2} END{if(mhz) printf(\"%.0f\\n\", mhz*1e6)}' /proc/cpuinfo"], text=True).strip()
        if txt: return int(float(txt))
    except Exception:
        pass
    return int(float(os.environ.get("CPU_HZ", "3500000000")))

PEAK_GFLOPS = float(os.environ.get("PEAK_GFLOPS", "50.906"))
MEM_BW_BPS  = float(os.environ.get("MEM_BW_BPS", "14530000000"))
CPE_SHA256  = float(os.environ.get("CPE_SHA256",  "12.0"))
CPE_CMAC    = float(os.environ.get("CPE_CMAC_AES128", "2.0"))

CPU_HZ      = cpu_hz_guess()

# -------- perf helpers --------
def which_perf():
    # 优先 /usr/bin/perf；找不到则回退到 /usr/lib/linux-tools/$(uname -r)/perf
    p = shutil.which("perf") if 'shutil' in globals() else None
    if p: return p
    try:
        kver = subprocess.check_output(["uname","-r"], text=True).strip()
        p2 = f"/usr/lib/linux-tools/{kver}/perf"
        if os.path.exists(p2): return p2
    except Exception:
        pass
    return "perf"

import shutil
PERF = which_perf()

def run(cmd, **kw):
    return subprocess.run(cmd, check=False, text=True, **kw)

def out(cmd):
    return subprocess.check_output(cmd, text=True)

def ldd_path(exe, needle):
    # 在 ldd 输出里找以 needle 开头的库路径
    try:
        txt = out(["bash","-lc", f"ldd {shlex.quote(exe)} | awk '/{needle}/ {{print $3; exit}}'"])
        return txt.strip()
    except Exception:
        return ""

def add_probe(lib, spec):
    # idempotent 添加：已存在则忽略
    # spec 例如: "SHA256_Update len=%dx"
    r = run([PERF,"probe","-x",lib,"--add",spec], capture_output=True)
    if r.returncode != 0 and "already exist" not in (r.stderr+r.stdout):
        return False, r.stderr.strip() or r.stdout.strip()
    return True, ""

def del_probe(pattern):
    run([PERF,"probe","--del",pattern], capture_output=True)

def list_funcs(lib):
    r = run([PERF,"probe","-x",lib,"--funcs"], capture_output=True)
    if r.returncode!=0: return ""
    return r.stdout

# -------- main --------
def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        print("用法：sudo ./perf_predict.py -- <command> [args...]", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[2:]

    # 解析目标二进制依赖库路径（尽量用 ldd 找实际映射路径，以减少 Uprobe 命名空间问题）
    exe = shutil.which(cmd[0]) or cmd[0]
    libcrypto  = ldd_path(exe, "libcrypto")  or "/lib/x86_64-linux-gnu/libcrypto.so.3"
    libopenblas= ldd_path(exe, "openblas")   or "/lib/x86_64-linux-gnu/libopenblas.so.0"
    libc       = ldd_path(exe, "libc.so.6")  or "/lib/x86_64-linux-gnu/libc.so.6"

    # 清理我们即将创建的同名探针（忽略错误）
    del_probe("probe_libcrypto:SHA256_Update*")
    del_probe("probe_libcrypto:CMAC_Update*")
    del_probe("probe_libopenblas:cblas_sgemm*")
    del_probe("probe_libopenblas:cblas_dgemm*")
    del_probe("probe_libc:memcpy*")
    del_probe("probe_libc:memmove*")

    # 添加探针（x86-64：len 第3参 %dx；GEMM 的 M,N,K 分别 %cx,%r8,%r9；memcpy/memmove 第3参 %dx）
    ok1, e1 = add_probe(libcrypto,   r"SHA256_Update len=%dx")
    ok2, e2 = add_probe(libcrypto,   r"CMAC_Update len=%dx")
    # OpenBLAS 可能没有链接 CBLAS 符号就会失败 —— 我们允许失败（只是不计 gemm）
    ok3, e3 = add_probe(libopenblas, r"cblas_sgemm M=%cx N=%r8 K=%r9")
    ok4, e4 = add_probe(libopenblas, r"cblas_dgemm M=%cx N=%r8 K=%r9")
    ok5, e5 = add_probe(libc,        r"memcpy n=%dx")
    ok6, e6 = add_probe(libc,        r"memmove n=%dx")

    # 对于 OpenSSL 3 带版本符号的系统，前两条若失败再尝试转义版本名（perf-probe 支持 \@GLIBC_xxx 风格）
    if not ok1:
        add_probe(libcrypto, r"SHA256_Update\@OPENSSL_3.0.0 len=%dx")
    if not ok2:
        add_probe(libcrypto, r"CMAC_Update\@OPENSSL_3.0.0 len=%dx")

    # 运行目标并记录 perf 数据
    perfdata = "perf.data"
    rec = run([PERF,"record","-q","-o",perfdata,
               "-e","probe_libcrypto:SHA256_Update",
               "-e","probe_libcrypto:CMAC_Update",
               "-e","probe_libopenblas:cblas_sgemm",
               "-e","probe_libopenblas:cblas_dgemm",
               "-e","probe_libc:memcpy",
               "-e","probe_libc:memmove",
               "--"] + cmd, capture_output=True)
    if rec.returncode != 0:
        # 记录失败也打印一份 info
        print(rec.stderr or rec.stdout, file=sys.stderr)

    # 解析脚本输出
    try:
        script = out([PERF,"script","-i",perfdata,"-F","comm,pid,time,event,trace"])
    except Exception as ex:
        print(f"perf script 失败: {ex}", file=sys.stderr)
        script = ""

    counts = {
        "memcpy_bytes": 0,
        "io_bytes": 0,  # 本脚本不抓 sys_read/write，保留字段
        "sha256_bytes": 0,
        "cmac_bytes_aligned": 0,
        "gemm_flops": 0
    }

    # 行解析
    for line in script.splitlines():
        if "probe_libcrypto:SHA256_Update" in line:
            m = re.search(r"\blen=(\d+)", line)
            if m: counts["sha256_bytes"] += int(m.group(1))
        elif "probe_libcrypto:CMAC_Update" in line:
            m = re.search(r"\blen=(\d+)", line)
            if m:
                # 16B 分组向上取整（CMAC/AES-128 的块大小）
                n = int(m.group(1))
                blocks = (n + 15)//16
                counts["cmac_bytes_aligned"] += blocks*16
        elif "probe_libopenblas:cblas_sgemm" in line or "probe_libopenblas:cblas_dgemm" in line:
            M = re.search(r"\bM=(\d+)", line)
            N = re.search(r"\bN=(\d+)", line)
            K = re.search(r"\bK=(\d+)", line)
            if M and N and K:
                m = int(M.group(1)); n = int(N.group(1)); k = int(K.group(1))
                counts["gemm_flops"] += 2*m*n*k
        elif "probe_libc:memcpy" in line or "probe_libc:memmove" in line:
            m = re.search(r"\bn=(\d+)", line)
            if m: counts["memcpy_bytes"] += int(m.group(1))
        # （如需统计 read/write 可加: -e syscalls:sys_enter_read/write 并解析其 trace 字段的 count）

    # 时间估算（秒）
    time = {
        "memcpy": counts["memcpy_bytes"] / MEM_BW_BPS if MEM_BW_BPS>0 else 0.0,
        "io":     counts["io_bytes"]     / MEM_BW_BPS if MEM_BW_BPS>0 else 0.0,
        "sha256": (counts["sha256_bytes"] * CPE_SHA256) / CPU_HZ if CPU_HZ>0 else 0.0,
        "aes128_cmac": (counts["cmac_bytes_aligned"] * CPE_CMAC) / CPU_HZ if CPU_HZ>0 else 0.0,
        "gemm":   (counts["gemm_flops"] / (PEAK_GFLOPS*1e9)) if PEAK_GFLOPS>0 else 0.0,
    }
    time["sum_partial"] = sum(time.values())

    result = {
        "probe": "perf-probe",
        "cpu_hz": CPU_HZ,
        "machine": {
            "peak_gflops": PEAK_GFLOPS,
            "mem_bw_bps":  MEM_BW_BPS
        },
        "counts": counts,
        "time_s": time
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
