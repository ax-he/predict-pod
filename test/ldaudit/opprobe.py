#!/usr/bin/env python3
# opprobe.py — eBPF/BCC 采集 + 估时
from bcc import BPF
import os, sys, json, yaml, subprocess, re, glob

# ---------- utils ----------
def maps_sum(bpf, name):
    m=bpf.get_table(name); s=0
    for _,v in m.items(): s+=int(v.value)
    return s

def ldd_map(exe):
    out = subprocess.check_output(["ldd", exe], text=True, stderr=subprocess.STDOUT)
    m = {}
    # e.g. "libcrypto.so.3 => /lib/x86_64-linux-gnu/libcrypto.so.3 (0x...)"
    mo1 = re.compile(r'^\s*(\S+)\s*=>\s*(/[^ ]+)')
    mo2 = re.compile(r'^\s*(/[^ ]+)')  # 静态路径形式
    for ln in out.splitlines():
        m1 = mo1.search(ln)
        if m1: m[m1.group(1)] = m1.group(2); continue
        m2 = mo2.search(ln)
        if m2:
            p = m2.group(1); k = os.path.basename(p)
            m[k] = p
    return m

def resolve_path(token, exe):
    """把短名（libcrypto/openblas/libc 或 soname）解析成绝对路径"""
    if token and token.startswith("/") and os.path.exists(token):
        return token
    lm = ldd_map(exe)

    # 1) 精确命中
    if token in lm: return lm[token]

    # 2) 常见库前缀优先（避免 libc 命中 libcrypto）
    pref = {
        "libc":      ("libc.so",),
        "libcrypto": ("libcrypto.so",),
        "openblas":  ("libopenblas.so", "openblas"),
    }.get(token, ())
    for k,v in lm.items():
        for pf in pref:
            if k.startswith(pf): return v

    # 3) 再做温和模糊匹配（整词/去版本）
    t = token.lower()
    for k,v in lm.items():
        base = k.split(".so")[0] if ".so" in k else k
        if t in base.lower(): return v

    # 4) 文件系统兜底
    pats = [f"/lib*/**/*{t}*.so*", f"/usr/lib*/**/*{t}*.so*"]
    cand=[]
    for p in pats: cand += glob.glob(p, recursive=True)
    return sorted(cand)[0] if cand else None

# ---------- ELF 解析：把 st_value -> 文件偏移 ----------
_sec_rx = re.compile(r'^\s*\[\s*(\d+)\]\s+\S+\s+\S+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+')
# [Nr] Name Type Address Off Size ...

_sym_rx = re.compile(
    r'^\s*\d+:\s*([0-9a-fA-F]+)\s+\d+\s+\w+\s+\w+\s+\w+\s+(\S+)\s+(\S+)\s*$'
)
# Num: Value Size Type Bind Vis Ndx Name

def read_sections(libpath):
    out = subprocess.check_output(["readelf","-WS",libpath], text=True)
    sec = {}
    for ln in out.splitlines():
        m = _sec_rx.match(ln)
        if m:
            idx = int(m.group(1)); sh_addr = int(m.group(2),16); sh_off = int(m.group(3),16)
            sec[idx] = (sh_addr, sh_off)
    return sec

def read_symbols(libpath):
    out = subprocess.check_output(["readelf","-Ws",libpath], text=True)
    syms = []
    for ln in out.splitlines():
        m = _sym_rx.match(ln)
        if m:
            st_value = int(m.group(1),16)
            ndx      = m.group(2)  # 可能是数字/UND/ABS
            name     = m.group(3)
            if ndx.isdigit():
                syms.append((name, st_value, int(ndx)))
    return syms

def resolve_addr_fileoff(libpath, want):
    """按优先级返回 (file_offset, realname)；优先 @@ 默认版本 → @任意版本 → 裸名"""
    secs = read_sections(libpath)
    syms = read_symbols(libpath)

    def pick(pred):
        for name,st,ndx in syms:
            if pred(name):
                sh_addr, sh_off = secs.get(ndx, (None, None))
                if sh_addr is None: continue
                # 关键换算：file_offset = st_value - sh_addr + sh_offset
                file_off = st - sh_addr + sh_off
                return file_off, name
        return (None, None)

    # 先 @@ 默认版本
    off,name = pick(lambda n: n.startswith(want+"@@"))
    if off is not None: return off,name
    # 再 @ 任意版本
    off,name = pick(lambda n: n.startswith(want+"@"))
    if off is not None: return off,name
    # 最后裸名
    off,name = pick(lambda n: n == want)
    return off,name

# ---------- eBPF 程序 ----------
HEADER = r"""
#include <uapi/linux/ptrace.h>
BPF_HASH(b_memcpy, u32, u64);
BPF_HASH(b_sha,    u32, u64);
BPF_HASH(b_cmacB,  u32, u64);
BPF_HASH(b_io,     u32, u64);
BPF_HASH(b_gemmF,  u32, u64);
"""

def upd(mapname, expr):
    return f"""
  u64 __add = ({expr});
  if (__add) {{
    u32 __k = (u32)bpf_get_current_pid_tgid();
    u64 *p = {mapname}.lookup(&__k);
    if (p) {{ __sync_fetch_and_add(p, __add); }}
    else    {{ {mapname}.update(&__k, &__add); }}
  }}
"""

def build_program(cfg):
    prog = HEADER
    for it in cfg.get("catalog", []):
        if "attach" in it:
            for a in it["attach"]:
                sym=a["sym"]
                fn=f"u_{it['cat']}_{re.sub(r'[^A-Za-z0-9_]+','_',sym)}"
                if it["cat"]=="gemm":
                    M,N,K = a["M"],a["N"],a["K"]
                    prog += f"""
int {fn}(struct pt_regs*ctx){{
  u64 m=(u64)PT_REGS_PARM{M}(ctx), n=(u64)PT_REGS_PARM{N}(ctx), kd=(u64)PT_REGS_PARM{K}(ctx);
  {upd("b_gemmF","2*m*n*kd")}
  return 0;
}}
"""
                else:
                    L=a["len_arg"]
                    val="len"
                    if it["cat"]=="cmac": val="((len + 15) / 16) * 16"
                    prog += f"""
int {fn}(struct pt_regs*ctx){{
  u64 len=(u64)PT_REGS_PARM{L}(ctx);
"""
                    if it["cat"]=="memcpy": prog += upd("b_memcpy","len")
                    if it["cat"]=="sha2":   prog += upd("b_sha","len")
                    if it["cat"]=="cmac":   prog += upd("b_cmacB",val)
                    prog += "  return 0;\n}\n"
    # I/O tracepoints
    if any("io"==it.get("cat") for it in cfg.get("catalog", [])):
        prog += r"""
int tp_r(struct tracepoint__syscalls__sys_enter_read *args){
""" + upd("b_io","args->count") + "  return 0;\n}\n"
        prog += r"""
int tp_w(struct tracepoint__syscalls__sys_enter_write *args){
""" + upd("b_io","args->count") + "  return 0;\n}\n"
    return prog

def attach_all(bpf, pid, cfg, exe):
    for it in cfg.get("catalog", []):
        if "attach" in it:
            for a in it["attach"]:
                tok = a["lib"]; sym=a["sym"]
                fn = f"u_{it['cat']}_{re.sub(r'[^A-Za-z0-9_]+','_',sym)}"

                lib = resolve_path(tok, exe)
                if not lib:
                    print(f"[warn] resolve lib path failed: {tok}", file=sys.stderr)
                    continue

                # 计算“文件内偏移”而不是虚拟地址
                off, realname = resolve_addr_fileoff(lib, sym)
                # 对 OpenBLAS 兜底到 Fortran 名字
                if not off and it["cat"]=="gemm":
                    alt = "dgemm_" if "dgemm" in sym else ("sgemm_" if "sgemm" in sym else None)
                    if alt:
                        off, realname = resolve_addr_fileoff(lib, alt)
                if not off:
                    print(f"[warn] resolve addr failed: {lib}:{sym}", file=sys.stderr)
                    continue
                try:
                    # 注意：addr=文件偏移
                    bpf.attach_uprobe(name=lib, addr=off, fn_name=fn, pid=pid)
                    print(f"[ok] attach {lib}:{realname} file_off=0x{off:x}", file=sys.stderr)
                except Exception as e:
                    print(f"[warn] uprobe attach failed: {lib}:{realname} -> {e}", file=sys.stderr)

        if it.get("tracepoints"):
            if "sys_enter_read"  in it["tracepoints"]:
                bpf.attach_tracepoint(tp="syscalls:sys_enter_read",  fn_name="tp_r")
            if "sys_enter_write" in it["tracepoints"]:
                bpf.attach_tracepoint(tp="syscalls:sys_enter_write", fn_name="tp_w")

# ---------- main ----------
def main():
    if len(sys.argv)<2:
        print("usage: sudo ./opprobe.py <program> [args...]"); sys.exit(1)
    with open("ops.yml") as f: cfg=yaml.safe_load(f)

    peak_gflops=float(cfg["machine"]["peak_gflops"])
    mem_bw_bps =float(cfg["machine"]["mem_bw_gbs"])*1e9
    cpb_sha    =float(cfg["costs"]["cpb_sha256"])
    cpb_aes    =float(cfg["costs"]["cpb_aes128"])
    cpu_hz     =float(os.getenv("PROBE_CPU_HZ","3.5e9"))

    exe_path = os.path.abspath(sys.argv[1])
    proc = subprocess.Popen(sys.argv[1:], env=os.environ.copy())
    pid  = proc.pid

    text = build_program(cfg)
    bpf  = BPF(text=text)
    attach_all(bpf, pid, cfg, exe_path)

    proc.wait()

    memcpyB = maps_sum(bpf,"b_memcpy")
    shaB    = maps_sum(bpf,"b_sha")
    cmacB   = maps_sum(bpf,"b_cmacB")
    ioB     = maps_sum(bpf,"b_io")
    gemmF   = maps_sum(bpf,"b_gemmF")

    t_memcpy = (memcpyB / mem_bw_bps) if mem_bw_bps>0 else 0.0
    t_io     = (ioB     / mem_bw_bps) if mem_bw_bps>0 else 0.0
    t_sha    = (shaB  * cpb_sha / cpu_hz)
    t_cmac   = (cmacB * cpb_aes / cpu_hz)
    t_gemm   = max((gemmF/1e9)/peak_gflops, 0.0)

    out = {
      "probe":"opprobe",
      "cpu_hz":cpu_hz,
      "machine":{"peak_gflops":peak_gflops,"mem_bw_bps":mem_bw_bps},
      "counts":{"memcpy_bytes":memcpyB,"io_bytes":ioB,"sha256_bytes":shaB,"cmac_bytes_aligned":cmacB,"gemm_flops":gemmF},
      "time_s":{"memcpy":t_memcpy,"io":t_io,"sha256":t_sha,"aes128_cmac":t_cmac,"gemm":t_gemm,"sum_partial": t_memcpy+t_io+t_sha+t_cmac+t_gemm}
    }
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()
