#!/usr/bin/env python3
import re, json, argparse

p_sha = re.compile(r'\bsha256_upd\b.*\blen=(\d+)')
p_cmu = re.compile(r'\bcmac_upd\b.*\blen=(\d+)')
p_mcp = re.compile(r'\bmemcpy_sz\b.*\bsize=(\d+)')
p_sgm = re.compile(r'\bsgemm\b.*\bM=(\d+)\s+N=(\d+)\s+K=(\d+)')
p_dgm = re.compile(r'\bdgemm\b.*\bM=(\d+)\s+N=(\d+)\s+K=(\d+)')

def ceil_div(a,b): return (a + b - 1)//b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--peak-gflops', type=float, required=True)
    ap.add_argument('--mem-bw-bps', type=float, required=True)
    args = ap.parse_args()

    sha_bytes = 0
    cmac_bytes = 0
    memcpy_bytes = 0
    gemm_flops = 0

    with open(args.trace, 'r', errors='ignore') as f:
        for line in f:
            m = p_sha.search(line)
            if m: sha_bytes += int(m.group(1))
            m = p_cmu.search(line)
            if m:
                ln = int(m.group(1))
                cmac_bytes += 16 * ceil_div(ln, 16)   # 16B 分组取整估算
            m = p_mcp.search(line)
            if m: memcpy_bytes += int(m.group(1))
            m = p_sgm.search(line)
            if m:
                M,N,K = map(int, m.groups())
                gemm_flops += 2.0 * M * N * K
            m = p_dgm.search(line)
            if m:
                M,N,K = map(int, m.groups())
                gemm_flops += 2.0 * M * N * K

    # Roofline 下界估算：
    # - memcpy/sha256/cmac 用内存带宽下界（它们大多内存带宽受限）
    # - gemm 用峰值 GFLOPS 的一定效率（给个保守 80%）
    bw = args.mem_bw_bps
    peak = args.peak_gflops * 1e9
    gemm_eff = 0.80

    t_memcpy = (memcpy_bytes / bw) if bw>0 else 0.0
    t_sha    = (sha_bytes    / bw) if bw>0 else 0.0
    t_cmac   = (cmac_bytes   / bw) if bw>0 else 0.0
    t_gemm   = (gemm_flops   / (peak * gemm_eff)) if peak>0 else 0.0

    out = {
        "probe": "perf-probe",
        "machine": {
            "peak_gflops": args.peak_gflops,
            "mem_bw_bps":  args.mem_bw_bps
        },
        "counts": {
            "memcpy_bytes": memcpy_bytes,
            "sha256_bytes": sha_bytes,
            "cmac_bytes_aligned": cmac_bytes,
            "gemm_flops": int(gemm_flops)
        },
        "time_s": {
            "memcpy": t_memcpy,
            "sha256": t_sha,
            "aes128_cmac": t_cmac,
            "gemm": t_gemm,
            "sum_partial": t_memcpy + t_sha + t_cmac + t_gemm
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
