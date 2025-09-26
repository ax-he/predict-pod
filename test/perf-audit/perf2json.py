import json, os, re, sys

CPU_HZ = int(os.environ.get("CPU_HZ","1800000000"))
PEAK_GFLOPS = float(os.environ.get("PEAK_GFLOPS","50.906"))
MEM_BW_BPS = float(os.environ.get("MEM_BW_BPS","14530000000"))
SHA256_CPB = float(os.environ.get("SHA256_CPB","12.0"))
CMAC_CPB   = float(os.environ.get("CMAC_CPB","20.0"))

sha_bytes = 0
cmac_bytes = 0

pat = re.compile(r'(probe_libcrypto:)?(SHA256_Update|CMAC_Update):.*len=(\d+)')
for line in sys.stdin:
    m = pat.search(line)
    if not m: continue
    fn = m.group(2)
    ln = int(m.group(3))
    if fn == "SHA256_Update":
        sha_bytes += ln
    elif fn == "CMAC_Update":
        cmac_bytes += ln

def time_from_cpb(bytes_, cpb):
    return (bytes_ * cpb) / CPU_HZ

def time_mem_bound(bytes_):
    return bytes_ / MEM_BW_BPS

# Take max of compute vs mem path (rough roofline-style upper bound)
sha_t  = max(time_from_cpb(sha_bytes, SHA256_CPB), time_mem_bound(sha_bytes))
cmac_t = max(time_from_cpb(cmac_bytes, CMAC_CPB), time_mem_bound(cmac_bytes))

out = {
  "probe": "perf-probe",
  "cpu_hz": CPU_HZ,
  "machine": {
    "peak_gflops": PEAK_GFLOPS,
    "mem_bw_bps": MEM_BW_BPS
  },
  "counts": {
    "sha256_bytes": sha_bytes,
    "cmac_bytes": cmac_bytes
  },
  "time_s": {
    "sha256": sha_t,
    "aes128_cmac": cmac_t,
    "sum_partial": sha_t + cmac_t
  }
}
print(json.dumps(out, indent=2))
