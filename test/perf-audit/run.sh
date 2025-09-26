#!/usr/bin/env bash
set -euo pipefail

# -------- Configurable machine model (from you) --------
PEAK_GFLOPS=50.906
MEM_BW_BPS=14530000000   # 14.53 GB/s
# Default CPU Hz (best-effort): use /proc/cpuinfo; fallback 1.8GHz (typical WSL)
CPU_HZ=$(awk -F': *' '/cpu MHz/ {printf "%.0f\n",$2*1e6; exit}' /proc/cpuinfo || true)
CPU_HZ=${CPU_HZ:-1800000000}

# SHA256/AES-CMAC cycles per byte (rough defaults; calibrate later if需要)
SHA256_CPB=${SHA256_CPB:-12.0}
CMAC_CPB=${CMAC_CPB:-20.0}

echo "[info] CPU_HZ=$CPU_HZ, PEAK_GFLOPS=$PEAK_GFLOPS, MEM_BW_BPS=$MEM_BW_BPS"

# -------- 0. Dependencies --------
echo "[info] Installing build deps for tools/perf and OpenSSL app ..."
sudo apt-get update -y
sudo apt-get install -y \
  build-essential git pkg-config zlib1g-dev \
  libelf-dev libdw-dev libunwind-dev libcap-dev \
  libtraceevent-dev libtracefs-dev \
  libslang2-dev libperl-dev libbabeltrace-dev \
  binutils-dev libiberty-dev \
  libssl-dev libopenblas-dev jq

# Lower paranoid so perf can record
echo "[info] Setting kernel.perf_event_paranoid=-1"
sudo sysctl kernel.perf_event_paranoid=-1 >/dev/null

# -------- 1. Build perf from WSL2 kernel sources (userspace tools only) --------
ROOT="$PWD"
PERF_SRC="$ROOT/WSL2-Linux-Kernel"
PERF_BIN="$PERF_SRC/tools/perf/perf"

if [[ ! -x "$PERF_BIN" ]]; then
  echo "[info] building perf from WSL2 kernel source..."
  if [[ ! -d "$PERF_SRC" ]]; then
    git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git "$PERF_SRC"
  fi
  make -C "$PERF_SRC/tools/perf" -j"$(nproc)"
else
  echo "[info] perf already built: $PERF_BIN"
fi
PERF="$PERF_BIN"
"$PERF" --version || (echo "[err] perf not runnable"; exit 1)

# -------- 2. Test app (OpenSSL SHA256 + CMAC + memcpy/expf mix) --------
cat > test_app.c <<'EOF'
#include <openssl/sha.h>
#include <openssl/cmac.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void do_sha256(const unsigned char* buf, size_t len, unsigned char out[32]) {
    SHA256_CTX c; SHA256_Init(&c);
    size_t off = 0;
    while (off < len) {
        size_t l = (len - off > 8192) ? 8192 : (len - off);
        SHA256_Update(&c, buf + off, l);
        off += l;
    }
    SHA256_Final(out, &c);
}

static void do_cmac(const unsigned char* key, const unsigned char* buf, size_t len, unsigned char mac[16]) {
    CMAC_CTX* ctx = CMAC_CTX_new();
    CMAC_Init(ctx, key, 16, EVP_aes_128_cbc(), NULL);
    size_t off = 0; size_t maclen = 0;
    while (off < len) {
        size_t l = (len - off > 4096) ? 4096 : (len - off);
        CMAC_Update(ctx, buf + off, l);
        off += l;
    }
    CMAC_Final(ctx, mac, &maclen);
    CMAC_CTX_free(ctx);
}

int main() {
    const size_t N = 16 * 1024 * 1024; // 16 MiB payload
    unsigned char *buf = (unsigned char*) aligned_alloc(64, N);
    if (!buf) { perror("alloc"); return 1; }
    RAND_bytes(buf, N);

    // mixed scalar work to avoid optimizing-away
    volatile float acc = 0.0f;
    for (size_t i = 0; i < (1<<20); i++) acc += expf((float)(i & 1023) * 1e-3f);

    // sha256 + cmac
    unsigned char out[32], mac[16], key[16];
    RAND_bytes((unsigned char*)key, sizeof(key));
    do_sha256(buf, N, out);
    do_cmac(key, buf, N, mac);

    // some memcpys
    unsigned char *tmp = (unsigned char*) aligned_alloc(64, N);
    memcpy(tmp, buf, N);
    memcpy(buf, tmp, N/2);

    printf("done acc=%f out0=%02x mac0=%02x\n", acc, out[0], mac[0]);
    free(tmp); free(buf);
    return 0;
}
EOF

echo "[info] building test_app ..."
gcc -O3 -march=native -mtune=native test_app.c -o test_app -lcrypto -lm

# -------- 3. Setup uprobes on libcrypto --------
LIBCRYPTO=$(ldd ./test_app | awk '/libcrypto\.so/ {print $3; exit}')
if [[ -z "${LIBCRYPTO:-}" ]]; then
  echo "[err] libcrypto path not found via ldd"
  exit 1
fi
echo "[info] libcrypto: $LIBCRYPTO"

# Clean old probes if any
sudo "$PERF" probe --del='probe_libcrypto:*' >/dev/null 2>&1 || true

# Add probes with argument capture: len=%dx (3rd arg in x86-64 SysV)
# If versioned symbols cause trouble, you can also use -F to list and adjust names.
sudo "$PERF" probe -x "$LIBCRYPTO" 'SHA256_Update len=%dx:u64'
sudo "$PERF" probe -x "$LIBCRYPTO" 'CMAC_Update  len=%dx:u64'

echo "[info] current probes:"
sudo "$PERF" probe -l | sed 's/^/  /'

# -------- 4. Record and script --------
echo "[info] recording ..."
sudo "$PERF" record -q -e probe_libcrypto:SHA256_Update -e probe_libcrypto:CMAC_Update -- ./test_app

echo "[info] decoding with perf script ..."
"$PERF" script -i perf.data -F comm,pid,time,event,trace > perf.out

# -------- 5. Parse & estimate --------
cat > perf2json.py <<PY
import json, os, re, sys

CPU_HZ = int(os.environ.get("CPU_HZ","$CPU_HZ"))
PEAK_GFLOPS = float(os.environ.get("PEAK_GFLOPS","$PEAK_GFLOPS"))
MEM_BW_BPS = float(os.environ.get("MEM_BW_BPS","$MEM_BW_BPS"))
SHA256_CPB = float(os.environ.get("SHA256_CPB","$SHA256_CPB"))
CMAC_CPB   = float(os.environ.get("CMAC_CPB","$CMAC_CPB"))

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
PY

python3 perf2json.py < perf.out | tee probe.json

echo
echo "[done] probe.json written:"
jq . probe.json || cat probe.json
