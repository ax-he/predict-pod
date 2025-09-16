#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# === 可调参数 ===
CORE="${CORE:-0}"            # 固定到哪个 CPU 核
RPT="${RPT:-5}"              # perf stat 重复次数 (-r)
RUNS_FALLBACK="${RUNS_FALLBACK:-3}"   # 无 cycles 时，纯时间法重复次数
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O3 -ffast-math -fno-math-errno -march=native -mtune=native}"
SRC="${SRC:-bench_libm.c}"
BIN="${BIN:-bench}"
CSV_OUT="${CSV_OUT:-cpe_results.csv}"

# 找 perf：优先环境变量 PERF；否则尝试内核匹配路径；再兜底扫描已安装版本
find_perf() {
  if [[ -n "${PERF:-}" && -x "${PERF}" ]]; then
    echo "${PERF}"; return
  fi
  local p="/usr/lib/linux-tools/$(uname -r)/perf"
  if [[ -x "$p" ]]; then echo "$p"; return; fi
  # 扫描已装的 perf 实体，取版本号最高的一个
  local cand
  cand=$(ls -1 /usr/lib/linux-tools-*/perf /usr/lib/linux-tools/*/perf 2>/dev/null | sort -V | tail -n1 || true)
  if [[ -n "$cand" && -x "$cand" ]]; then echo "$cand"; return; fi
  echo ""   # 没找到
}

PERF_BIN="$(find_perf || true)"
if [[ -n "$PERF_BIN" ]]; then
  echo "[info] using perf: $PERF_BIN"
else
  echo "[warn] perf not found or not executable; will use time-based fallback."
fi

# 编译 bench（若需要）
if [[ ! -x "$BIN" || "$BIN" -ot "$SRC" ]]; then
  echo "[info] building $BIN from $SRC ..."
  $CC $CFLAGS "$SRC" -lm -o "$BIN"
fi

# 所有要测的函数
FUNCS=(expf logf sinf cosf powf sqrtf exp log sin cos pow sqrt)

# 获取 CPU_HZ（用于 fallback）
CPU_HZ=$(awk '/cpu MHz/ {printf("%.0f",$4*1000000); exit}' /proc/cpuinfo 2>/dev/null || echo 3500000000)

# 工具函数：从 bench 输出提取 ELEMS / ELAPSED_NS
get_elems()   { sed -n 's/.*ELEMS=\([0-9]\+\).*/\1/p'   | tail -n1; }
get_elapsed() { sed -n 's/.*ELAPSED_NS=\([0-9]\+\).*/\1/p' | tail -n1; }

# 计算并打印表头
printf "%-7s %14s %14s %12s\n" "FUNC" "CYCLES(avg)" "ELEMS" "CPE"
printf "%-7s %14s %14s %12s\n" "-----" "-----------" "--------------" "------------" | sed 's/./-/g' >/dev/null

# CSV 头
echo "func,cycles_avg,elems,cpe" > "$CSV_OUT"

# 结果 map，用于最后生成 export
declare -A CPE_MAP

for f in "${FUNCS[@]}"; do
  # 先跑一次拿 ELEMS（stdout）
  ELE_OUT=$(taskset -c "$CORE" "./$BIN" "$f")
  ELEMS=$(echo "$ELE_OUT" | get_elems)
  if [[ -z "$ELEMS" ]]; then
    echo "[error] cannot get ELEMS for $f"; continue
  fi

  CYCLES=""
  if [[ -n "$PERF_BIN" ]]; then
    # perf CSV 输出到 stderr；-r RPT 会给出均值+stddev；我们取最后一个 cycles 行的数值
    CYCLES=$(taskset -c "$CORE" "$PERF_BIN" stat -x, -e cycles -r "$RPT" -- "./$BIN" "$f" 2>&1 \
             | awk -F, '$3 ~ /^cycles/ {val=$1} END{print val}')
    # perf 会把多次运行的 bench stdout 也打印，但与解析 cycles 不冲突
  fi

  # fallback：没有 cycles（WSL2/权限/PMU 限制），用时间近似 cycles
  if [[ -z "$CYCLES" ]]; then
    sum_ns=0
    for i in $(seq 1 "$RUNS_FALLBACK"); do
      out=$(taskset -c "$CORE" "./$BIN" "$f")
      ns=$(echo "$out" | get_elapsed)
      : $(( sum_ns += ns ))
    done
    avg_ns=$(( sum_ns / RUNS_FALLBACK ))
    # cycles ≈ ns * CPU_HZ / 1e9
    CYCLES=$(awk -v ns="$avg_ns" -v hz="$CPU_HZ" 'BEGIN{printf("%.0f", ns*hz/1e9)}')
  fi

  # CPE = cycles / elems
  CPE=$(awk -v cyc="$CYCLES" -v el="$ELEMS" 'BEGIN{if(el>0) printf("%.4f", cyc/el); else print "nan"}')

  printf "%-7s %14s %14s %12s\n" "$f" "$CYCLES" "$ELEMS" "$CPE"
  echo "$f,$CYCLES,$ELEMS,$CPE" >> "$CSV_OUT"

  # 存到 map（转为 libprobe 的变量名）
  case "$f" in
    expf)  CPE_MAP[PROBE_CPE_expf]="$CPE" ;;
    logf)  CPE_MAP[PROBE_CPE_logf]="$CPE" ;;
    sinf)  CPE_MAP[PROBE_CPE_sinf]="$CPE" ;;
    cosf)  CPE_MAP[PROBE_CPE_cosf]="$CPE" ;;
    powf)  CPE_MAP[PROBE_CPE_powf]="$CPE" ;;
    sqrtf) CPE_MAP[PROBE_CPE_sqrtf]="$CPE" ;;
    exp)   CPE_MAP[PROBE_CPE_expd]="$CPE" ;;
    log)   CPE_MAP[PROBE_CPE_logd]="$CPE" ;;
    sin)   CPE_MAP[PROBE_CPE_sind]="$CPE" ;;
    cos)   CPE_MAP[PROBE_CPE_cosd]="$CPE" ;;
    pow)   CPE_MAP[PROBE_CPE_powd]="$CPE" ;;
    sqrt)  CPE_MAP[PROBE_CPE_sqrtd]="$CPE" ;;
  esac
done

echo
echo "[info] CSV saved to: $CSV_OUT"
echo
echo "# === export to libprobe.so ==="
for k in "${!CPE_MAP[@]}"; do
  printf "export %s=%s\n" "$k" "${CPE_MAP[$k]}"
done
