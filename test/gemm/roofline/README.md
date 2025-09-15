# 1) 编译 TOTAL + 仅预测不执行（编译期开启）
make MODE=total DRYRUN=1

# 2) 或者：编译 MAX，运行时按环境开关 DRY-RUN
make MODE=max

PROBE_DRY_RUN=1 TIME_BUDGET=0.5 \
PEAK_GFLOPS=50.906 MEM_BW_GBS=14.53 \
LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test

# 3) 多个 budget（逗号/空格分隔）
PROBE_DRY_RUN=1 TIME_BUDGETS="0.2, 0.5, 1.0" LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test

# 4) 模式专属（与两模式同名变量兼容）
PROBE_DRY_RUN=1 TIME_BUDGET_TOTAL=3.0 LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test    # TOTAL

PROBE_DRY_RUN=1 TIME_BUDGET_MAX=0.2   LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test    # MAX
