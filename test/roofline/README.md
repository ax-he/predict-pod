# 构建 TOTAL（或 MODE=max）
make MODE=total

# 单预算：末尾会额外输出一行裸 yes/no
TIME_BUDGET=0.5 PEAK_GFLOPS=50.906 MEM_BW_GBS=14.53 \
LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test

# 多预算
TIME_BUDGETS="0.1,0.5,1.0" LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test

# 模式专属预算
TIME_BUDGET_MAX=0.2    LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test   # MAX
TIME_BUDGET_TOTAL=3.0  LD_PRELOAD="$PWD/libblasprobe.so" .././gemm-test   # TOTAL
