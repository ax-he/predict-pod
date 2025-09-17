**1) 编译**

make            # 生成 libfftprobe.so 与 fft_test

**2) 仅统计（DRY-RUN），并给出预算评估**
<!-- export PEAK_GFLOPS=250.91

export MEM_BW_GBS=14.53 -->

export TIME_BUDGETS="0.02,0.05"

export IO_FACTOR=2.0          # 估算 IO 流量系数，按需调整

LD_PRELOAD=./libfftprobe.so ./fft_test

TIME_BUDGETS="0.02,0.05" LD_PRELOAD=./libfftprobe.so ./fft_test

**3) 真正执行 FFT（关闭 DRY-RUN）**
make clean && make DRYRUN=0

LD_PRELOAD=./libfftprobe.so ./fft_test
