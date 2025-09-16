bench_libm.c: 在本地校准exp/log/pow/sin/cos/sqrt等数学函数的CPE (cycles per element)

calibrate_cpe.sh: 自动化脚本，输出各函数在6轮下的平均值

libprobe.c：面向math.h的数学函数和memcpy，并支持Intel SVML的常见向量入口：统计 libm 标量函数调用次数，统计 memcpy 总字节数用于内存带宽时间估算

gcc -O2 -fPIC -shared -o libprobe.so libprobe.c -ldl -pthread

LD_PRELOAD=$PWD/libprobe.so ./your_program