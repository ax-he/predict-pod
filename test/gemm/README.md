# 依赖安装
sudo apt-get update

sudo apt-get install -y build-essential libopenblas-dev python3

# 编译
make

// 查看程序是否确实动态链接到BLAS

ldd ./gemm-test | grep -i blas

# 查看参数，避免运行
LOG_DIR=./predict PREDICT_ONLY=1 LD_PRELOAD=$PWD/libblasprobe.so ./gemm-test



// 当前问题，只会记录第一个钩子，如果第一个很小但后续很大的话会导致输出错误的预检结果，

gcc -shared -fPIC -O2 -std=c11 -Wall -Werror   libblasprobe_stream_safe.c -o libblasprobe.so -ldl -lpthread

LOG_DIR=./predict LD_PRELOAD="$PWD/libblasprobe.so" ./gemm-test