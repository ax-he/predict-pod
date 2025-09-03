CC := gcc
CFLAGS := -O2
LIBS := -lopenblas -lm

.PHONY: all clean

all: gemm-test

# $@ 是自动变量，代表“当前规则的目标”（这里是 gemm-test）。
# $< 是第一个先决条件（这里是对应的 .c 文件）
gemm-test: complex_gemm.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f gemm-test