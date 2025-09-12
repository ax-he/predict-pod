CC := gcc
CFLAGS := -O2 -fPIC -std=c11 -Wall -Wextra
LDFLAGS := -ldl -lpthread

# 模式开关（默认 TOTAL + DRY-RUN）。可在命令行传：MODE=MAX 或 DRYRUN=0
MODE ?= TOTAL   # TOTAL 或 MAX
DRYRUN ?= 1

ifeq ($(MODE),MAX)
  CFLAGS += -DPROBE_MODE_MAX
else
  CFLAGS += -DPROBE_MODE_TOTAL
endif

ifeq ($(DRYRUN),0)
  CFLAGS += -DPROBE_DRY_RUN=0
endif

all: libfftprobe.so fft_test

libfftprobe.so: libfftprobe.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(LDFLAGS)

fft_test: fft_test.c
	$(CC) -O2 -std=c11 $< -o $@ -lfftw3 -lfftw3f

clean:
	rm -f libfftprobe.so fft_test
