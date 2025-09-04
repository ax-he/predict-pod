CC := gcc
CFLAGS := -O2 -std=c11 -Wall -Wextra -fPIC
LDFLAGS := -ldl -lpthread
MODE ?= max    # max 或 total
DRYRUN ?= 0    # 1 = 编译期启用 DRY-RUN（也可运行时 PROBE_DRY_RUN=1）

ifeq ($(MODE),total)
  CFLAGS += -DPROBE_MODE_TOTAL
else
  CFLAGS += -DPROBE_MODE_MAX
endif
ifeq ($(DRYRUN),1)
  CFLAGS += -DPROBE_DRY_RUN
endif

all: libblasprobe.so
libblasprobe.so: libblasprobe_roofline_modes.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(LDFLAGS)

clean:
	rm -f libblasprobe.so
