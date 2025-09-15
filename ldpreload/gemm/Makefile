CC := gcc
CFLAGS := -O2 -std=c11 -Wall -Wextra -fPIC
LDFLAGS := -ldl -lpthread

# 默认：TOTAL 模式 + DRYRUN=1
MODE   ?= total      # 可用 make MODE=max 覆盖
DRYRUN ?= 1          # 可用 make DRYRUN=0 覆盖

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
