CC := gcc
CFLAGS := -O2 -std=c11 -Wall -Wextra -fPIC
LDFLAGS := -ldl -lpthread
MODE ?= max  # max 或 total

ifeq ($(MODE),total)
  CFLAGS += -DPROBE_MODE_TOTAL
else
  CFLAGS += -DPROBE_MODE_MAX
endif

all: libblasprobe.so
libblasprobe.so: libblasprobe_roofline_modes.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(LDFLAGS)

clean:
	rm -f libblasprobe.so
