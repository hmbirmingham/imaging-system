# Makefile — build the LED PWM shared library
# Run: make
# Clean: make clean

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -shared -fPIC

led_pwm.so: led_pwm.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Built $@"

clean:
	rm -f led_pwm.so

.PHONY: clean
