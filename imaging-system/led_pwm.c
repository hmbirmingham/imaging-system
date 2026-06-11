/*
 * led_pwm.c — Hardware PWM driver for Raspberry Pi 5 RP1 peripheral.
 *
 * Compiled as a shared library:
 *   gcc -O2 -shared -fPIC -o led_pwm.so led_pwm.c
 *
 * Directly mmaps the RP1 PWM and GPIO register banks from /dev/mem and
 * configures PWM0_CHAN0 on GPIO 12 for LED brightness control.
 *
 * PWM clock on RP1 is fixed at 25 MHz. With a range of 1000 the switching
 * frequency is 25 kHz — above audible range, below any camera flicker threshold.
 *
 * Hardware target : Raspberry Pi 5
 * GPIO            : 12  (PWM0_CHAN0, ALT4)
 * PWM base (phys) : 0x1f00098000
 * GPIO base (phys): 0x1f000d0000
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* ── RP1 physical addresses ──────────────────────────────────────────────── */
#define RP1_PWM0_PHYS   0x1f00098000UL
#define RP1_GPIO_PHYS   0x1f000d0000UL

/* ── PWM register offsets (byte) ─────────────────────────────────────────── */
#define PWM_CTL         0x00    /* Control register          */
#define PWM_STA         0x04    /* Status register           */
#define PWM_RNG1        0x10    /* Channel 1 range (period)  */
#define PWM_DAT1        0x14    /* Channel 1 data  (duty)    */

/* ── GPIO register offset within per-pin 8-byte block ───────────────────── */
#define GPIO_CTRL_OFF   0x04    /* FUNCSEL lives in CTRL word */

/* ── PWM configuration ───────────────────────────────────────────────────── */
#define PWM_RANGE       1000    /* 25 MHz / 1000 = 25 kHz switching freq */
#define PWM_GPIO        12      /* Physical GPIO pin                      */
#define PWM_FUNCSEL     4       /* ALT4 routes GPIO 12 to PWM0_CHAN0      */

/* ── CTL register bit masks ──────────────────────────────────────────────── */
#define CTL_PWEN1       (1u << 0)   /* Enable channel 1          */
#define CTL_MSEN1       (1u << 7)   /* M/S mode — true duty cycle */

/* ── Module state ────────────────────────────────────────────────────────── */
static volatile uint32_t *_pwm  = NULL;
static volatile uint32_t *_gpio = NULL;
static long               _page = 0;
static int                _available = 0;
static float              _brightness = 0.0f;

/* ── Internal helpers ────────────────────────────────────────────────────── */

/*
 * write_pwm — write a 32-bit value to a PWM register.
 * reg is a byte offset; dividing by 4 gives the uint32_t word index.
 * The memory barrier ensures the store is visible to hardware before returning.
 */
static void write_pwm(uint32_t reg, uint32_t val)
{
    _pwm[reg / 4] = val;
    __sync_synchronize();
}

/*
 * set_gpio_function — configure the FUNCSEL field in GPIO_CTRL to route the
 * pin to the PWM peripheral.  RP1 allocates 8 bytes per GPIO pin; CTRL is at
 * byte offset 4 within that block.  Only the bottom 5 bits are FUNCSEL.
 */
static void set_gpio_function(int gpio)
{
    uint32_t byte_off  = (uint32_t)(gpio * 8 + GPIO_CTRL_OFF);
    uint32_t word_idx  = byte_off / 4;
    uint32_t val       = _gpio[word_idx];

    val = (val & ~0x1Fu) | (PWM_FUNCSEL & 0x1Fu);
    _gpio[word_idx] = val;
    __sync_synchronize();
}

/* ── Public API ──────────────────────────────────────────────────────────── */

/*
 * pwm_init — open /dev/mem, mmap PWM and GPIO register pages, configure
 * GPIO function select, set the PWM range, and enable the channel.
 * Returns 1 on success, 0 on any failure (access denied, not Pi hardware, etc.).
 */
int pwm_init(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "LED PWM: cannot open /dev/mem: %s\n", strerror(errno));
        return 0;
    }

    _page = sysconf(_SC_PAGESIZE);

    /* Map PWM register page */
    off_t pwm_aligned = (off_t)(RP1_PWM0_PHYS & (unsigned long)~(_page - 1));
    void *pwm_ptr = mmap(NULL, (size_t)_page,
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd, pwm_aligned);
    if (pwm_ptr == MAP_FAILED) {
        fprintf(stderr, "LED PWM: mmap PWM failed: %s\n", strerror(errno));
        close(fd);
        return 0;
    }

    /* Map GPIO register page */
    off_t gpio_aligned = (off_t)(RP1_GPIO_PHYS & (unsigned long)~(_page - 1));
    void *gpio_ptr = mmap(NULL, (size_t)_page,
                          PROT_READ | PROT_WRITE, MAP_SHARED, fd, gpio_aligned);
    if (gpio_ptr == MAP_FAILED) {
        fprintf(stderr, "LED PWM: mmap GPIO failed: %s\n", strerror(errno));
        munmap(pwm_ptr, (size_t)_page);
        close(fd);
        return 0;
    }

    close(fd);   /* fd no longer needed once pages are mapped */

    _pwm  = (volatile uint32_t *)pwm_ptr;
    _gpio = (volatile uint32_t *)gpio_ptr;

    /* Configure peripheral */
    set_gpio_function(PWM_GPIO);
    write_pwm(PWM_RNG1, PWM_RANGE);
    write_pwm(PWM_DAT1, 0);
    write_pwm(PWM_CTL,  CTL_MSEN1 | CTL_PWEN1);

    _available = 1;
    return 1;
}

/*
 * pwm_set_brightness — set LED brightness as a percentage (0.0 – 100.0).
 * Clamps silently to range.  Converts percentage to a duty cycle count
 * within [0, PWM_RANGE] and writes it to the DAT1 register.
 */
void pwm_set_brightness(float percent)
{
    if (percent < 0.0f)   percent = 0.0f;
    if (percent > 100.0f) percent = 100.0f;
    _brightness = percent;

    if (!_available) return;

    uint32_t duty = (uint32_t)(PWM_RANGE * percent / 100.0f);
    write_pwm(PWM_DAT1, duty);
}

/*
 * pwm_get_brightness — return the last brightness value set (0.0 – 100.0).
 * Returns 0.0 if the driver was never initialised.
 */
float pwm_get_brightness(void)
{
    return _brightness;
}

/*
 * pwm_off — convenience wrapper: set brightness to zero.
 */
void pwm_off(void)
{
    pwm_set_brightness(0.0f);
}

/*
 * pwm_is_available — return 1 if pwm_init succeeded, 0 otherwise.
 * Used by the Python wrapper to set the `available` flag without
 * catching exceptions.
 */
int pwm_is_available(void)
{
    return _available;
}

/*
 * pwm_cleanup — turn off LED and release mmap regions.
 * Call before process exit to avoid leaving the LED on.
 */
void pwm_cleanup(void)
{
    if (_available)
        pwm_off();

    if (_pwm) {
        munmap((void *)_pwm, (size_t)_page);
        _pwm = NULL;
    }
    if (_gpio) {
        munmap((void *)_gpio, (size_t)_page);
        _gpio = NULL;
    }
    _available = 0;
}
