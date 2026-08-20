import time

import board
import pwmio
import usb_cdc


TO = 10 * 60
D0 = 0
D1 = 32768
F0 = 500
ON0 = 2.0
OFF0 = 3.0
FMIN = 10
FMAX = 5000
TMIN = 10
TMAX = 120000


A3 = board.A3
A4 = board.A4


def mk(pin, freq):
    return pwmio.PWMOut(pin, frequency=freq, duty_cycle=D0, variable_frequency=False)


def pui(buf, start, end):
    if start >= end:
        return None
    v = 0
    i = start
    while i < end:
        c = buf[i]
        if c < 48 or c > 57:
            return None
        v = (v * 10) + (c - 48)
        i += 1
    return v


def refreq(pwm, st, ph, nx, freq):
    a, b = st[0], st[1]
    for p in pwm:
        try:
            p.duty_cycle = D0
        except Exception:
            pass
    for p in pwm:
        try:
            p.deinit()
        except Exception:
            pass
    pwm[0] = mk(A3, freq)
    pwm[1] = mk(A4, freq)
    ph[0] = 0
    ph[1] = 0
    nx[0] = 0.0
    nx[1] = 0.0
    st[0] = a
    st[1] = b


def main():
    ds = usb_cdc.data
    ds.timeout = 0

    freq = F0
    on_s = ON0
    off_s = OFF0

    pwm = [mk(A3, freq), mk(A4, freq)]
    st = [0, 0]
    ph = [0, 0]  # 0=off, 1=on-phase, 2=rest-phase
    nx = [0.0, 0.0]

    last = time.monotonic()
    standby = 1

    rx = bytearray(1)
    cmd = bytearray(24)
    n = 0

    while True:
        now = time.monotonic()

        if ds.readinto(rx):
            c = rx[0]

            if c == 10 or c == 13:
                if n:
                    op = cmd[0]

                    if n == 1:
                        if op == 63:  # ?
                            try:
                                ds.write(b"l2\n")
                            except Exception:
                                pass
                        elif op == 114:  # r
                            on_s = ON0
                            off_s = OFF0
                            if freq != F0:
                                freq = F0
                                refreq(pwm, st, ph, nx, freq)
                            else:
                                pwm[0].duty_cycle = D0
                                pwm[1].duty_cycle = D0
                                ph[0] = 0
                                ph[1] = 0
                                nx[0] = 0.0
                                nx[1] = 0.0
                            last = now
                            standby = 0

                    elif n == 2:
                        s = cmd[1]
                        if s == 48 or s == 49:
                            v = 1 if s == 49 else 0
                            if op == 97:  # a
                                st[0] = v
                                last = now
                                standby = 0
                            elif op == 98:  # b
                                st[1] = v
                                last = now
                                standby = 0

                    elif op == 102:  # f<hz>
                        hz = pui(cmd, 1, n)
                        if hz is not None and FMIN <= hz <= FMAX:
                            freq = hz
                            refreq(pwm, st, ph, nx, freq)
                            last = now
                            standby = 0

                    elif op == 119:  # w<on_ms>,<off_ms>
                        comma = -1
                        i = 1
                        while i < n:
                            if cmd[i] == 44:
                                comma = i
                                break
                            i += 1
                        if comma > 1 and comma < (n - 1):
                            on_ms = pui(cmd, 1, comma)
                            off_ms = pui(cmd, comma + 1, n)
                            if (
                                on_ms is not None
                                and off_ms is not None
                                and TMIN <= on_ms <= TMAX
                                and TMIN <= off_ms <= TMAX
                            ):
                                on_s = on_ms * 0.001
                                off_s = off_ms * 0.001
                                if st[0]:
                                    pwm[0].duty_cycle = D0
                                    ph[0] = 0
                                    nx[0] = 0.0
                                if st[1]:
                                    pwm[1].duty_cycle = D0
                                    ph[1] = 0
                                    nx[1] = 0.0
                                last = now
                                standby = 0

                n = 0

            elif c == 9 or c == 32:
                pass

            else:
                if n < len(cmd):
                    cmd[n] = c
                    n += 1
                else:
                    n = 0

        if (not standby) and (now - last >= TO):
            st[0] = 0
            st[1] = 0
            standby = 1

        if st[0]:
            if ph[0] == 0:
                pwm[0].duty_cycle = D1
                ph[0] = 1
                nx[0] = now + on_s
            elif now >= nx[0]:
                if ph[0] == 1:
                    pwm[0].duty_cycle = D0
                    ph[0] = 2
                    nx[0] = now + off_s
                else:
                    pwm[0].duty_cycle = D1
                    ph[0] = 1
                    nx[0] = now + on_s
        elif ph[0] != 0 or pwm[0].duty_cycle != D0:
            pwm[0].duty_cycle = D0
            ph[0] = 0
            nx[0] = 0.0

        if st[1]:
            if ph[1] == 0:
                pwm[1].duty_cycle = D1
                ph[1] = 1
                nx[1] = now + on_s
            elif now >= nx[1]:
                if ph[1] == 1:
                    pwm[1].duty_cycle = D0
                    ph[1] = 2
                    nx[1] = now + off_s
                else:
                    pwm[1].duty_cycle = D1
                    ph[1] = 1
                    nx[1] = now + on_s
        elif ph[1] != 0 or pwm[1].duty_cycle != D0:
            pwm[1].duty_cycle = D0
            ph[1] = 0
            nx[1] = 0.0

        time.sleep(0.01)


main()
