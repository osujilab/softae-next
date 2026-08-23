# Trinket M0 device firmware (versioned record)

These are **device-side CircuitPython programs**, not host code. They run on the two Adafruit
Trinket M0 microcontrollers wired into the rig, and they execute continuously from the moment the
board is powered — no host process is required to keep them running.

**The files in this directory are the versioned record. The `CIRCUITPY` volume on each device is
the live deployment.** They are copies, not links: nothing here is imported by `src/softae/`, and
nothing here is executed by the test suite. Their only job is to make the code that is actually
running on the hardware readable, diffable, and recoverable from git.

Copies were taken **2026-08-19**, byte-exact, from the mounted volumes (read-only; nothing was
written to either device).

## Contents

| Path | Device volume | Role |
|---|---|---|
| `dac0_rh/` | `DAC0` | Relative-humidity controller — drives two Aalborg PSV proportional valves (wet-air / dry-air mix) |
| `pwm0_piezo/` | `PWM0` | Piezo driver — gates two piezo channels with a duty-cycled on/rest sweep |

Each directory holds the device's `code.py` (the program), `boot.py` (enables the second USB CDC
endpoint), and `boot_out.txt` (the CircuitPython banner recorded at boot — firmware version and
board UID, useful for telling the two identical boards apart).

### SHA-256 as copied on 2026-08-19

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `dac0_rh/code.py` | 2279 | `2515E7F15142A6F56E783F10D648050A9180896E1F9F5C829928AA08DFE88847` |
| `pwm0_piezo/code.py` | 6381 | `4B835CBE3EEDB17F90D22DC310288100EA8113DA6C35C87AAA595CB23DD2648E` |
| `dac0_rh/boot.py` | 68 | `3328EC036462449DFAE00B27D1575E54E0AD3F2B0E4B6223F8EB75CFD2CEE1EE` |
| `pwm0_piezo/boot.py` | 68 | `3328EC036462449DFAE00B27D1575E54E0AD3F2B0E4B6223F8EB75CFD2CEE1EE` |
| `dac0_rh/boot_out.txt` | 156 | `F67795D1C0C4E4BFCAEA7C6CEBA74B7ACB677A4AB8E412F8495C2D1E63A008B9` |
| `pwm0_piezo/boot_out.txt` | 157 | `74FC6F45FB80E35A4BD99DD9CE7F9B123B273A5D8329197D11F0E789A56B3FF6` |

The two `boot.py` files are identical (same hash) — both merely call
`usb_cdc.enable(console=True, data=True)`, which is what gives the host a data endpoint separate
from the REPL console. The `boot_out.txt` files differ because the boards run different
CircuitPython builds and carry different UIDs:

| Device | CircuitPython | Board UID |
|---|---|---|
| `DAC0` | 9.2.7 (2025-04-01) | `8B8A77EE4A57305020312E592F1E0FFF` |
| `PWM0` | 10.2.1 (2026-05-13) | `B646AFC24A57305020312E59161610FF` |

---

## `dac0_rh` — RH controller

**Host-side counterpart:** `src/softae/drivers/async_rh_controller.py` (default port `COM11`,
115200 baud).

**Protocol — bare duty floats, one per line, host → device only.** The host writes
`f"{duty:.4f}\n"` and nothing else; there is no command grammar, no acknowledgement, and no
capability handshake. `duty` is the control value `ctrl` in `[0, 1]`: **`ctrl = 1` is fully humid
air, and `ctrl` just above 0 is the driest *flowing* state.** `ctrl == 0` exactly is not the dry
end of that range — it is the firmware's auto-shutoff, which closes **both** valves and therefore
stops supplying gas at all, leaving the chamber to drift towards room humidity. The device maps
`ctrl` linearly onto two PWM outputs — `board.A3` (the humid channel) over
`V0_range = [1.4, 2.5] V` rising with `ctrl`, and `board.A4` (the dry channel) over
`V1_range = [1.15, 2.7] V` falling with `ctrl` — at 10 kHz against a 3.33 V full scale. Each loop
pass holds the computed duty for 0.4 s, then drops to a reduced duty (÷1.5 on the humid channel,
÷2 on the dry channel) for 0.1 s; the humid-side brake exists to stop the bubbler over-bubbling.

> **`code.py`'s `# 0 is humid air, 1 is dry air` is a pin index, not a `ctrl` range.**
>
> That comment heads two lines beginning `0:` and `1:` that list voltage bounds, so its `0` and `1`
> are `pwmpin0` (`board.A3`, humid) and `pwmpin1` (`board.A4`, dry) — the same labelling
> `V0_range`'s own *"humidity signal range"* and `V1_range`'s *"dry air signal range"* carry. Read
> instead as the endpoints of `ctrl`, it asserts the exact opposite of what the arithmetic does,
> and that misreading has already inverted the direction for two careful readers and for an earlier
> revision of this README (SESSION_MAIL `[e10]` §1; bench-confirmed by the operator 2026-08-21).
> **The arithmetic is the authority:** `V0targ` scales with `ctrl`, `V1targ` with `1 - ctrl`.

The device prints status back on the same endpoint (`setting to <ctrl>` / `no value received,
remaining at <ctrl>`). The host does not parse it.

### DEADMAN — ≈ 25 s, self-recovering

- `usb_cdc.data.timeout = 0.75` — each `readline` blocks at most 0.75 s.
- A failed read *or* a failed `float()` parse falls into the `except` branch: `ctrl` reverts to
  `ctrl_latent` (the last good value) and `try_counter` increments.
- **20 consecutive failures** (`ctrl_timeout = 20`) force **both** `ctrl` and `ctrl_latent` to `0`.
  Zeroing `ctrl_latent` is the load-bearing part — it destroys the value the hold would otherwise
  keep reverting to.
- The `ctrl == 0` branch is an explicit auto-shutoff: both valve targets go to `0.0001 V`
  (≈ 0 V), closing both Aalborg PSVs.
- Loop period ≈ 1.25 s (0.75 s read timeout + 0.4 s + 0.1 s settle), so 20 failures ≈ **25 s**.
- **Self-recovering:** any subsequent valid float resets `try_counter` to 0 and resumes control
  immediately. No reset, no reconnect, no operator action.

The practical consequence: a host SIGKILL, power-cut or BSOD mid-hold latches the current duty for
about 25 seconds, after which the device closes the valves on its own. Orderly host-side shutdown
(`safe_off`) still writes duty 0 immediately — the deadman is the backstop, not the primary path.

---

## `pwm0_piezo` — piezo driver

**Host-side counterpart:** `src/softae/drivers/async_piezo.py` (default port `COM16`, 115200 baud),
with the command grammar centralized in `src/softae/core/piezo_protocol.py`.

**Protocol — the `l2` letter/command grammar,** ASCII, newline-terminated (`\n` or `\r`), tabs and
spaces ignored, command buffer capped at 24 bytes:

| Command | Meaning | Formatter in `piezo_protocol.py` |
|---|---|---|
| `?` | Capability query; device replies `l2\n` | `format_l2_caps_query()` |
| `a0` / `a1` | Channel A (`board.A3`) off / on | `format_l2_legacy_command("A", …)` |
| `b0` / `b1` | Channel B (`board.A4`) off / on | `format_l2_legacy_command("B", …)` |
| `f<hz>` | Set PWM carrier frequency, 10–5000 Hz | `format_l2_freq(hz)` |
| `w<on_ms>,<off_ms>` | Set sweep on/rest times, 10–120000 ms each | `format_l2_sweep_ms(on, off)` |
| `r` | Reset to defaults (500 Hz, 2.0 s on / 3.0 s rest) and clear both channels | `format_l2_reset()` |

The `?` → `l2` exchange is how the host discovers the device speaks this grammar at all
(`caps_supports_l2`). An enabled channel is not held on: it alternates duty `32768` for `on_s`
then duty `0` for `off_s`, indefinitely. `f<hz>` cannot be applied in place — the handler zeroes
and `deinit()`s both `PWMOut` objects and rebuilds them at the new frequency, preserving only the
enable states. Out-of-range or malformed values are silently ignored, so the host-side validators
in `piezo_protocol.py` are the only thing that turns a bad request into an error.

### DEADMAN — 600 s, requires a new command to clear

- `TO = 10 * 60` seconds. Every accepted command (`a`/`b`/`f`/`w`/`r`) refreshes `last` and clears
  the `standby` flag.
- After **600 s of command silence**, both channel states are zeroed and `standby` is set: the
  main loop then drives both duty cycles to `0` and the device idles.
- Unlike the RH deadman there is no latent value to restore — the host must issue a fresh `a1`/`b1`
  to resume. `standby` also suppresses repeated timeout trips until the next command arrives.
- Note the device sets `ds.timeout = 0` (non-blocking `readinto`) and paces itself with a 10 ms
  sleep, so the 600 s window is measured against `time.monotonic()`, not against read timeouts.

---

## Redeploying

**Copy `code.py` onto the device volume; CircuitPython detects the write and restarts the
program automatically.** Same for `boot.py`, except that `boot.py` only takes effect on a hard
reset or re-plug (it runs before `code.py`). `boot_out.txt` is generated by the device — never
copy it *to* the device.

> **Editing the copies in this repository does NOT change the device.**
>
> There is no sync, no build step, and no deploy hook. A change committed here is a change to the
> record only; the hardware keeps running whatever is on its `CIRCUITPY` volume until someone
> copies a file across. **Deployment is a manual operator act, and it is hardware actuation** — the
> device restarts mid-loop, which drops valve or piezo drive at an arbitrary moment. Do not
> redeploy while a campaign is running.

If the repo copy and the device ever disagree, **the device is the truth about what is running**
and this directory is stale. Re-copy and re-hash rather than assuming.

---

## Provenance

This check-in closes out the retraction filed as **SESSION_MAIL `[a70]`** (2026-08-19), which
corrected the earlier documented claim that no firmware deadman existed. That claim originated as
"the firmware is not in the repository" and degraded into "the mechanism does not exist" — a slide
that was possible precisely *because* the firmware was unversioned and therefore unreadable
alongside the host code. Versioning it here removes the conditions that produced the error: the
deadman constants above are now checkable against the code sitting next to them.

Task **T9.2c**, claimed in SESSION_MAIL `[a71]` §1, operator-approved.
