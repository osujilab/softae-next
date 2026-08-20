## Testing the Adafruit Trinket M0 for humidity control (DAC0 drive in My Computer).
 
## This is run remotely on the Trinket. 
# CDC serial connection is used to send a float value between 0 and 1 to set the humidity level.

import time
import board
import usb_cdc
import pwmio

datastream = usb_cdc.data
datastream.timeout = 0.75

# Setup pins connected to Aalborg PSV drivers:
pwmpin0 = pwmio.PWMOut(board.A3, frequency=10000, duty_cycle=0)
pwmpin1 = pwmio.PWMOut(board.A4, frequency=10000, duty_cycle=0)

#assuming 10 kHz, bounding voltages for full range operation:
# 0 is humid air, 1 is dry air
# 0: low bound: 0.0001, start: 1.7, high bound: 3.3 (may over-bubble in flask above 3)
# 1: low bound: 0.0001, start: 1.2, high: 3.3

try_counter = 0
ctrl_timeout = 20 #if no new serial event in this many tries, revert to ctrl_latent
ctrl = 0.01  # (autoupdated by control code)
ctrl_latent = 0.01

V0_range = [1.4, 2.5] #humidity signal range (optimized for uniform response across control value)
V1_range = [1.15,2.7] #dry air signal range (optimized for uniform response across control value)

# Trying a simple linear combination of the two voltage signal ranges.

V0tot = 3.33
V1tot = 3.33

while True:
    if try_counter >= ctrl_timeout:
        ctrl = 0
        ctrl_latent = 0

    try:
        a = datastream.readline(-1)
        ctrl = float(a)
        print(f'setting to {ctrl}')
        ctrl_latent = ctrl

        try_counter = 0
    
    except:
        ctrl = ctrl_latent
        print(f"no value received, remaining at {ctrl}")
        try_counter += 1
        
    finally:
        if ctrl == 0: #auto shutoff at 0
            V0targ = 0.0001
            V1targ = 0.0001
        else:
            V0targ = (V0_range[1] - V0_range[0]) * (ctrl) + V0_range[0]
            V1targ = (V1_range[1] - V1_range[0]) * (1-ctrl) + V1_range[0]

        DC0 = V0targ/V0tot
        DC1 = V1targ/V1tot

        pwmpin0.duty_cycle = int(DC0 * 65535)  
        pwmpin1.duty_cycle = int(DC1 * 65535)
        time.sleep(0.4)

        pwmpin0.duty_cycle = int(DC0 * 65535 / 1.5) # a little brake on the humid air to prevent over-bubbling
        pwmpin1.duty_cycle = int(DC1 * 65535 / 2)
        time.sleep(0.1)