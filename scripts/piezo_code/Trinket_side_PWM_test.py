
import time
import board
import pwmio

# Initialize PWM on an available pin (e.g., D0) at a safe 1 kHz starting frequency
# A 50% duty cycle (32768 out of 65535) creates a perfect square wave
piezo_pwm = pwmio.PWMOut(board.A3, frequency=1000, duty_cycle=32768, variable_frequency=True)

print("Starting frequency sweep up to the DRV2700 limit...")

while True:
    # Set the frequency
        for freq in [100, 250, 500]:
            piezo_pwm.frequency = freq
            piezo_pwm.duty_cycle = 32768
            time.sleep(2) # Hold at the current frequency for 2 seconds
            print(f"Current Frequency: {freq} Hz")
            piezo_pwm.duty_cycle = 0
            time.sleep(3) # Rest for 3 seconds


# while True:
#     # Set the frequency
#         freq = 500
#         piezo_pwm.frequency = freq
#         print(f"Current Frequency: {freq} Hz")
#         time.sleep(2) # Pause before restarting
        