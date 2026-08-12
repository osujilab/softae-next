import time
from piezoControl_sender import TrinketLetterPairInstrument

def run_pwm_cycle_test():
    # Initialize the instrument. 
    # Change "COM16" to your specific port if needed (e.g., "/dev/ttyACM0" on Linux/Mac)
    PORT = "COM16" 
    
    print(f"Connecting to Trinket on {PORT}...")
    instrument = TrinketLetterPairInstrument(port=PORT)
    
    try:
        print("\n--- Starting PWM State Cycling Test ---")
        print("Press Ctrl+C at any time to abort safely.\n")
        
        # Step 1: Clear the board to a known state
        print("Step 1: Setting initial standby state (all channels OFF)...")
        instrument.standby()
        time.sleep(5)
        
        # Step 2: Cycle Channel A ON, Channel B OFF
        print("Step 2: Activating Channel A only...")
        instrument.set_channel("A", enabled=True)
        instrument.set_channel("B", enabled=False)
        time.sleep(5)
        
        # Step 3: Cycle Channel A OFF, Channel B ON
        print("Step 3: Activating Channel B only...")
        instrument.set_channel("A", enabled=False)
        instrument.set_channel("B", enabled=True)
        time.sleep(5)
        
        # Step 4: Turn both channels ON
        print("Step 4: Activating both channels (A and B ON)...")
        instrument.set_channel("A", enabled=True)
        instrument.set_channel("B", enabled=True)
        time.sleep(5)
        
        # Step 5: Test the raw string command parser method
        print("Step 5: Testing raw command string interface ('A0', 'B0')...")
        instrument.send_command("A0")
        instrument.send_command("B0")
        time.sleep(5)
        
        print("\nCycling test completed successfully!")

    except KeyboardInterrupt:
        print("\n[-] Test interrupted by user.")
        
    finally:
        # The finally block guarantees the hardware won't be left stuck in an active state
        print("\n--- Cleaning Up ---")
        print("Putting channels into standby and closing serial port connection...")
        instrument.standby()
        instrument.close()
        print("Safe to disconnect.")

if __name__ == "__main__":
    run_pwm_cycle_test()