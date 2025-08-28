import serial
import numpy as np
import time
import multiprocessing
from matplotlib import pyplot as plt

from ribbn_scripts.hardware_api.hardware import Exciter,Tag

## ------------------------------------------------------------------
## Multiprocessing Logic
## ------------------------------------------------------------------

def device_worker(com_port, tag_id, command_queue, result_queue):
    """
    A worker function to be run in a separate PROCESS. It instantiates its
    own Tag object to avoid sharing non-serializable objects.
    """
    print(f"Process for Tag {tag_id} on {com_port} started.")
    # Each process creates its own instance of the Tag class
    tag_instance = Tag(com_port)
    
    while True:
        try:
            command = command_queue.get()

            if command == "STOP":
                tag_instance.disconnect()
                print(f"Process for Tag {tag_id} stopping.")
                break
            
            if command == "get_mac":
                result = tag_instance.get_mac()
                result_queue.put((tag_id, "mac", result))
            elif command == "begin_reading":
                tag_instance.begin_reading()
            elif command == "perform_mpp":
                result = tag_instance.perform_mpp()
                result_queue.put((tag_id, "mpp_times", result))
            elif command == "stop_reading":
                result = tag_instance.stop_reading()
                result_queue.put((tag_id, "voltage_readings", result))

        except Exception as e:
            print(f"🛑 ERROR in process for Tag {tag_id} ({com_port}): {e}")
            continue

## ------------------------------------------------------------------
## Main Application
## ------------------------------------------------------------------

if __name__ == "__main__":
    # This check is crucial for multiprocessing to prevent runaway processes
    com_port1 = "/dev/tty.usbserial-2130"
    com_port2 = "/dev/tty.usbserial-2120"
    

    # Create queues from the multiprocessing module
    cmd_q1 = multiprocessing.Queue()
    cmd_q2 = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    
    # Create Process objects. Note we pass the com_port string, NOT a Tag object.
    process1 = multiprocessing.Process(target=device_worker, args=(com_port1, 1, cmd_q1, result_q), daemon=True)
    process2 = multiprocessing.Process(target=device_worker, args=(com_port2, 2, cmd_q2, result_q), daemon=True)

    # Start the child processes
    process1.start()
    process2.start()

    # --- Get MAC addresses ---
    print("\nRequesting MAC addresses from devices...")
    cmd_q1.put("get_mac")
    cmd_q2.put("get_mac")

    mac_results = {}
    while len(mac_results) < 2:
        tag_id, res_type, data = result_q.get()
        if res_type == 'mac':
            print(f"✅ Main process received: MAC for Tag {tag_id} is {data}")
            mac_results[tag_id] = data

    t_start=time.time()
    # --- Main Experiment Loop ---
    time_per_phase = 10/1000  # s
    num_phases=6
    plt.figure(figsize=(20, 10))
    num_loops=5
    num_mpps=5

    for rep in range(num_loops):
        print(f"\n--- Starting Repetition {rep+1}/5 ---")
        plt.subplot(2, 3, rep + 1)

        print("Main: Sending 'begin_reading' to Tag 1 and 'perform_mpp' to Tag 2.")
        cmd_q1.put("begin_reading")
        cmd_q2.put("perform_mpp")

        mpp_done = False
        while not mpp_done:
            tag_id, res_type, data = result_q.get()
            if res_type == "mpp_times":
                mpp_start_time, mpp_stop_time = data
                print("Main: Received MPP completion from Tag 2.")
                mpp_done = True
        
        print("Main: Sending 'stop_reading' to Tag 1.")
        cmd_q1.put("stop_reading")
        
        voltage_readings = None
        while voltage_readings is None:
            tag_id, res_type, data = result_q.get()
            if res_type == "voltage_readings":
                voltage_readings = data
                print(f"Main: Received {len(voltage_readings)} voltage readings from Tag 1.")
        # --- Plotting Logic ---
        # voltage_readings=voltage_readings[:int(time_per_phase*Tag.observedSamplingRate*num_phases*num_mpps)]
        mpp_time_elapsed = len(voltage_readings)/Tag.observedSamplingRate
        if voltage_readings is not None and len(voltage_readings) > 0:
            plot_time_axis = np.linspace(0, mpp_time_elapsed, len(voltage_readings))
            # ver_lines = [time_per_phase * (i + 1) for i in range(num_phases*num_mpps)]

            #Adding 1% error in time per phase
            ver_lines = [ (time_per_phase-time_per_phase*0.01) * Tag.observedSamplingRate * (i + 1) for i in range(num_phases*num_mpps)]
            
            for v in ver_lines:
                if v > 0: plt.axvline(x=v, color='b', linestyle='--')
            # plt.plot(plot_time_axis, voltage_readings, color='r', marker='.', markersize=2, linestyle='-.')
            plt.plot(np.arange(len(voltage_readings)), voltage_readings, color='r', marker='.', markersize=5, linestyle='-')
            plt.title(f"Repetition {rep+1}")
            plt.xlabel("Time [s]")
            plt.ylabel("ADC out [mV]")
            plt.grid(True)
        else:
            print("No voltage readings to plot for this repetition.")
            plt.title(f"Repetition {rep+1} (No Data)")

    t_end=time.time()
    print(f"Time elapsed for {num_loops} loops: {t_end-t_start}")
    
    
    # --- Cleanup ---
    print("\nExperiment finished. Cleaning up processes...")
    cmd_q1.put("STOP")
    cmd_q2.put("STOP")

    # Wait for processes to finish
    process1.join(timeout=2)
    process2.join(timeout=2)
    
    print("Processes have been joined. Showing plot.")
    plt.tight_layout()
    plt.show()