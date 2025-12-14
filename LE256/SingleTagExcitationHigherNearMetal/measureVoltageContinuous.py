import serial
import pandas as pd
from ribbn_scripts.hardware_api.hardware import Exciter,Tag
import numpy as np
import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing

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
            elif command == "get_adc_val":
                result = tag_instance.get_adc_val()
                result_queue.put((tag_id, "adc_vals", result))
            elif command[:2]=='ch':
                tag_instance.reflect(command.encode())

        except Exception as e:
            print(f"🛑 ERROR in process for Tag {tag_id} ({com_port}): {e}")
            continue

SLEEPTIME=0.5
READTIME=5

# Default Settings
INBUILT_REPETITIONS=5
FREQ_RANGE=[915]
# FREQ_RANGE=list(range(775,1005,10))

# # Setting up the exciter
EXC_POWER=6
exc = Exciter()
# exc.set_freq(915)
# exc.set_pwr(EXC_POWER)

# Connecting to Tags
# TAG1_COM="/dev/cu.usbserial-1140"
TAG1_COM="COM3"
# TAG2_COM="/dev/tty.usbserial-2120"
TAG2_COM="COM6"


# FOLDER_PATH="C:/git/T2TExperiments/DistExperiments/EstimatingDistNCS330Feb2_1"
FOLDER_PATH=os.getcwd()

cmd_q1, cmd_q2, result_q, process1, process2=None, None, None, None, None

def initialize():
    global cmd_q1, cmd_q2, result_q, process1, process2
    
    # Create queues from the multiprocessing module
    cmd_q1 = multiprocessing.Queue()
    # cmd_q2 = multiprocessing.Queue()
    result_q = multiprocessing.Queue()

    process1 = multiprocessing.Process(target=device_worker, args=(TAG1_COM, 1, cmd_q1, result_q), daemon=True)
    # process2 = multiprocessing.Process(target=device_worker, args=(TAG2_COM, 2, cmd_q2, result_q), daemon=True)

    # Start the child processes
    process1.start()
    # process2.start()

def MPP(cmdq_rx,cmdq_tx, result_q):
    """
        @args: Tx, Rx: Tag type objects.
    
        - Set Rx to receiving state.
        - Go through all phases of Tx (includes non-reflecting (ch5) and receiving (ch2), for completeness.
        
        @returns a dictionary of "phase":"voltage at Rx" mappings.
    """

    cmdq_rx.put("begin_reading")
    cmdq_tx.put("perform_mpp")
    mpp_done = False
    mpp_start_time=None
    mpp_stop_time=None
    while not mpp_done:
        tag_id, res_type, data = result_q.get()
        if res_type == "mpp_times":
            mpp_start_time, mpp_stop_time = data
            mpp_done = True
    cmdq_rx.put("stop_reading")
    voltage_readings = None
    while voltage_readings is None:
        tag_id, res_type, data = result_q.get()
        if res_type == "voltage_readings":
            voltage_readings = data

    return voltage_readings, mpp_start_time, mpp_stop_time

def getExperimentNo():
    all_files=os.listdir(f"{FOLDER_PATH}/dataframes/")
    if len(all_files)==0:
        return 0
    else:
        to_ret=0
        while True:
            if f"{to_ret}.df" in all_files:
                to_ret+=1
            else:
                break
        return to_ret





def main(run_exp_num, freq_range=FREQ_RANGE, repetitions=INBUILT_REPETITIONS):
    exc.set_pwr(EXC_POWER)
    exp_no=getExperimentNo()
    
    
    t_start=time.time()
    premature_stop=0
    premature_stop_error=""
    FREQS_DONE=[]
    DF=pd.DataFrame(columns=["Rx","Tx", "MPP Start Time (s)",
                              "MPP Stop Time (s)","Voltages (mV)",
                                "Frequency (MHz)", "Run Exp Num", "NumMPPs"])
    DF_SNAPSHOP=DF


    try:
        for freq in freq_range:
            exc.set_freq(freq)
            print("FREQ:",freq)

            voltage_readings_1, mpp_start_time_1, mpp_stop_time_1=MPP(cmdq_rx=cmd_q1, cmdq_tx=cmd_q2, result_q=result_q)
            entry={"Rx":"Tag1", 
                "Tx":"Tag2", 
                "MPP Start Time (s)":mpp_start_time_1, 
                "MPP Stop Time (s)":mpp_stop_time_1, 
                "Voltages (mV)":voltage_readings_1,
                "Frequency (MHz)":freq, 
                "Run Exp Num":run_exp_num,
                "NumMPPs": repetitions
                }
            DF=pd.concat([DF,pd.DataFrame([entry])],ignore_index=True)

            voltage_readings_2, mpp_start_time_2, mpp_stop_time_2=MPP(cmdq_rx=cmd_q2, cmdq_tx=cmd_q1, result_q=result_q)
            entry={"Rx":"Tag2", 
                "Tx":"Tag1", 
                "MPP Start Time (s)":mpp_start_time_2, 
                "MPP Stop Time (s)":mpp_stop_time_2, 
                "Voltages (mV)":voltage_readings_2,
                "Frequency (MHz)":freq, 
                "Run Exp Num":run_exp_num,
                "NumMPPs": repetitions,
                }
            DF=pd.concat([DF,pd.DataFrame([entry])],ignore_index=True)

            FREQS_DONE.append(freq)
            DF_SNAPSHOP=DF
            
       
    except Exception as e:
        # Even if there is some error while running the experiments, 
        print(DF_SNAPSHOP)
        print("Had to stop script prematurely. Had the following exception: ",e)
        premature_stop=1
        premature_stop_error=e
            
    
    save_path=f"{FOLDER_PATH}/dataframes/{exp_no}.df"
    pickle.dump(DF_SNAPSHOP,open(save_path,"wb"))
    print(f"DF saved at: {save_path}")

    metadata_save_path=f"{FOLDER_PATH}/metaData/{exp_no}.txt"
    f = open(metadata_save_path, "w")
    f.write(f"Frequencies covered: {FREQS_DONE}\n")
    if premature_stop:
        f.write(f"Premature Stop: {str(premature_stop_error)}\n")
    time_taken=time.time()-t_start
    f.write(f"Time taken: {time_taken} seconds\n")
    f.close()

    print(f"Time taken: {time_taken}")
   
    exc.set_pwr(-30)
    return premature_stop


def test():
    exc.set_pwr(EXC_POWER)
    exc.set_freq(915)
    # Pre-testing tags
    print("TESTING")
    
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

    time.sleep(SLEEPTIME)

    print("Changing phase to 1.")
    cmd_q1.put("ch_1\0\n")
    cmd_q2.put("ch_1\0\n")

    time.sleep(SLEEPTIME)
    cmd_q1.put("get_adc_val")
    cmd_q2.put("get_adc_val")
    adc_results = {}
    while len(adc_results) < 2:
        tag_id, res_type, data = result_q.get()
        if res_type == 'adc_vals':
            print(f"✅ ADC val received for tag {tag_id} is {np.median(data)}")
            adc_results[tag_id] = data

    time.sleep(SLEEPTIME)

    print("Changing phase to 2.")
    cmd_q1.put("ch_2\0\n")
    cmd_q2.put("ch_2\0\n")

    time.sleep(SLEEPTIME)
    cmd_q1.put("get_adc_val")
    cmd_q2.put("get_adc_val")
    adc_results = {}
    while len(adc_results) < 2:
        tag_id, res_type, data = result_q.get()
        if res_type == 'adc_vals':
            print(f"✅ ADC val received for tag {tag_id} is {np.median(data)}")
            adc_results[tag_id] = data

    input("Print enter to continue")
    
    return {"Test": "done"}
    
# @app.get("/ping")
def ping():
    return {"status": "good"}


def save_excitations():
    try:
        if not os.path.isfile("voltageData.csv"):
            # if file doesn't exist
            with open("voltageData.csv",'a') as f:
                    f.write(f"Exc-wall dist (cm),Tag-wall dist (cm),Tag Voltage (mV)\n")
        print("TESTING")
        
        # --- Get MAC addresses ---
        print("\nRequesting MAC addresses from devices...")
        cmd_q1.put("get_mac")

        mac_results = {}
        while len(mac_results) < 1:
            tag_id, res_type, data = result_q.get()
            if res_type == 'mac':
                print(f"✅ Main process received: MAC for Tag {tag_id} is {data}")
                mac_results[tag_id] = data

        time.sleep(SLEEPTIME)

        cmd_q1.put("ch_2\0\n")
        time.sleep(2)
        exciter_wall_dist=float(input("Exciter wall dist:"))
        while True:
            tag_wall_dist=float(input("Tag to wall dist:"))
            
            # cmd_q1.put("get_adc_val")
            # adc_results = {}
            # while len(adc_results) < 1:
            #     tag_id, res_type, data = result_q.get()
            #     if res_type == 'adc_vals':
            #         print(f"✅ ADC val received for tag {tag_id} is {np.median(data)}")
            #         adc_results[tag_id] = data

            # tag_voltage=adc_results[1]
            # with open("voltageData.csv",'a') as f:
            #     f.write(f"{exciter_wall_dist},{tag_wall_dist},{tag_voltage}\n")
            all_voltage_readings=[]
            all_voltage_readings_split=[]
            start_times=[]
            end_times=[]
            maxreps=5
            for reps in range(maxreps):
                start_time=time.time()
                cmd_q1.put("begin_reading")
                time.sleep(5)
                cmd_q1.put("stop_reading")
                end_time=time.time()
                voltage_readings = None
                while voltage_readings is None:
                    tag_id, res_type, data = result_q.get()
                    if res_type == "voltage_readings":
                        voltage_readings = data
                all_voltage_readings+=list(voltage_readings)
                all_voltage_readings_split.append(voltage_readings)
                start_times.append(start_time)
                end_times.append(end_time)
                print(len(voltage_readings))
            print(len(all_voltage_readings))
            print(min(all_voltage_readings))
            for reps in range(maxreps):
                plt.plot(np.linspace(start_times[reps]-start_times[0], end_times[reps]-start_times[0], len(all_voltage_readings_split[reps])), all_voltage_readings_split[reps],'.', ms=1)
            plt.show()
            
            info = {
                "all_voltage_readings": all_voltage_readings,
                "all_voltage_readings_split": all_voltage_readings_split,
                "start_times": start_times,
                "end_times": end_times,
            }
            with open(f"AllData/{int(time.time())}_EW{exciter_wall_dist}TW{tag_wall_dist}.pkl",'wb') as f:
                pickle.dump(info, f)
                
            with open("voltageData.csv",'a') as f:
                f.write(f"{exciter_wall_dist},{tag_wall_dist},{[min(all_voltage_readings)]}\n")
            
    finally:
        exc.set_pwr(-30)



if __name__=="__main__":
    
    initialize()
    exc.set_pwr(EXC_POWER)
    exc.set_freq(915)
    save_excitations()