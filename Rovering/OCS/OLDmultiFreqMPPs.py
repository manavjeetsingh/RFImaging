import serial
import pandas as pd
from ribbn_scripts.hardware_api.hardware import Exciter,Tag
import numpy as np
import time
import pickle
import os
from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ribbn_scripts.hardware_api.hardware import Exciter,Tag
from pydantic import BaseModel
import uvicorn
import os
from multiprocessing import Process

app = FastAPI()


SLEEPTIME=0.05
READTIME=5

# Default Settings
REPETITIONS=1
FREQ_RANGE=list(range(775,1005,10))

# Setting up the exciter
EXC_POWER=12.9
exc = Exciter()
exc.set_freq(915)
exc.set_pwr(EXC_POWER)

# Connecting to Tags
TAG1=Tag("COM3")
TAG2=Tag("COM6")


# FOLDER_PATH="C:/git/T2TExperiments/DistExperiments/EstimatingDistNCS330Feb2_1"
FOLDER_PATH=os.getcwd()

def MPP(Rx,Tx):
    """
        @args: Tx, Rx: Tag type objects.
    
        - Set Rx to receiving state.
        - Go through all phases of Tx (includes non-reflecting (ch5) and receiving (ch2), for completeness.
        
        @returns a dictionary of "phase":"voltage at Rx" mappings.
    """

    Rx.reflect("ch_2\0\n".encode())
    vals={}
    for ph in ["ch_5\0\n","ch_2\0\n","ch_1\0\n","ch_3\0\n","ch_4\0\n","ch_6\0\n","ch_7\0\n","ch_8\0\n"]:
        Tx.reflect(ph.encode())
        start_time=time.time()
        vals[ph]=[]
        while time.time()-start_time<READTIME:
            vals[ph]+=(time.time(),Rx.get_adc_val())
    return vals



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


def main(run_exp_num, freq_range=FREQ_RANGE, repetitions=REPETITIONS):
    exc.set_pwr(EXC_POWER)
    exp_no=getExperimentNo()
    t_start=time.time()
    premature_stop=0
    premature_stop_error=""
    FREQS_DONE=[]
    DF=pd.DataFrame(columns=["Rx","Tx", "MPP Start Time (s)",
                              "MPP Stop Time (s)","Voltages (mV)",
                                "Frequency (MHz)", "Run Exp Num"])
    DF_SNAPSHOP=DF


    try:
        for freq in freq_range:
            exc.set_freq(freq)
            print("FREQ:",freq)
            for rep_no in range(repetitions):
                print("Rep:",rep_no)

                # On tag MPP RX=T1, TX=T2
                TAG1.begin_reading()
                mpp_start_time_1,mpp_stop_time_1=TAG2.perform_mpp()
                voltage_readings_1=TAG1.stop_reading()
                entry={"Rx":"Tag1", 
                    "Tx":"Tag2", 
                    "MPP Start Time (s)":mpp_start_time_1, 
                    "MPP Stop Time (s)":mpp_stop_time_1, 
                    "Voltages (mV)":voltage_readings_1,
                    "Frequency (MHz)":freq, 
                    "Run Exp Num":run_exp_num
                    }
                DF=pd.concat([DF,pd.DataFrame([entry])],ignore_index=True)

                # On tag MPP RX=T2, TX=T1
                TAG2.begin_reading()
                mpp_start_time_2,mpp_stop_time_2=TAG1.perform_mpp()
                voltage_readings_2=TAG2.stop_reading()
                entry={"Rx":"Tag2", 
                    "Tx":"Tag1", 
                    "MPP Start Time (s)":mpp_start_time_2, 
                    "MPP Stop Time (s)":mpp_stop_time_2, 
                    "Voltages (mV)":voltage_readings_2,
                    "Frequency (MHz)":freq, 
                    "Run Exp Num":run_exp_num
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

class ExpParams(BaseModel):
    run_exp_num: int
    freq_range_start: int
    freq_range_stop: int
    freq_range_interval: int
    repetitions: int

@app.post("/MPP")
def MPPNetReq(conf: ExpParams):
    freq_range=np.arange(conf.freq_range_start,
                        conf.freq_range_stop,
                        conf.freq_range_interval)
    err=main(run_exp_num=conf.run_exp_num,
         freq_range=freq_range,
         repetitions=conf.repetitions)
    
    return {"Error Encountered": err}


@app.get("/test")
def test():
    # Pre-testing tags
    print("TESTING")
    print(f"Tag 1 mac:{TAG1.get_mac()}")
    print(f"Tag 2 mac:{TAG2.get_mac()}")

    time.sleep(SLEEPTIME)

    print("Changing phase to 1.")
    TAG1.reflect("ch_1\0\n".encode())
    TAG2.reflect("ch_1\0\n".encode())

    time.sleep(SLEEPTIME)
    T1_ph1_exc=np.median(TAG1.get_adc_val())
    T2_ph1_exc=np.median(TAG2.get_adc_val())
    print(f"Tag 1 excitation:{T1_ph1_exc}")
    print(f"Tag 2 excitation:{T2_ph1_exc}")

    time.sleep(SLEEPTIME)

    print("Changing phase to 2.")
    TAG1.reflect("ch_2\0\n".encode())
    TAG2.reflect("ch_2\0\n".encode())

    time.sleep(SLEEPTIME)

    T1_ph2_exc=np.median(TAG1.get_adc_val())
    T2_ph2_exc=np.median(TAG2.get_adc_val())
    print(f"Tag 1 excitation:{T1_ph2_exc}")
    print(f"Tag 2 excitation:{T2_ph2_exc}")
    
    return {"Test": "done"}
    
@app.get("/ping")
def ping():
    return {"status": "good"}


# def run_server():
#     print("Starting Uvicorn server")
#     uvicorn.run("multiFreqMPPs:app", host="0.0.0.0", port=8001, reload=True)

# if __name__=="__main__":

#     server_process=Process(target=run_server())

    

    

#     while True:
#         time.sleep(200)

    




