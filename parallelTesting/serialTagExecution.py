import serial
import numpy as np
import time
from matplotlib import pyplot as plt
from ribbn_scripts.hardware_api.hardware import Exciter,Tag



com_port2="/dev/tty.usbserial-2120"
com_port1="/dev/tty.usbserial-2130"

tag1=Tag(com_port1)
print(tag1.get_mac())

tag2=Tag(com_port2)
print(tag2.get_mac())

time_per_phase=100/1000 #s
num_loops=5

t_start=time.time()
plt.figure(figsize=(20,10))
for rep in range(num_loops):
    plt.subplot(2,3,rep+1)
    print("Begin reading")     
    tag1.begin_reading()

    print("Starting MPP")     
    mpp_start_time,mpp_stop_time=tag2.perform_mpp()

    print("Reading response")
    voltage_readings=tag1.stop_reading()
    print("Num voltage samples:",len(voltage_readings))
    mpp_time_elapsed=mpp_stop_time-mpp_start_time
    plot_all_time=np.arange(0,mpp_time_elapsed,mpp_time_elapsed/len(voltage_readings))
    plot_end_time=plot_all_time[-1]
    ver_lines=[]
    for i in range(6):
        ver_lines.append(plot_end_time-time_per_phase*(i+1))

    for v in ver_lines:
        plt.axvline(x = v, color = 'b', label = 'axvline - full height')

    plt.plot(np.arange(0,mpp_time_elapsed,mpp_time_elapsed/len(voltage_readings))[:len(voltage_readings)],voltage_readings)

    plt.xlabel("Time [s]")
    plt.ylabel("ADC out [mV]")
t_end=time.time()
print(f"Time elapsed for {num_loops} loops: {t_end-t_start}")

plt.show()

tag1.disconnect()
tag2.disconnect()