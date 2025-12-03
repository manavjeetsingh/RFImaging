import pandas as pd
from matplotlib import pyplot as plt
from ribbn_scripts.hardware_api.hardware import *
from ribbn_scripts.ref_functions.spec_functions import *
from ribbn_scripts.ref_functions.util_functions import *
import numpy as np
import json

fn = 'v32-3'
# fn = 'tag5'
ch_list = [b'ch_1\0\n', b'ch_2\0\n', b'ch_3\0\n', b'ch_4\0\n', b'ch_5\0\n', b'ch_6\0\n', b'ch_7\0\n', b'ch_8\0\n']

ch_ig = []
ch_l = []
d = {}
for ch in ch_list:
    if ch in ch_ig:
        continue

    ch_l.append(ch)
    df = pd.DataFrame(
        pd.read_csv('C:/git/T2TExperiments/coba/calibrations/VNA_Dec2025/'+fn+'_channel_' + str(ch[0:4]) + '_vna_pwr_' + str(15) + '.csv'))
        # pd.read_csv('C:/git/T2TExperiments/coba/calibrations/VNA_Oct2024/'+fn+'_channel_' + str(ch[0:4]) + '_vna_pwr_' + str(15) + '.csv'))
    df = df[(700e6 < df['Frequency'])]
    df = df[(1000e6 > df['Frequency'])]
    freq = df['Frequency']
    phase = df[' Formatted Data.1']
    amp = df[' Formatted Data']
    p = np.polyfit(freq, phase, 1)
    xnew = np.linspace(min(freq), max(freq), 100)
    # plt.plot(xnew,np.polyval(p,xnew))
    plt.plot(freq, phase, 'o')
    plt.ylim([-190,190])

    ds = {'freq': list(freq), 'phase': list(phase), 'amp': list(amp)}
    d[ch] = ds

# with open('../../csv_exported_data/network_analyzer/s11_v0_t2_diff_cap.json', 'w') as outfile:
#     json.dump(d, outfile)

# write_pickle('C:/Users/Manavjeet Singh/Git/T2TExperiments/coba/calibrations/VNA_Oct2024/'+fn+'_s11_poly.txt', d)
write_pickle('C:/git/T2TExperiments/coba/calibrations/VNA_Dec2025/'+fn+'_s11_poly.txt', d)
plt.legend(ch_l)
plt.xlabel('freq')
plt.ylabel('amp')
plt.show()
