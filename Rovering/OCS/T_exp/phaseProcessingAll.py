import os
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
folders = [os.path.join(current_dir, d) for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

for folder in folders:
    dataframes_dir = os.path.join(folder, "dataframes")
    if not os.path.exists(dataframes_dir):
        print(f"Skipping {folder} — no dataframes/ directory")
        continue

    print(folder)

    df_all = None
    file_names = sorted([f for f in os.listdir(dataframes_dir) if f[0] != '.'], key=lambda x: int(x.split('.')[0]))
    for file_name in file_names:
        exp_num = int(file_name.split(".")[0])
        with open(os.path.join(dataframes_dir, file_name), "rb") as f:
            print(file_name)
            _df = pickle.load(f)
            _df['Run Exp Num'] = exp_num
            if df_all is None:
                df_all = _df
            else:
                df_all = pd.concat([df_all, _df], ignore_index=True)

    time_per_phase = 10/1000  # s
    num_phases = 6
    default_MPPs = 5
    num_mpps = df_all.iloc[0]['NumMPPs'] * default_MPPs
    sampling_rate = 1000
    plotting = False
    ver_lines = [0] + [(time_per_phase - time_per_phase * 0.01) * sampling_rate * (i + 1) for i in range(num_phases * num_mpps)]

    processedDF = pd.DataFrame(columns=["Rx", "Tx", "Voltages (mV)", "Phase1", "Phase3", "Phase4", "Phase6", "Phase7", "Phase8",
                                         "Frequency (MHz)", "Run Exp Num", "NumMPPs"])
    processedDF_aggregated = pd.DataFrame(columns=["Rx", "Tx", "phase", "median", "std", "freq", "dist", 'delta', "Experiment Number", "Unique Exp Number"])

    if plotting:
        plt.figure(figsize=(20, 90))

    unique_exp_no = 0
    phase_order = [1, 3, 4, 6, 7, 8]

    for df_idx in range(len(df_all)):
        voltages = df_all.iloc[df_idx]['Voltages (mV)']
        phase_medians = {1: [], 3: [], 4: [], 6: [], 7: [], 8: []}
        phase_all     = {1: [], 3: [], 4: [], 6: [], 7: [], 8: []}

        for idx, v in enumerate(ver_lines):
            if idx < len(ver_lines) - 1:
                phase_medians[phase_order[int(idx % num_phases)]].append(np.median(voltages[int(ver_lines[idx]):int(ver_lines[idx + 1])]))
                phase_all[phase_order[int(idx % num_phases)]].append(voltages[int(ver_lines[idx]):int(ver_lines[idx + 1])])
            if plotting:
                plt.subplot(len(df_all) // 3 + 1, 3, df_idx + 1)
                if idx % num_phases == 0:
                    plt.axvline(x=v, color='b', linestyle='-')
                else:
                    plt.axvline(x=v, color='r', linestyle='--')

        for phase_out_idx in range(len(phase_medians[1])):
            entry = {
                "Rx": df_all.iloc[df_idx]["Rx"],
                "Tx": df_all.iloc[df_idx]["Tx"],
                "Voltages (mV)": df_all.iloc[df_idx]["Voltages (mV)"],
                "Phase1": phase_all[1][phase_out_idx],
                "Phase3": phase_all[3][phase_out_idx],
                "Phase4": phase_all[4][phase_out_idx],
                "Phase6": phase_all[6][phase_out_idx],
                "Phase7": phase_all[7][phase_out_idx],
                "Phase8": phase_all[8][phase_out_idx],
                "Frequency (MHz)": df_all.iloc[df_idx]["Frequency (MHz)"],
                "Run Exp Num": df_all.iloc[df_idx]["Run Exp Num"],
                "Dist (m)": 1.185,
                "NumMPPs": df_all.iloc[df_idx]["NumMPPs"],
            }
            processedDF = pd.concat([processedDF, pd.DataFrame([entry])], ignore_index=True)

        for phase in phase_order:
            entry_aggregated = {
                "Rx": df_all.iloc[df_idx]["Rx"],
                "Tx": df_all.iloc[df_idx]["Tx"],
                "phase": str(phase),
                "median": np.mean(phase_medians[phase]),
                "std": np.std(phase_medians[phase]),
                "freq": df_all.iloc[df_idx]["Frequency (MHz)"] * 1e6,
                "dist": 1.185,
                'delta': max(phase_medians[phase]) - min(phase_medians[phase]),
                "Experiment Number": df_all.iloc[df_idx]["Run Exp Num"],
                "Unique Exp Number": unique_exp_no,
                "allVoltages": phase_medians[phase],
            }
            processedDF_aggregated = pd.concat([processedDF_aggregated, pd.DataFrame([entry_aggregated])], ignore_index=True)

        unique_exp_no += 1

        if plotting:
            plt.plot(voltages, '.')
            plt.title(f'{df_idx}. Freq: {df_all.iloc[df_idx]["Frequency (MHz)"]}; Tx:{df_all.iloc[df_idx]["Tx"]}, Rx:{df_all.iloc[df_idx]["Rx"]}')
            plt.ylim([np.percentile(voltages, 1), np.percentile(voltages, 99)])
            plt.tight_layout()

    if plotting:
        plt.savefig(os.path.join(folder, "MPPs.pdf"))
        plt.close()

    with open(os.path.join(folder, "processedDF.pkl"), 'wb') as f:
        pickle.dump(processedDF, f)

    print(f"  Saved processedDF with {len(processedDF)} rows to {folder}/processedDF.pkl")
