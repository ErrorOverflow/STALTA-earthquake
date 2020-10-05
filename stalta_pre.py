import pandas as pd
import h5py
import numpy as np
import obspy
import matplotlib.pyplot as plt
from obspy.signal.trigger import recursive_sta_lta, classic_sta_lta, trigger_onset, ar_pick, recursive_sta_lta

num = "1"
file_name = "/media/wml/新加卷/地震数据集/STEAD/chunk"+num+"/chunk"+num+".hdf5"
csv_file = "/media/wml/新加卷/地震数据集/STEAD/chunk"+num+"/chunk"+num+".csv"
sta_window = 30
lta_window = 120
on_trigger = 1.4
off_trigger = 0.7

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')
# filterering the dataframe
df = df[(df.trace_category == 'earthquake_local') & (
    df.source_distance_km <= 20) & (df.source_magnitude > 3)]
print(f'total events selected: {len(df)}')

# making a list of trace names for the selected data
ev_list = df['trace_name'].to_list()

# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, 'r')
for c, evi in enumerate(ev_list):
    dataset = dtfl.get('data/'+str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)

    pre_E = trigger_onset(recursive_sta_lta(
        data[:, 0], sta_window, lta_window), on_trigger, off_trigger)
    pre_N = trigger_onset(recursive_sta_lta(
        data[:, 1], sta_window, lta_window), on_trigger, off_trigger)
    pre_Z = trigger_onset(recursive_sta_lta(
        data[:, 2], sta_window, lta_window), on_trigger, off_trigger)

    # cft = recursive_sta_lta(data[:, 0], sta_window, lta_window)
    # fig = plt.figure()
    # plt.plot(cft, 'k')
    # plt.show()  

    try:
        E_end_time = pre_E[-1][1]
    except:
        E_end_time = 6000
    try:
        N_end_time = pre_E[-1][1]
    except:
        N_end_time = 6000
    
    end_time = (E_end_time + N_end_time) / 2

    p_pick, s_pick = ar_pick(data[:, 0], data[:, 1], data[:, 2], 100,
                         1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
    p_pick, s_pick = p_pick * 100, s_pick * 100

    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight': 'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()

    pl = plt.vlines(p_pick, ymin,
                    ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_pick, ymin,
                    ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(end_time, ymin,
                    ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc='upper right',
               borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])

    ax = fig.add_subplot(312)
    plt.plot(data[:, 1], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight': 'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(p_pick, ymin,
                    ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_pick, ymin,
                    ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(end_time, ymin,
                    ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc='upper right',
               borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])

    ax = fig.add_subplot(313)
    plt.plot(data[:, 2], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight': 'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(p_pick, ymin,
                    ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_pick, ymin,
                    ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(end_time, ymin,
                    ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc='upper right',
               borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])
    # plt.show()
    plt.savefig("./predict_pic/chunk"+num+"/"+str(c)+".png")
    plt.close()
