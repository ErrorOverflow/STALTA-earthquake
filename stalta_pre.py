import pandas as pd
import h5py
import numpy as np
import obspy
from keras import backend as K
import matplotlib.pyplot as plt
from obspy.signal.trigger import recursive_sta_lta, classic_sta_lta, trigger_onset, ar_pick, recursive_sta_lta

import os
os.environ['KMP_WARNINGS'] = 'off'

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'
        tmp = y_true * y_pred
        print(tmp)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def judge(x):
    try:
        tmp = float(x)
        if tmp < 30:
            return True
        else:
            return False
    except ValueError:
        return False

def main():
    file_name = "/media/wml/新加卷/flushSTEAD/merged.hdf5"
    csv_file = "/media/wml/新加卷/flushSTEAD/merged.csv"
    sta_window = 30
    lta_window = 120
    on_trigger = 1.4
    off_trigger = 0.7

    # reading the csv file into a dataframe:
    df = pd.read_csv(csv_file)
    print(f'total events in csv file: {len(df)}')
    # filterering the dataframe
    df = df[((df.trace_category == 'earthquake_local') & (
        df.source_distance_km <= 300) & (df.source_magnitude < 5))]  # | (df.trace_category == 'noise')
    df = df.loc[df.source_depth_km.apply(lambda x: judge(x))]
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
        # if len(pre_E) != 0 or len(pre_N) != 0:
        try:
            E_end_time = pre_E[-1][1]
        except:
            E_end_time = 6000
        try:
            N_end_time = pre_N[-1][1]
        except:
            N_end_time = 6000

        end_time = (E_end_time + N_end_time) / 2

        p_pick, s_pick = ar_pick(data[:, 0], data[:, 1], data[:, 2], 100,
                                1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
        p_pick, s_pick = p_pick * 100, s_pick * 100

        y_true = [float(dataset.attrs['p_arrival_sample']),
                float(dataset.attrs['s_arrival_sample']), float(dataset.attrs['coda_end_sample'][0][0])]
        y_pred = [p_pick, s_pick, end_time]
        a = np.array(y_true)
        b = np.array(y_pred)
        print(a * b)
        break
        def show_result():
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
            plt.savefig("./predict_pic/merged/"+str(c)+".png")
            plt.close()

if __name__ == "__main__":
    main()
