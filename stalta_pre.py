import pandas as pd
import h5py
import numpy as np
import obspy
from keras import backend as K
import matplotlib.pyplot as plt
import time
import csv
from obspy.signal.trigger import recursive_sta_lta, classic_sta_lta, trigger_onset, ar_pick

config = {
    scale = 6000,
    sta_window = 30,
    lta_window = 120,
    on_trigger = 1.4,
    off_trigger = 0.7
}


def judge(x):
    try:
        tmp = float(x)
        if tmp < 30:
            return True
        else:
            return False
    except ValueError:
        return False


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


class DataOperator():
    def __init__(self, hdfpath, csvpath, outpath):
        self.hdfpath = hdfpath
        self.csvpath = csvpath
        self.csvfile = open(out_path, 'w')
        data_writer_initial()

    def data_generator(self):
        df = pd.read_csv(csvpath)
        print(f'total events in csv file: {len(df)}')

        df = df[((df.trace_category == 'earthquake_local') & (
            df.source_distance_km <= 300) & (df.source_magnitude < 5)) | (df.trace_category == 'noise')]
        df = df.loc[df.source_depth_km.apply(lambda x: judge(x))]
        print(f'total events selected: {len(df)}')

        ev_list = df['trace_name'].to_list()
        dtfl = h5py.File(hdfpath, 'r')
        return dtfl, ev_list

    def data_writer_initial(self):
        output_writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # output_writer.writerow(['network_code', 'ID', 'earthquake_distance_km', 'snr_db', 'trace_name', 'trace_category', 'trace_start_time', 'source_magnitude', 'p_arrival_sample', 'p_status', 'p_weight', 's_arrival_sample', 's_status',
        #                         's_weight', 'receiver_type', 'number_of_detections', 'detection_probability', 'detection_uncertainty', 'P_pick', 'P_probability', 'P_uncertainty', 'P_error', 'S_pick', 'S_probability', 'S_uncertainty', 'S_error'])       
        output_writer.writerow(['p_arrival_sample', 's_arrival_sample','conda_sample_time','P_pick', 'S_pick', 'coda_end_pick'])
        csvfile.flush()
    
    def data_writer(self, p_pick, s_pick, end_time):
        output_writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow([p_pick, s_pick, end_time])
        csvfile.flush()


def predict(dtfl, ev_list):
    for c, evi in enumerate(ev_list):
        dataset = dtfl.get('data/'+str(evi))
        data = np.array(dataset)

        pre_E = trigger_onset(recursive_sta_lta(
            data[:, 0], sta_window, lta_window), on_trigger, off_trigger)
        pre_N = trigger_onset(recursive_sta_lta(
            data[:, 1], sta_window, lta_window), on_trigger, off_trigger)
        pre_Z = trigger_onset(recursive_sta_lta(
            data[:, 2], sta_window, lta_window), on_trigger, off_trigger)

        N_end_time, E_end_time = 6000, 6000
        if len(pre_E):
            E_end_time = pre_E[-1][1]

        if len(pre_N):
            N_end_time = pre_N[-1][1]

        end_time = (E_end_time + N_end_time) / 2

        p_pick, s_pick = ar_pick(data[:, 0], data[:, 1], data[:, 2], 100,
                                 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2) * 100

        # y_true = [float(dataset.attrs['p_arrival_sample']),
        #           float(dataset.attrs['s_arrival_sample']), float(dataset.attrs['coda_end_sample'][0][0])]
        # y_pred = [p_pick, s_pick, end_time]

        # p_true = np.zeros(shape=(6000,))
        # p_true[p_pick-20:p_pick+21] = 1

        # a = np.array(y_true)
        # b = np.array(y_pred)
        # print(a * b)
        # break
    return p_pick, s_pick, end_time


def show_result(p_pick, s_pick, end_time, is_save=False, is_show=False):
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
    if is_show:
        plt.show()
    if is_save:
        plt.savefig("./predict_pic/merged/"+str(c)+".png")
    plt.close()


def main(hdfpath, csvpath, out_path):
    dataOperator = DataOperator(hdfpath, csvpath, out_path)
    dtfl, ev_list = dataOperator.data_generator()
    p_pick, s_pick, end_time = predict(dtfl, ev_list, dataOperator)


if __name__ == "__main__":
    file_name = "/media/wml/新加卷/flushSTEAD/merged.hdf5"
    csv_file = "/media/wml/新加卷/flushSTEAD/merged.csv"
    out_path = "/media/wml/新加卷/flushSTEAD/result" + str(time.time()) + ".csv"
    main(file_name, csv_file, out_path)
