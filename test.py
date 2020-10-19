import numpy as np
import pandas as pd

def create_csv():
    source = pd.read_csv("/media/wml/新加卷/flushSTEAD/merged.csv")
    source = source.drop(['network_code','receiver_code','receiver_type','receiver_latitude','receiver_longitude',
                                    'receiver_elevation_m','p_status','p_weight','p_travel_sec',
                                    's_status','s_weight',
                                    'source_id','source_origin_time','source_origin_uncertainty_sec',
                                    'source_latitude','source_longitude','source_error_sec',
                                    'source_gap_deg','source_horizontal_uncertainty_km', 'source_depth_uncertainty_km',
                                    'source_magnitude', 'source_magnitude_type', 'source_magnitude_author','source_mechanism_strike_dip_rake',
                                    'source_distance_deg', 'back_azimuth_deg', 'snr_db',
                                    'trace_start_time'], axis = 1)
    source.to_csv("/media/wml/新加卷/flushSTEAD/predict.csv")

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 1e-07)
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'
        
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + 1e-07)
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return precision, recall, 2 * ((precision * recall) / (precision + recall + 1e-07))

def array_generate(point):
    array = np.zeros(shape = (6000,))
    if point < 50:
        array[0 : point+51] = 1
    elif point > 6000 - 50:
        array[point-50 : 6000] = 1 
    else:
        array[point-50 : point+51] = 1 
    return array

def calcu():
    df = pd.read_csv("/media/wml/新加卷/flushSTEAD/result1602857324.2880619.csv")
    p_f1_total = 0
    p_pr_total = 0
    p_re_total = 0
    s_f1_total = 0
    s_pr_total = 0
    s_re_total = 0
    end_f1_total = 0
    end_re_total = 0
    end_pr_total = 0
    p_all_array = 0
    s_all_array = 0
    for index in df.index:
        p_array_t = array_generate(int(df.loc[index].values[1]))
        p_array_p = array_generate(df.loc[index].values[4])
        s_array_t = array_generate(int(df.loc[index].values[2]))
        s_array_p = array_generate(df.loc[index].values[5])

        end_array_t = np.zeros(shape = (6000,))
        end_array_t[int(df.loc[index].values[1]) : int(df.loc[index].values[3][3:-3])] = 1
        end_array_p = np.zeros(shape = (6000,))
        if df.loc[index].values[6] >= df.loc[index].values[4]:
            end_array_p[int(df.loc[index].values[4]) : int(df.loc[index].values[6])] = 1

        p_all_array += abs(df.loc[index].values[4] - int(df.loc[index].values[1]))
        s_all_array += abs(df.loc[index].values[5] - int(df.loc[index].values[2]))

        p_pr, p_re, p_f1 = f1(p_array_t, p_array_p)
        print(p_pr, p_re, p_f1)
        p_pr_total += p_pr
        p_re_total += p_re
        p_f1_total += p_f1

        s_pr, s_re, s_f1 = f1(s_array_t, s_array_p)
        s_pr_total += s_pr
        s_re_total += s_re
        s_f1_total += s_f1
        
        end_pr, end_re, end_f1 = f1(end_array_t, end_array_p)
        end_pr_total += end_pr
        end_re_total += end_re
        end_f1_total += end_f1

        # if index % 100 == 0:
        #     print(f'round {index} : p_f1_score {p_f1_total/index}, s_f1_score {s_f1_total/index}, end_f1_score {end_f1_total/index}')
        #     print(f'round {index} : p_pr_score {p_pr_total/index}, s_pr_score {s_pr_total/index}, end_pr_score {end_pr_total/index}')
        #     print(f'round {index} : p_re_score {p_re_total/index}, s_re_score {s_re_total/index}, end_re_score {end_re_total/index}')
        #     print(f'round {index} : p_mae_score {p_all_array/index/1000}, s_mae_score {s_all_array/index/1000}')
        #     print()

if __name__ == "__main__":
    calcu()
