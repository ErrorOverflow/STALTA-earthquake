import numpy as np
import pandas as pd

source = pd.read_csv("D:\\flushSTEAD\\merged.csv")
source.drop(['network_code','receiver_code','receiver_type','receiver_latitude','receiver_longitude',
                                 'receiver_elevation_m','p_status','p_weight','p_travel_sec',
                                 's_status','s_weight',
                                 'source_id','source_origin_time','source_origin_uncertainty_sec',
                                 'source_latitude','source_longitude','source_error_sec',
                                 'source_gap_deg','source_horizontal_uncertainty_km', 'source_depth_uncertainty_km',
                                 'source_magnitude', 'source_magnitude_type', 'source_magnitude_author','source_mechanism_strike_dip_rake',
                                 'source_distance_deg', 'back_azimuth_deg', 'snr_db',
                                 'trace_start_time'], axis = 1)
source.to_csv("D:\\flushSTEAD\\predict.csv")
