import h5py
import numpy as np

data_path = "C:\\Users\\wml\\Downloads\\scsn_ps_2000_2017_shuf.hdf5"

with h5py.File(data_path,'r') as source_data:
    print(source_data.keys())
    print(len(source_data['X']))
    source_data_X = source_data['X'][1000]
    source_data_Y = source_data['Y'][1000]

    data_X = np.array(source_data_X, dtype=float)
    data_Y = np.array(source_data_Y, dtype=int)

    print(data_X.shape)
    print(data_Y)