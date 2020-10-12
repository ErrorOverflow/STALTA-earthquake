import pandas as pd
import h5py
import numpy as np


def judge(x):
    try:
        tmp = float(x)
        if tmp < 30:
            return True
        else:
            return False
    except ValueError:
        return False

write_hdf_path = "/media/wml/新加卷/flushSTEAD/all.hdf5"
write_csv_path = "/media/wml/新加卷/flushSTEAD/all.csv"

for num in range(2, 4):
    read_hdf_path = "/media/wml/新加卷/地震数据集/STEAD/chunk" + \
        str(num)+"/chunk"+str(num)+".hdf5"
    read_csv_path = "/media/wml/新加卷/地震数据集/STEAD/chunk" + \
        str(num)+"/chunk"+str(num)+".csv"
    df = pd.read_csv(read_csv_path)

    df = df[(df.trace_category == 'earthquake_local') & (
        df.source_distance_km <= 300) & (df.source_magnitude < 5)]
    print(f'total events selected: {len(df)}')
    df = df.loc[df.source_depth_km.apply(lambda x: judge(x))]

    ev_list = df['trace_name'].to_list()
    dtfl = h5py.File(read_hdf_path, 'r')
    fd = h5py.File(write_hdf_path, 'a')

    for c, evi in enumerate(ev_list):
        dataset = dtfl.get('data/'+str(evi))
        fd.create_dataset('data/'+str(evi), data=dataset)
    
    fd.flush()
