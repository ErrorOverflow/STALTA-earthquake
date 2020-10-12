import pandas as pd
import glob
import numpy as np
import h5py

def main(inp_name):
    output_merge = inp_name+'.csv'
    frames = []
    for csvname in glob.glob('/media/wml/新加卷/地震数据集/STEAD/chunk*.csv'):
        print('working on '+csvname+'...')
        df = pd.read_csv(csvname)
        frames.append(df)
    
    all_csv = pd.concat(frames)
    all_csv.to_csv(output_merge)

if __name__ == '__main__':
  #  inp_name=input("Please enter a name for the output files!")
    inp_name = "/media/wml/新加卷/flushSTEAD/merged"
    main(inp_name)
