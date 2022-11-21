import pandas as pd
from os import path as osp

def split_csv(args, path):
    data_names = ['ACC','GYRO', 'HeartBeat','GPS','GAME RV']
    index_positions = []
    df = pd.read_csv(osp.join(args.root_dir, path), header= None)

    for data_name in data_names:
        # if data_name != data_names[0]:
        index_positions.append(df[df.iloc[:,0] == data_name].index[0])

    tables = {}

    for i, index_position in enumerate(index_positions):
        if i == len(index_positions)-1:
            tables[data_names[i]] = df[index_positions[i]:]
        else:    tables[data_names[i]] = df[index_positions[i]:index_positions[i+1]]

    for data_name in data_names:

        # if tables[data_name].columns[0] != data_name:
        tables[data_name] = tables[data_name].set_axis(tables[data_name].iloc[0].to_list(), axis=1)
        tables[data_name] = tables[data_name].iloc[1:]
        tables[data_name] = tables[data_name].dropna(axis=1)
        tables[data_name].to_csv(osp.join(args.root_dir,data_name+'.csv'), index = False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.root_dir = './data/'
    path = 'test221120/3.csv'
    split_csv(args, path)