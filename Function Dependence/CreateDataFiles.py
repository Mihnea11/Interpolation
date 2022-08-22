import os
import pandas

def remove_nan_rows(rawData):
    drops = []
    for i, element in enumerate(rawData[0]):
        if element != element:
            drops.append(i)
    rawData.drop(drops, axis=0, inplace=True)


def create_input_file(path, rawData : pandas.DataFrame):
    newData = rawData

    remove_nan_rows(newData)
    newData.drop(5, axis=1, inplace=True)
    newData.drop(0, axis=1, inplace=True)

    newData.to_csv(os.path.join(path, r'input.csv'), index=False)


def create_target_file(path, rawData : pandas.DataFrame):
    newData = rawData

    remove_nan_rows(rawData)
    newData.drop(newData.iloc[:, 0:5], axis=1, inplace=True)

    newData.to_csv(os.path.join(path, r'target.csv'), index=False)
