import math
import os

import pandas


def create_input_file(path, rawData : pandas.DataFrame):
    newData = rawData
    drops = []
    for i, element in enumerate(newData[0]):
        if element != element:
            drops.append(i)
    newData.drop(drops, axis=0, inplace=True)
    newData.drop(5, axis=1, inplace=True)

    newData.to_csv(os.path.join(path, r'input.csv'))


def create_target_file(path, rawData : pandas.DataFrame):
    newData = rawData
    newData.drop(newData.iloc[:, 0:5], axis=1, inplace=True)

    newData.to_csv(os.path.join(path, r'target.csv'))
    