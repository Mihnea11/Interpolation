import os
import pandas


def remove_nan_rows(raw_data):
    drops = []
    for i, element in enumerate(raw_data[0]):
        if element != element:
            drops.append(i)
    raw_data.drop(drops, axis=0, inplace=True)


def create_input_file(path, raw_data: pandas.DataFrame):
    new_data = raw_data

    remove_nan_rows(new_data)
    new_data.drop(5, axis=1, inplace=True)
    new_data.drop(0, axis=1, inplace=True)

    new_data.to_csv(os.path.join(path, r'input.csv'), index=False)


def create_target_file(path, raw_data: pandas.DataFrame):
    new_data = raw_data

    remove_nan_rows(raw_data)
    new_data.drop(new_data.iloc[:, 0:5], axis=1, inplace=True)

    new_data.to_csv(os.path.join(path, r'target.csv'), index=False)
