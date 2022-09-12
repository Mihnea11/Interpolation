import os
import pandas


def remove_nan_rows(raw_data):
    drops = []
    column = raw_data['plasma_CA19_9']

    i = 0
    for element in column:
        if element != element:
            drops.append(i)
        i = i + 1
    raw_data.drop(drops, axis=0, inplace=True)


def create_input_file(path, raw_data: pandas.DataFrame):
    new_data = raw_data

    remove_nan_rows(new_data)
    new_data.drop('age', axis=1, inplace=True)
    new_data.drop('plasma_CA19_9', axis=1, inplace=True)

    # new_data['REG1B_log'] = numpy.log(new_data['REG1B'])
    # new_data.drop('REG1B', axis=1, inplace=True)

    new_data.to_csv(os.path.join(path, r'input.csv'), index=False)


def create_target_file(path, raw_data: pandas.DataFrame):
    new_data = raw_data

    remove_nan_rows(raw_data)
    new_data.drop('age', axis=1, inplace=True)
    new_data.drop('creatinine', axis=1, inplace=True)
    new_data.drop('LYVE1', axis=1, inplace=True)
    new_data.drop('REG1B', axis=1, inplace=True)
    new_data.drop('TFF1', axis=1, inplace=True)

    # new_data['plasma_CA19_9_log'] = numpy.log(new_data['plasma_CA19_9'])
    # new_data.drop('plasma_CA19_9', axis=1, inplace=True)
    new_data.to_csv(os.path.join(path, r'target.csv'), index=False)
