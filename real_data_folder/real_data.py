import os.path
import pathlib
import numpy
import pandas

import real_data_folder.create_data_files as create_data_files
import real_data_folder.data_prediction as data_prediction


def real_data_interpolation():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    input_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'))
    raw_data = input_data[['age', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']]
    raw_copy = raw_data.copy()

    create_data_files.create_input_file(current_path, raw_copy)
    raw_copy = raw_data.copy()
    create_data_files.create_target_file(current_path, raw_copy)

    predicted = data_prediction.predict_dependence_data(current_path, raw_data)

    return find_success_rate(raw_data, predicted)


def find_success_rate(raw_data: pandas.DataFrame, prediction: pandas.DataFrame):
    compare = pandas.DataFrame()

    raw_rows = raw_data['plasma_CA19_9'].tolist()
    predicted_rows = prediction['plasma_CA19_9'].tolist()

    length = len(raw_rows)
    for i in range(length):
        if not(raw_rows[i] != raw_rows[i]):
            compare.at[i, 0] = raw_rows[i]
            compare.at[i, 1] = predicted_rows[i]

    rate = numpy.corrcoef(compare[0], compare[1])
    return rate