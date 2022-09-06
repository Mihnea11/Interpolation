import os
import numpy
import pandas
import pathlib

import independent_function_folder.generate_data as generate_data
import independent_function_folder.data_prediction as data_prediction
import independent_function_folder.create_data_files as create_data_files


def initial_setup():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = generate_data.generate_raw_data(current_path)
    raw_copy = raw_data.copy()

    create_data_files.create_input_file(current_path, raw_copy)
    raw_copy = raw_data.copy()
    create_data_files.create_target_file(current_path, raw_copy)


def function_independence_interpolation_CMAGEP():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'), index_col=0)
    predicted = data_prediction.predict_dependence_data_CMAGEP(current_path, raw_data)
    success_rate = numpy.corrcoef(predicted['0'], predicted['5'])

    return success_rate


def function_independence_interpolation_SVM():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'), index_col=0)
    predicted = data_prediction.predict_dependence_data_SVM(current_path, raw_data)
    success_rate = numpy.corrcoef(predicted['0'], predicted['5'])

    return success_rate
