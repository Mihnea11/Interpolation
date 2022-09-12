import os
import math
import numpy
import pandas
import pathlib
import dependent_function_folder.generate_data as generate_data
import dependent_function_folder.data_prediction as data_prediction
import dependent_function_folder.create_data_files as create_data_files

from sklearn.metrics import r2_score


def initial_setup():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = generate_data.generate_raw_data(current_path)
    raw_copy = raw_data.copy()

    create_data_files.create_input_file(current_path, raw_copy)
    raw_copy = raw_data.copy()
    create_data_files.create_target_file(current_path, raw_copy)


def function_dependence_interpolation_CMAGEP():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'), index_col=0)
    predicted = data_prediction.predict_dependence_data_CMAGEP(current_path, raw_data)
    success_rate = numpy.corrcoef(predicted['0'], predicted['5'])

    return find_success_rate(predicted)


def function_dependence_interpolation_SVM():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'), index_col=0)
    predicted = data_prediction.predict_dependence_data_SVM(current_path, raw_data)
    # success_rate = numpy.corrcoef(predicted['0'], predicted['5'])

    return find_success_rate(predicted)


def function_dependence_interpolation_KNN():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'), index_col=0)
    predicted = data_prediction.predict_dependence_data_KNN(current_path, raw_data)
    # success_rate = numpy.corrcoef(predicted['0'], predicted['5'])

    return find_success_rate(predicted)


def function_dependence_interpolation_RandomForest():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'), index_col=0)
    predicted = data_prediction.predict_dependence_data_RandomForest(current_path, raw_data)
    # success_rate = numpy.corrcoef(predicted['0'], predicted['5'])

    return find_success_rate(predicted)


def find_rmse(prediction: pandas.DataFrame):
    y_actual = prediction['0']
    y_predicted = prediction['5']

    rmse = 0.0

    i = 0
    for i in range(len(y_actual)):
        rmse = rmse + (y_predicted[i] - y_actual[i]) ** 2
        i = i + 1

    rmse = rmse / i
    rmse = math.sqrt(rmse)

    return rmse


def find_success_rate(prediction: pandas.DataFrame):
    rate = r2_score(prediction['5'], prediction['0'])

    return rate
