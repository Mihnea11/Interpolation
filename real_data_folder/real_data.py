import math
import numpy
import pandas
import os.path
import pathlib
import real_data_folder.create_data_files as create_data_files
import real_data_folder.data_prediction as data_prediction

from sklearn.metrics import r2_score


def initial_setup():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    input_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'))
    raw_data = input_data[['age', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']]

    raw_copy = raw_data.copy()
    create_data_files.create_input_file(current_path, raw_copy)
    raw_copy = raw_data.copy()
    create_data_files.create_target_file(current_path, raw_copy)


def real_data_interpolation_CMAGEP():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    input_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'))
    raw_data = input_data[['age', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']]

    predicted = data_prediction.predict_real_data_CMAGEP(current_path, raw_data)

    return find_success_rate(raw_data, predicted)


def real_data_interpolation_SVM():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    input_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'))
    raw_data = input_data[['age', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']]

    predicted = data_prediction.predict_real_data_SVM(current_path, raw_data)

    return find_success_rate(raw_data, predicted)


def real_data_interpolation_KNN():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    input_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'))
    raw_data = input_data[['age', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']]

    data_prediction.predict_real_data_KNN(current_path, raw_data)

    predicted = data_prediction.predict_real_data_KNN(current_path, raw_data)

    return find_success_rate(raw_data, predicted)


def real_data_interpolation_RandomForest():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    input_data = pandas.read_csv(os.path.join(current_path, r'raw_data.csv'))
    raw_data = input_data[['age', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']]

    data_prediction.predict_real_data_KNN(current_path, raw_data)

    predicted = data_prediction.predict_real_data_RandomForest(current_path, raw_data)

    return find_success_rate(raw_data, predicted)


def find_success_rate(raw_data: pandas.DataFrame, prediction: pandas.DataFrame):
    compare = create_real_data_comparison(raw_data, prediction)
    rate = r2_score(compare[0], compare[1])

    return rate


def find_rmse(raw_data: pandas.DataFrame, prediction: pandas.DataFrame):
    compare = create_real_data_comparison(raw_data, prediction)

    y_actual = compare[0].tolist()
    y_predicted = compare[1].tolist()

    rmse = 0.0

    i = 0
    for i in range(len(y_actual)):
        rmse = rmse + (y_predicted[i] - y_actual[i]) ** 2
        i = i + 1

    rmse = rmse / i
    rmse = math.sqrt(rmse)

    return rmse


def create_real_data_comparison(raw_data: pandas.DataFrame, prediction: pandas.DataFrame):
    compare = pandas.DataFrame()

    raw_rows = raw_data['plasma_CA19_9'].tolist()
    predicted_rows = prediction['plasma_CA19_9'].tolist()

    length = len(raw_rows)
    for i in range(length):
        if not (raw_rows[i] != raw_rows[i]):
            compare.at[i, 0] = raw_rows[i]
            compare.at[i, 1] = predicted_rows[i]

    return compare
