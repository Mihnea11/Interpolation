import math
import os
import pandas
import pathlib
import matplotlib.pyplot as plt

import real_data_folder.real_data
from dependent_function_folder.dependent_function import function_dependence_interpolation
from independent_function_folder import independent_function
from real_data_folder import real_data


def main():
    current_path = pathlib.Path(__file__).parent.resolve()

    independent_function.initial_setup()
    real_data.initial_setup()

    success_rate = pandas.DataFrame()
    success_rate = CMAGEP(current_path, success_rate)
    success_rate = SVM(current_path, success_rate)

    success_rate.to_csv(os.path.join(current_path, r'success_rates.csv'), index=False)


def draw_dependence_graph(path):
    data = pandas.read_csv(os.path.join(path, r'dependent_function_folder\data\prediction.csv'))

    x = []
    y = data['0']
    y_real = data['5']

    i = 0
    for i in range(len(data.index)):
        x.append(i)
        i = i + 1

    plt.plot(x, y_real, label='generated')
    plt.plot(x, y, label='predicted')
    plt.legend()
    plt.savefig(os.path.join(path, r'data_graphs_folder\dependent_function_graph.pdf'))
    plt.close()


def draw_independence_graph_CMAGEP(path):
    data = pandas.read_csv(os.path.join(path, r'independent_function_folder\data\prediction_CMAGEP.csv'))

    x = []
    y = data['0']
    y_real = data['5']

    i = 0
    for i in range(len(data.index)):
        x.append(i)
        i = i + 1

    plt.plot(x, y_real, label='generated')
    plt.plot(x, y, label='predicted')
    plt.legend()
    plt.savefig(os.path.join(path, r'data_graphs_folder\independent_function_graph_CMAGEP.pdf'))
    plt.close()


def draw_independence_graph_SVM(path):
    data = pandas.read_csv(os.path.join(path, r'independent_function_folder\data\prediction_SVM.csv'))

    x = []
    y = data['0']
    y_real = data['5']

    i = 0
    for i in range(len(data.index)):
        x.append(i)
        i = i + 1

    plt.plot(x, y_real, label='generated')
    plt.plot(x, y, label='predicted')
    plt.legend()
    plt.savefig(os.path.join(path, r'data_graphs_folder\independent_function_graph_SVM.pdf'))
    plt.close()


def draw_real_data_graph_CMAGEP(path):
    original = pandas.read_csv(os.path.join(path, r'real_data_folder\data\raw_data.csv'))
    predicted = pandas.read_csv(os.path.join(path, r'real_data_folder\data\prediction_CMAGEP.csv'))

    compare = real_data_folder.real_data.create_real_data_comparison(original, predicted)

    x = []
    y = compare[1]
    y_real = compare[0]

    i = 0
    for i in range(len(compare.index)):
        x.append(i)
        i = i + 1

    plt.plot(x, y_real, label='original')
    plt.plot(x, y, label='predicted')
    plt.legend()
    plt.savefig(os.path.join(path, r'data_graphs_folder\real_data_graph_CMAGEP.pdf'))
    plt.close()


def draw_real_data_graph_SVM(path):
    original = pandas.read_csv(os.path.join(path, r'real_data_folder\data\raw_data.csv'))
    predicted = pandas.read_csv(os.path.join(path, r'real_data_folder\data\prediction_SVM.csv'))

    compare = real_data_folder.real_data.create_real_data_comparison(original, predicted)

    x = []
    y = compare[1]
    y_real = compare[0]

    i = 0
    for i in range(len(compare.index)):
        x.append(i)
        i = i + 1

    plt.plot(x, y_real, label='original')
    plt.plot(x, y, label='predicted')
    plt.legend()
    plt.savefig(os.path.join(path, r'data_graphs_folder\real_data_graph_SVM.pdf'))
    plt.close()


def CMAGEP(current_path, success_rates: pandas.DataFrame):
    dependence_rate = function_dependence_interpolation()
    independence_rate = independent_function.function_independence_interpolation_CMAGEP()
    real_data_rate = real_data.real_data_interpolation_CMAGEP()

    rates = {math.nan: 'CMAGEP',
             'Dependent function': [str(dependence_rate[0][1] * 100) + '%'],
             'Independent function': [str(independence_rate[0][1] * 100) + '%'],
             'Real data': [str(real_data_rate[0][1] * 100) + '%']}

    rate_dataframe = pandas.DataFrame(rates)
    success_rates = success_rates.append(rate_dataframe)

    draw_dependence_graph(current_path)
    draw_independence_graph_CMAGEP(current_path)
    draw_real_data_graph_CMAGEP(current_path)

    return success_rates


def SVM(current_path, success_rates: pandas.DataFrame):
    independence_rate = independent_function.function_independence_interpolation_SVM()
    real_data_rate = real_data.real_data_interpolation_SVM()

    rates = {math.nan: 'SVM',
             'Dependent function': math.nan,
             'Independent function': [str(independence_rate[0][1] * 100) + '%'],
             'Real data': [str(real_data_rate[0][1] * 100) + '%']}

    rate_dataframe = pandas.DataFrame(rates)
    success_rates = success_rates.append(rate_dataframe)

    draw_real_data_graph_SVM(current_path)
    draw_independence_graph_SVM(current_path)

    return success_rates


if __name__ == '__main__':
    main()
