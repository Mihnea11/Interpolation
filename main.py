import os
import pandas
import pathlib
import matplotlib.pyplot as plt

import real_data_folder.real_data
from dependent_function_folder.dependent_function import function_dependence_interpolation
from independent_function_folder.independent_function import function_independence_interpolation
from real_data_folder.real_data import real_data_interpolation


def main():
    current_path = pathlib.Path(__file__).parent.resolve()

    dependence_rate = function_dependence_interpolation()
    independence_rate = function_independence_interpolation()
    real_data_rate = real_data_interpolation()

    rates = {'Dependent function': [str(dependence_rate[0][1] * 100) + '%'],
             'Independent function': [str(independence_rate[0][1] * 100) + '%'],
             'Real data': [str(real_data_rate[0][1] * 100) + '%']}

    success_rates = pandas.DataFrame(rates)
    success_rates.to_csv(os.path.join(current_path, r'success_rates.csv'), index=False)

    draw_dependence_graph(current_path)
    draw_independence_graph(current_path)
    draw_real_data_graph(current_path)


def draw_dependence_graph(path):
    data = pandas.read_csv(os.path.join(path, r'dependent_function_folder\data\prediction.csv'))

    x = []
    y = data['0']
    y_real = data['5']

    i = 0
    for i in range(len(data.index)):
        x.append(i)
        i = i + 1

    plt.plot(x, y_real, 'o', label='generated')
    plt.plot(x, y, 'o', label='predicted')
    plt.legend()
    plt.savefig(os.path.join(path, r'data_graphs_folder\dependent_function_graph.pdf'))
    plt.close()


def draw_independence_graph(path):
    data = pandas.read_csv(os.path.join(path, r'independent_function_folder\data\prediction.csv'))

    x = []
    y = data['0']
    y_real = data['5']

    i = 0
    for i in range(len(data.index)):
        x.append(i)
        i = i + 1

    plt.plot(x, y_real, 'o', label='generated')
    plt.plot(x, y, 'o', label='predicted')
    plt.legend()
    plt.savefig(os.path.join(path, r'data_graphs_folder\'independent_function_graph.pdf'))
    plt.close()


def draw_real_data_graph(path):
    original = pandas.read_csv(os.path.join(path, r'real_data_folder\data\raw_data.csv'))
    predicted = pandas.read_csv(os.path.join(path, r'real_data_folder\data\prediction.csv'))

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
    plt.savefig(os.path.join(path, r'data_graphs_folder\'real_data_graph.pdf'))
    plt.close()


if __name__ == '__main__':
    main()
