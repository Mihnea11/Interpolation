from dependent_function_folder.dependent_function import function_dependence_interpolation
from independent_function_folder.independent_function import function_independence_interpolation
import pandas
import pathlib
import os


def main():
    current_path = pathlib.Path(__file__).parent.resolve()

    dependence_rate = function_dependence_interpolation()
    independence_rate = function_independence_interpolation()
    # real_data_rate = ?

    rates = {'Dependent function': [str(dependence_rate[0][1] * 100) + '%'],
             'Independent function': [str(independence_rate[0][1] * 100) + '%'],
             'Real data': [99]}

    success_rates = pandas.DataFrame(rates)
    success_rates.to_csv(os.path.join(current_path, r'success_rates.csv'), index=False)


if __name__ == '__main__':
    main()
