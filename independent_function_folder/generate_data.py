import os
import math
import numpy
import pandas
import warnings


warnings.filterwarnings('ignore')


def generate_raw_data(path):
    # Create a sample dataset with n samples
    n_sample = 1000
    # time = numpy.arrange(0, n_sample, 0.1)
    # signal = numpy.sin(time)
    # Generate normally distributed random samples with two features

    data = numpy.empty((n_sample, 6))
    x = numpy.linspace(0, 10 * numpy.pi, num=n_sample)

    n = numpy.random.gumbel(loc=10.0, size=n_sample)
    data[:, 0] = numpy.sin(x) + n ** 2 - x * 5

    n = numpy.random.gumbel(loc=20.0, size=n_sample)
    data[:, 1] = 50 * numpy.sin(x) + n

    n = numpy.random.laplace(loc=30.0, size=n_sample)
    data[:, 2] = 100 * numpy.sin(x) + n

    n = numpy.random.gumbel(loc=40.0, size=n_sample)
    data[:, 3] = 150 * numpy.sin(x) + n

    n = numpy.random.laplace(loc=50.0, size=n_sample)
    data[:, 4] = 200 * numpy.sin(x) + n

    data_format = pandas.DataFrame(data)
    data_format[5] = data_format.iloc[:, [0]].copy()

    for i, j in data_format.iterrows():
        if i % 2 == 0 or i % 5 == 0:
            data_format.at[i, 0] = math.nan

    data_format.to_csv(os.path.join(path, r'raw_data.csv'))
    return data_format
