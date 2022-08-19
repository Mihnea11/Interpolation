import os
import math
import numpy 
import pandas
import warnings


warnings.filterwarnings('ignore')


def generate_data(path):
    # Create a sample dataset with n samples
    N_SAMPLES = 100
    time = numpy.arange(0, N_SAMPLES, 0.1)
    # signal = numpy.sin(time)
    # Generate normally distributed random samples with two features

    data = numpy.empty((N_SAMPLES, 6))
    x = numpy.linspace(0, 10 * numpy.pi, num=N_SAMPLES)

    n = numpy.random.gumbel(loc=20.0, size=N_SAMPLES)
    data[:, 1] = 50 * numpy.sin(x) + n

    n = numpy.random.laplace(loc=30.0, size=N_SAMPLES)
    data[:, 2] = 100 * numpy.sin(x) + n

    n = numpy.random.gumbel(loc=40.0, size=N_SAMPLES)
    data[:, 3] = 150 * numpy.sin(x) + n

    n = numpy.random.laplace(loc=50.0, size=N_SAMPLES)
    data[:, 4] = 200 * numpy.sin(x) + n

    n = numpy.random.gumbel(loc=10.0, size=N_SAMPLES)
    data[:, 0] = data[:, 1] * n + data[:, 3] / data[:, 4]

    dataFormat = pandas.DataFrame(data)
    dataFormat[5] = dataFormat.iloc[:, [0]].copy()
    i=0
    for i, j in dataFormat.iterrows():
        if i % 2 == 0 or i % 3 == 0 or i % 5 == 0:
            dataFormat.at[i, 0] = math.nan

    dataFormat.to_csv(os.path.join(path, r'raw_data.csv'))
    return dataFormat
