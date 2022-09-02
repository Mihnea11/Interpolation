import os
import math
import pandas

# used function: 1.5 * x_2 - 3.3 * x_3 / (0.33 * x_3 + 1.0) + 1.2 * log(3.173 * x_4) + 12.0

constant = 12.0


def predict_dependence_data(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    for i, rows in raw_data.iterrows():
        result = 1.5 * rows['LYVE1'] - 3.3 * rows['REG1B'] / (0.33 * rows['REG1B'] + 1.0) + 1.2 * math.log(3.173 * rows['TFF1']) + constant
        prediction.at[i, 'plasma_CA19_9'] = result

    prediction.to_csv(os.path.join(path, r'prediction.csv'))
    return prediction
