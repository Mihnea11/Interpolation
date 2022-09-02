import os
import math
import pandas

# used function: 0.074 * x_1 - 9.7 * cos(3.668 * x_1) + 86.0

constant = 86.0


def predict_dependence_data(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()
    for i, rows in raw_data.iterrows():
        result = 0.074 * rows[1] - 9.7 * math.cos(3.668 * rows[1]) + constant
        prediction.at[i, 0] = result

    prediction.to_csv(os.path.join(path, r'prediction.csv'))
    return prediction
