import os
import math
import pandas

# used function: 0.074 * x_1 - 9.7 * cos(3.668 * x_1) + 86.0

firstOperand = 0.074
secondOperand = 9.7
thirdOperand = 3.668
constant = 86.0


def predict_dependence_data(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()
    for i, rows in raw_data.iterrows():
        result = firstOperand * rows[1] - secondOperand * math.cos(thirdOperand * rows[1]) + constant
        prediction.at[i, 0] = result

    prediction.to_csv(os.path.join(path, r'prediction.csv'))
    return prediction
