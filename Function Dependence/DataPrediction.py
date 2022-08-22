import os
import pandas

# used function: 8.1 * x_1 + 0.62 * x_2 + 19

firstOperand = 8.1
secondOperand = 0.62
constant = 19.0


def predict_data(path, rawData: pandas.DataFrame):
    prediction = rawData.copy()
    for i, rows in rawData.iterrows():
        result = firstOperand * float(rows[1]) + secondOperand * float(rows[2]) + constant
        prediction.at[i, 0] = result

    prediction.to_csv(os.path.join(path, r'prediction.csv'))
    return prediction
