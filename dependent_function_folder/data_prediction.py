import os
import pandas

# used function: 8.1 * x_1 + 0.62 * x_2 + 19

constant = 19.0


def predict_dependence_data(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()
    for i, rows in raw_data.iterrows():
        result = 8.1 * float(rows[1]) + 0.62 * float(rows[2]) + constant
        prediction.at[i, 0] = result

    prediction.to_csv(os.path.join(path, r'prediction.csv'))
    return prediction
