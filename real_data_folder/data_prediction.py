import os
import math
import pandas

# used function: 1.5 * x_2 - 3.3 * x_3 / (0.33 * x_3 + 1.0) + 1.2 * log(3.173 * x_4) + 12.0
# 1.4*x_4^0.5*cos(0.703*x_1)
# 2.5*cos(1.967*sin(0.143*log(0.025*x_1 + 0.105*x_2))) + 0.19
# x_1+x_2*cos(x_2^(1/4))+log(exp(x_2))+3.8
# 1.1*x_2 + 0.99*x_4 + 2.5*sin(0.774*x_3) + 6.4
# 0.8*log(0.586*cos(0.319333333333333*x_1)) + 1.6*sin(0.322*x_2) + 1.4*cos(0.74*cos(1.076*x_1)) + 1.2*cos(1.558*cos(2.055*x_1 + 3.640835*x_2))
constant = 12.0


def predict_dependence_data(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    for i, rows in raw_data.iterrows():
        # result = 1.5 * rows['LYVE1'] - 3.3 * rows['REG1B'] / (0.33 * rows['REG1B'] + 1.0) + 1.2 * math.log(3.173 * rows['TFF1']) + constant
        # result = 1.4 * rows['TFF1'] ** 0.5 * math.cos(0.703 * rows['creatinine'])
        # result = math.sqrt(rows['LYVE1']) + 3 + math.exp(1 - rows['creatinine']) + math.sin(math.log(2)) + rows['LYVE1'] + math.log(rows['TFF1'])
        # result = 2.5 * math.cos(1.967 * math.sin(0.143* math.log(0.025*rows['creatinine'] + 0.105*rows['LYVE1']))) + 0.19
        # result = rows['creatinine'] + rows['LYVE1'] * math.cos(rows['LYVE1'] ** (1/4)) + math.log(math.exp(rows['LYVE1'])) + 3.8

        result = 1.1 * rows['LYVE1'] + 0.99 * rows['TFF1'] + 2.5 * math.sin(0.774 * rows['REG1B']) + 6.4
        prediction.at[i, 'plasma_CA19_9'] = result

    prediction.to_csv(os.path.join(path, r'prediction.csv'))
    return prediction
