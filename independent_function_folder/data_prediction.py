import os
import math
import pandas

from sklearn.svm import SVC


def predict_dependence_data_CMAGEP(path, raw_data: pandas.DataFrame):
    # used function: 0.074 * x_1 - 9.7 * cos(3.668 * x_1) + 86.0

    prediction = raw_data.copy()
    for i, rows in raw_data.iterrows():
        result = 0.074 * rows[1] - 9.7 * math.cos(3.668 * rows[1]) + 86.0
        prediction.at[i, '0'] = result

    prediction.to_csv(os.path.join(path, r'prediction_CMAGEP.csv'))
    return prediction


def predict_dependence_data_SVM(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    target = raw_data['0']

    x = pandas.read_csv(os.path.join(path, r'input.csv'))
    y = []

    i = 0
    for element in target:
        if not (element != element):
            y.append(int(target[i]))

        i = i + 1

    x = x.to_numpy()

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x, y)

    parameters = raw_data
    parameters.drop('0', axis=1, inplace=True)
    parameters.drop('5', axis=1, inplace=True)
    parameters.to_numpy()

    y_predict = classifier.predict(parameters)

    i = 0
    for element in y_predict:
        prediction['0'][i] = element
        i = i + 1

    prediction.to_csv(os.path.join(path, r'prediction_SVM.csv'))
    return prediction
