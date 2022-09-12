import os
import math
import pandas

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


def predict_real_data_CMAGEP(path, raw_data: pandas.DataFrame):
    # used function: 1.1 * x_2 + 0.99 * log(x_4) + 2.5 * sin(0.774 * x_3) + 6.4

    prediction = raw_data.copy()
    for i, rows in raw_data.iterrows():
        result = 1.1 * rows['LYVE1'] + 0.99 * math.log(rows['TFF1']) + 2.5 * math.sin(0.774 * rows['REG1B']) + 6.4
        prediction.at[i, 'plasma_CA19_9'] = result

    prediction.to_csv(os.path.join(path, r'prediction_CMAGEP.csv'))
    return prediction


def predict_real_data_SVM(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    target = raw_data['plasma_CA19_9']

    x = pandas.read_csv(os.path.join(path, r'input.csv'))
    y = pandas.read_csv(os.path.join(path, r'target.csv'))

    #i = 0
    #for element in target:
    #    if not (element != element):
    #        y.append(int(target[i]))
    #    i = i + 1

    x = x.to_numpy()
    y = y.to_numpy()

    classifier = SVR(kernel='linear')
    classifier.fit(x, y)

    parameters = raw_data[['creatinine', 'LYVE1', 'REG1B', 'TFF1']]
    parameters.to_numpy()

    y_predict = classifier.predict(parameters)

    i = 0
    for element in y_predict:
        prediction['plasma_CA19_9'][i] = element
        i = i + 1

    prediction.to_csv(os.path.join(path, r'prediction_SVM.csv'))
    return prediction


def predict_real_data_KNN(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    target = raw_data['plasma_CA19_9']

    x = pandas.read_csv(os.path.join(path, r'input.csv'))
    y = pandas.read_csv(os.path.join(path, r'target.csv'))

    #i = 0
    #for element in target:
    #    if not (element != element):
    #        y.append(int(target[i]))
    #    i = i + 1

    x = x.to_numpy()
    y = y.to_numpy()

    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(x, y)

    parameters = raw_data[['creatinine', 'LYVE1', 'REG1B', 'TFF1']]
    parameters.to_numpy()

    y_predict = knn.predict(parameters)

    i = 0
    for element in y_predict:
        prediction['plasma_CA19_9'][i] = element
        i = i + 1

    prediction.to_csv(os.path.join(path, r'prediction_KNN.csv'))
    return prediction


def predict_real_data_RandomForest(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    target = raw_data['plasma_CA19_9']

    x = pandas.read_csv(os.path.join(path, r'input.csv'))
    y = pandas.read_csv(os.path.join(path, r'target.csv'))

    #i = 0
    #for element in target:
    #    if not (element != element):
    #        y.append(int(target[i]))
    #    i = i + 1

    x = x.to_numpy()
    y = y.to_numpy()

    regressor = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=0)
    regressor.fit(x, y)

    parameters = raw_data[['creatinine', 'LYVE1', 'REG1B', 'TFF1']]
    parameters.to_numpy()

    y_predict = regressor.predict(parameters)

    i = 0
    for element in y_predict:
        prediction['plasma_CA19_9'][i] = element
        i = i + 1

    prediction.to_csv(os.path.join(path, r'prediction_RandomForest.csv'))
    return prediction
