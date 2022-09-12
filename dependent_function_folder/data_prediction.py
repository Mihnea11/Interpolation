import os
import pandas

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


def predict_dependence_data_CMAGEP(path, raw_data: pandas.DataFrame):
    # used function: 8.1 * x_1 + 0.62 * x_2 + 19

    prediction = raw_data.copy()
    for i, rows in raw_data.iterrows():
        result = 8.1 * float(rows[1]) + 0.62 * float(rows[2]) + 19.0
        prediction.at[i, '0'] = result

    prediction.to_csv(os.path.join(path, r'prediction_CMAGEP.csv'))
    return prediction


def predict_dependence_data_SVM(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    target = raw_data['0']

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


def predict_dependence_data_KNN(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    target = raw_data['0']

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

    parameters = raw_data
    parameters.drop('0', axis=1, inplace=True)
    parameters.drop('5', axis=1, inplace=True)
    parameters.to_numpy()

    y_predict = knn.predict(parameters)

    i = 0
    for element in y_predict:
        prediction['0'][i] = element
        i = i + 1

    prediction.to_csv(os.path.join(path, r'prediction_KNN.csv'))
    return prediction


def predict_dependence_data_RandomForest(path, raw_data: pandas.DataFrame):
    prediction = raw_data.copy()

    target = raw_data['0']

    x = pandas.read_csv(os.path.join(path, r'input.csv'))
    y = pandas.read_csv(os.path.join(path, r'target.csv'))

    #i = 0
    #for element in target:
    #    if not (element != element):
    #        y.append(int(target[i]))
    #    i = i + 1

    x = x.to_numpy()
    y = y.to_numpy()

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(x, y)

    parameters = raw_data
    parameters.drop('0', axis=1, inplace=True)
    parameters.drop('5', axis=1, inplace=True)
    parameters.to_numpy()

    y_predict = regressor.predict(parameters)

    i = 0
    for element in y_predict:
        prediction['0'][i] = element
        i = i + 1

    prediction.to_csv(os.path.join(path, r'prediction_RandomForest.csv'))
    return prediction
