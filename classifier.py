import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

methods = ["LogisticRegression"]
# preprocess_method - способ подготовки данных. 1 - нормализация; 2 - стандартизация; по умолчанию - нормализация
def load_and_preprocess_data(path_to_data, preprocess_method=1):
    data = pd.read_csv(path_to_data, sep=';', thousands=",", low_memory=False)
    del data["path_to_cell"]

    # Вытаскиваем данные
    X = data.iloc[:, 1:]
    # Вытаскиваем лейблы
    y = data["Label"]

    if (preprocess_method == 2):
        # standardize the data attributes
        X = preprocessing.scale(X)
    else:
        # normalize the data attributes
        X = preprocessing.normalize(X)

    return (X, y)

def get_load_file_stats(y):
    stats = Counter(y)
    # print(len(stats))
    stats = pd.DataFrame.from_dict(stats, orient='index').reset_index()
    stats.rename(columns={"index": "Клетка", 0: "Количество клеток"}, inplace=True)
    stats_list = []
    for i in range(0, len(stats)):
        stats_list.append((stats.loc[i][0], str(stats.loc[i][1])))

    return stats_list

def logistic_regression(X, y):
    model = LogisticRegression(max_iter=10000)
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    return (expected,predicted)
