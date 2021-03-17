import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from joblib import dump, load
import datetime
import sklearn.utils._weight_vector

# Перечень доступных моделей в Системе
classifiers = [
    LogisticRegression(max_iter=10000),
    KNeighborsClassifier(),
    ExtraTreesClassifier(),
    GaussianNB()
]

# генератор имен файлов
def generate_save_name(classifier,preprocess_method):

    classifier = str(classifier)
    name = "ProjectData/Weights/Classic/"
    file_format = ".joblib"
    now = datetime.datetime.now()
    name = name + classifier + "_" + str(preprocess_method) + "_" + str(now.strftime("%d-%m-%Y %H:%M")) + file_format
    return name

# preprocess_method - способ подготовки данных. 1 - нормализация; 2 - стандартизация
def load_and_preprocess_data(path_to_data, preprocess_method):
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

# функция считывания количества данных по каждому классу в файле
def get_load_file_stats(y):
    stats = Counter(y)
    # print(len(stats))
    stats = pd.DataFrame.from_dict(stats, orient='index').reset_index()
    stats.rename(columns={"index": "Клетка", 0: "Количество клеток"}, inplace=True)
    stats_list = []
    for i in range(0, len(stats)):
        stats_list.append((stats.loc[i][0], str(stats.loc[i][1])))

    return stats_list


# функция обучения классификаторов
def control_classifiers(X, y, num_classifier,preprocess_method):
    model = classifiers[num_classifier]
    model.fit(X, y)
    expected = y
    predicted = model.predict(X)

    file_name = generate_save_name(classifiers[num_classifier],preprocess_method)
    dump(model, file_name)

    return (expected, predicted)
