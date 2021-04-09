import timeit
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.multiclass import OneVsOneClassifier, OutputCodeClassifier, OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, \
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeCV, RidgeClassifier, \
    TweedieRegressor, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from collections import Counter
# from joblib import dump, load
import pickle
import joblib
import datetime

from sklearn.neural_network import MLPClassifier
import sklearn.utils._weight_vector

# Перечень доступных моделей в Системе
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    # VotingClassifier([("1",KNeighborsClassifier()), ("2",LinearSVC()), ("3",MLPClassifier())]),
    LinearSVC(),
    # NuSVC(),
    SVC(),
    RadiusNeighborsClassifier(),
    # MultiOutputClassifier(KNeighborsClassifier()),
    # ClassifierChain(),
    # OutputCodeClassifier(),
    # OneVsRestClassifier(),
    # OneVsOneClassifier(),
    RidgeClassifierCV(),

    # StackingClassifier(),
    # HistGradientBoostingClassifier(),
    GradientBoostingClassifier(),
    BaggingClassifier(),
    DummyClassifier(),
    RidgeClassifier(),
    LogisticRegression(max_iter=10000),
    SGDClassifier(),
    PassiveAggressiveClassifier(),
    KNeighborsClassifier(),
    ExtraTreesClassifier(),
    GaussianNB(),
    AdaBoostClassifier(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier()
]


# генератор имен файлов
def generate_save_name(classifier, preprocess_method):
    classifier = str(classifier)
    name = "ProjectData/Weights/Classic/"
    file_format = ".pkl"
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
    preprocess_method = 1
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
def control_classifiers(X, y, num_classifier, preprocess_method):
    model = classifiers[num_classifier]
    model.fit(X, y)
    expected = y
    predicted = model.predict(X)

    file_name = generate_save_name(classifiers[num_classifier], preprocess_method)
    print(file_name)
    # joblib.dump(model, file_name)
    with open("test.pkl", 'wb') as file:
        pickle.dump(model, file)
    return (expected, predicted)


def test_model(X, y, path):
    # model = classifiers[4]
    # path = path+".pkl"
    with open("test.pkl", 'rb') as file:
        model = pickle.load(file)

    expected = y
    predicted = model.predict(X)

    return (expected, predicted)


# x,y= load_and_preprocess_data(r"C:\_Programming\BloodClassifierClassic\ProjectData\Data\test_multiclass.csv",2)
# expected, predicted =test_model(x,y,r"C:\_Programming\BloodClassifierClassic\ProjectData\Weights\Classic\MLPClassifier()_Стандартизация_24-03-2021 23")
#
# a =metrics.classification_report(expected, predicted, zero_division=0)
# print(a)
# a= metrics.confusion_matrix(expected, predicted)
# print(a)

def get_best_x(X, y, best=-1):
    if best == -1:
        clf = RandomForestClassifier()
        clf.fit(X, y)
        params = clf.feature_importances_
        x = 0
        for p in params:
            x += p
        x = x / 702
        print(x)

        best = []
        for i in range(0, 702):
            if params[i] >= x:
                best.append(i)
    X = X[:, best]
    return X, best


def get_all_stats(path_train, path_test):
    list_acc = []
    for m in classifiers:
        X, y = load_and_preprocess_data(path_train, 1)
        X, best = get_best_x(X, y)
        print("Обучение и тестирование: ", m)
        model = m
        a = timeit.default_timer()
        model.fit(X, y)
        train_time = timeit.default_timer() - a

        print("Показатели на обучающих данных: ")
        expected = y
        predicted = model.predict(X)
        # print(metrics.classification_report(expected, predicted, zero_division=0))
        # print(metrics.confusion_matrix(expected, predicted))
        acc_train = metrics.accuracy_score(expected, predicted)
        print("Точность: ", acc_train)

        X, y = load_and_preprocess_data(path_test, 1)
        X = get_best_x(X, y, best)

        print("Показатели на тестовых данных: ")

        expected = y
        a = timeit.default_timer()
        predicted = model.predict(X)
        test_time = timeit.default_timer() - a

        # print(metrics.classification_report(expected, predicted, zero_division=0))
        # print(metrics.confusion_matrix(expected, predicted))
        acc_test = metrics.accuracy_score(expected, predicted)
        print("Точность: ", acc_test)

        list_acc.append((m, acc_train, acc_test, train_time, test_time))
    for acc in list_acc:
        print(acc)


# get_all_stats("ProjectData/Data/train_multiclass.csv", "ProjectData/Data/test_multiclass.csv")
