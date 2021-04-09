import json
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
        RadiusNeighborsClassifier(),
        RidgeClassifierCV(),
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


def load_params():
    with open("params.json") as json_file:
        data = json.load(json_file)
    classifiers.append(
        LinearSVC(
            data["LinearSVC"]["penalty"],
            data["LinearSVC"]["loss"],
            data["LinearSVC"]["dual"],
            data["LinearSVC"]["tol"],
            data["LinearSVC"]["C"],
            data["LinearSVC"]["multi_class"],
            data["LinearSVC"]["fit_intercept"],
            data["LinearSVC"]["intercept_scaling"],
            data["LinearSVC"]["class_weight"],
            data["LinearSVC"]["verbose"],
            data["LinearSVC"]["random_state"],
            data["LinearSVC"]["max_iter"]
        )
    )
    classifiers.append(
        SVC(
            data["SVC"]["C"],
            data["SVC"]["kernel"],
            data["SVC"]["degree"],
            data["SVC"]["gamma"],
            data["SVC"]["coef0"],
            data["SVC"]["shrinking"],
            data["SVC"]["probability"],
            data["SVC"]["tol"],
            data["SVC"]["cache_size"],
            data["SVC"]["class_weight"],
            data["SVC"]["verbose"],
            data["SVC"]["max_iter"],
            data["SVC"]["decision_function_shape"],
            data["SVC"]["break_ties"],
            data["SVC"]["random_state"]
        )
    )
    classifiers.append(
        RadiusNeighborsClassifier(
            data["RadiusNeighborsClassifier"]["radius"],
            data["RadiusNeighborsClassifier"]["weights"],
            data["RadiusNeighborsClassifier"]["algorithm"],
            data["RadiusNeighborsClassifier"]["leaf_size"],
            data["RadiusNeighborsClassifier"]["p"],
            data["RadiusNeighborsClassifier"]["metric"],
            data["RadiusNeighborsClassifier"]["outlier_label"],
            data["RadiusNeighborsClassifier"]["metric_params"],
            data["RadiusNeighborsClassifier"]["n_jobs"]
        )
    )
    classifiers.append(
        RidgeClassifierCV(
            data["RidgeClassifierCV"]["alphas"],
            data["RidgeClassifierCV"]["fit_intercept"],
            data["RidgeClassifierCV"]["normalize"],
            data["RidgeClassifierCV"]["scoring"],
            data["RidgeClassifierCV"]["cv"],
            data["RidgeClassifierCV"]["class_weight"],
            data["RidgeClassifierCV"]["store_cv_values"]
        )
    )
    classifiers.append(
        GradientBoostingClassifier(
            data["GradientBoostingClassifier"]["loss"],
            data["GradientBoostingClassifier"]["learning_rate"],
            data["GradientBoostingClassifier"]["n_estimators"],
            data["GradientBoostingClassifier"]["subsample"],
            data["GradientBoostingClassifier"]["criterion"],
            data["GradientBoostingClassifier"]["min_samples_split"],
            data["GradientBoostingClassifier"]["min_samples_leaf"],
            data["GradientBoostingClassifier"]["min_weight_fraction_leaf"],
            data["GradientBoostingClassifier"]["max_depth"],
            data["GradientBoostingClassifier"]["min_impurity_decrease"],
            data["GradientBoostingClassifier"]["min_impurity_split"],
            data["GradientBoostingClassifier"]["init"],
            data["GradientBoostingClassifier"]["random_state"],
            data["GradientBoostingClassifier"]["max_features"],
            data["GradientBoostingClassifier"]["verbose"],
            data["GradientBoostingClassifier"]["max_leaf_nodes"],
            data["GradientBoostingClassifier"]["warm_start"],
            data["GradientBoostingClassifier"]["validation_fraction"],
            data["GradientBoostingClassifier"]["n_iter_no_change"],
            data["GradientBoostingClassifier"]["tol"],
            data["GradientBoostingClassifier"]["ccp_alpha"]
        )
    )
    classifiers.append(
        BaggingClassifier(
            data["BaggingClassifier"]["base_estimator"],
            data["BaggingClassifier"]["n_estimators"],
            data["BaggingClassifier"]["max_samples"],
            data["BaggingClassifier"]["max_features"],
            data["BaggingClassifier"]["bootstrap"],
            data["BaggingClassifier"]["bootstrap_features"],
            data["BaggingClassifier"]["oob_score"],
            data["BaggingClassifier"]["warm_start"],
            data["BaggingClassifier"]["n_jobs"],
            data["BaggingClassifier"]["random_state"],
            data["BaggingClassifier"]["verbose"]
        )
    )
    classifiers.append(
        DummyClassifier(
            data["DummyClassifier"]["strategy"],
            data["DummyClassifier"]["random_state"],
            data["DummyClassifier"]["constant"]
        )
    )
    classifiers.append(
        RidgeClassifier(
            data["RidgeClassifier"]["alpha"],
            data["RidgeClassifier"]["fit_intercept"],
            data["RidgeClassifier"]["normalize"],
            data["RidgeClassifier"]["copy_X"],
            data["RidgeClassifier"]["max_iter"],
            data["RidgeClassifier"]["tol"],
            data["RidgeClassifier"]["class_weight"],
            data["RidgeClassifier"]["solver"],
            data["RidgeClassifier"]["random_state"]
        )
    )
    classifiers.append(
        LogisticRegression(
            data["LogisticRegression"]["penalty"],
            data["LogisticRegression"]["dual"],
            data["LogisticRegression"]["tol"],
            data["LogisticRegression"]["C"],
            data["LogisticRegression"]["fit_intercept"],
            data["LogisticRegression"]["intercept_scaling"],
            data["LogisticRegression"]["class_weight"],
            data["LogisticRegression"]["random_state"],
            data["LogisticRegression"]["solver"],
            data["LogisticRegression"]["max_iter"],
            data["LogisticRegression"]["multi_class"],
            data["LogisticRegression"]["verbose"],
            data["LogisticRegression"]["warm_start"],
            data["LogisticRegression"]["n_jobs"],
            data["LogisticRegression"]["l1_ratio"]
        )
    )
    classifiers.append(
        SGDClassifier(
            data["SGDClassifier"]["loss"],
            data["SGDClassifier"]["penalty"],
            data["SGDClassifier"]["alpha"],
            data["SGDClassifier"]["l1_ratio"],
            data["SGDClassifier"]["fit_intercept"],
            data["SGDClassifier"]["max_iter"],
            data["SGDClassifier"]["tol"],
            data["SGDClassifier"]["shuffle"],
            data["SGDClassifier"]["verbose"],
            data["SGDClassifier"]["epsilon"],
            data["SGDClassifier"]["n_jobs"],
            data["SGDClassifier"]["random_state"],
            data["SGDClassifier"]["learning_rate"],
            data["SGDClassifier"]["eta0"],
            data["SGDClassifier"]["power_t"],
            data["SGDClassifier"]["early_stopping"],
            data["SGDClassifier"]["validation_fraction"],
            data["SGDClassifier"]["n_iter_no_change"],
            data["SGDClassifier"]["class_weight"],
            data["SGDClassifier"]["warm_start"],
            data["SGDClassifier"]["average"]
        )
    )
    classifiers.append(
        PassiveAggressiveClassifier(

            data["PassiveAggressiveClassifier"]["C"],
            data["PassiveAggressiveClassifier"]["fit_intercept"],
            data["PassiveAggressiveClassifier"]["max_iter"],
            data["PassiveAggressiveClassifier"]["tol"],
            data["PassiveAggressiveClassifier"]["early_stopping"],
            data["PassiveAggressiveClassifier"]["validation_fraction"],
            data["PassiveAggressiveClassifier"]["n_iter_no_change"],
            data["PassiveAggressiveClassifier"]["shuffle"],
            data["PassiveAggressiveClassifier"]["verbose"],
            data["PassiveAggressiveClassifier"]["loss"],
            data["PassiveAggressiveClassifier"]["n_jobs"],
            data["PassiveAggressiveClassifier"]["random_state"],
            data["PassiveAggressiveClassifier"]["warm_start"],
            data["PassiveAggressiveClassifier"]["class_weight"],
            data["PassiveAggressiveClassifier"]["average"]
        )
    )
    classifiers.append(
        KNeighborsClassifier(
            data["KNeighborsClassifier"]["n_neighbors"],
            data["KNeighborsClassifier"]["weights"],
            data["KNeighborsClassifier"]["algorithm"],
            data["KNeighborsClassifier"]["leaf_size"],
            data["KNeighborsClassifier"]["p"],
            data["KNeighborsClassifier"]["metric"],
            data["KNeighborsClassifier"]["metric_params"],
            data["KNeighborsClassifier"]["n_jobs"]
        )
    )
    classifiers.append(
        ExtraTreesClassifier(
            data["ExtraTreesClassifier"]["n_estimators"],
            data["ExtraTreesClassifier"]["criterion"],
            data["ExtraTreesClassifier"]["max_depth"],
            data["ExtraTreesClassifier"]["min_samples_split"],
            data["ExtraTreesClassifier"]["min_samples_leaf"],
            data["ExtraTreesClassifier"]["min_weight_fraction_leaf"],
            data["ExtraTreesClassifier"]["max_features"],
            data["ExtraTreesClassifier"]["max_leaf_nodes"],
            data["ExtraTreesClassifier"]["min_impurity_decrease"],
            data["ExtraTreesClassifier"]["min_impurity_split"],
            data["ExtraTreesClassifier"]["bootstrap"],
            data["ExtraTreesClassifier"]["oob_score"],
            data["ExtraTreesClassifier"]["n_jobs"],
            data["ExtraTreesClassifier"]["random_state"],
            data["ExtraTreesClassifier"]["verbose"],
            data["ExtraTreesClassifier"]["warm_start"],
            data["ExtraTreesClassifier"]["class_weight"],
            data["ExtraTreesClassifier"]["ccp_alpha"],
            data["ExtraTreesClassifier"]["max_samples"]
        )
    )
    classifiers.append(
        GaussianNB(
            data["GaussianNB"]["priors"],
            data["GaussianNB"]["var_smoothing"]
        )
    )
    classifiers.append(
        AdaBoostClassifier(
            data["AdaBoostClassifier"]["base_estimator"],
            data["AdaBoostClassifier"]["n_estimators"],
            data["AdaBoostClassifier"]["learning_rate"],
            data["AdaBoostClassifier"]["algorithm"],
            data["AdaBoostClassifier"]["random_state"]
        )
    )
    classifiers.append(
        GaussianProcessClassifier(
            data["GaussianProcessClassifier"]["kernel"],
            data["GaussianProcessClassifier"]["optimizer"],
            data["GaussianProcessClassifier"]["n_restarts_optimizer"],
            data["GaussianProcessClassifier"]["max_iter_predict"],
            data["GaussianProcessClassifier"]["warm_start"],
            data["GaussianProcessClassifier"]["copy_X_train"],
            data["GaussianProcessClassifier"]["random_state"],
            data["GaussianProcessClassifier"]["multi_class"],
            data["GaussianProcessClassifier"]["n_jobs"]
        )
    )
    classifiers.append(
        DecisionTreeClassifier(
            data["DecisionTreeClassifier"]["criterion"],
            data["DecisionTreeClassifier"]["splitter"],
            data["DecisionTreeClassifier"]["max_depth"],
            data["DecisionTreeClassifier"]["min_samples_split"],
            data["DecisionTreeClassifier"]["min_samples_leaf"],
            data["DecisionTreeClassifier"]["min_weight_fraction_leaf"],
            data["DecisionTreeClassifier"]["max_features"],
            data["DecisionTreeClassifier"]["random_state"],
            data["DecisionTreeClassifier"]["max_leaf_nodes"],
            data["DecisionTreeClassifier"]["min_impurity_decrease"],
            data["DecisionTreeClassifier"]["min_impurity_split"],
            data["DecisionTreeClassifier"]["class_weight"],
            data["DecisionTreeClassifier"]["ccp_alpha"]
        )
    )
    classifiers.append(
        RandomForestClassifier(
            data["RandomForestClassifier"]["n_estimators"],
            data["RandomForestClassifier"]["criterion"],
            data["RandomForestClassifier"]["max_depth"],
            data["RandomForestClassifier"]["min_samples_split"],
            data["RandomForestClassifier"]["min_samples_leaf"],
            data["RandomForestClassifier"]["min_weight_fraction_leaf"],
            data["RandomForestClassifier"]["max_features"],
            data["RandomForestClassifier"]["max_leaf_nodes"],
            data["RandomForestClassifier"]["min_impurity_decrease"],
            data["RandomForestClassifier"]["min_impurity_split"],
            data["RandomForestClassifier"]["bootstrap"],
            data["RandomForestClassifier"]["oob_score"],
            data["RandomForestClassifier"]["n_jobs"],
            data["RandomForestClassifier"]["random_state"],
            data["RandomForestClassifier"]["verbose"],
            data["RandomForestClassifier"]["warm_start"],
            data["RandomForestClassifier"]["class_weight"],
            data["RandomForestClassifier"]["ccp_alpha"],
            data["RandomForestClassifier"]["max_samples"]
        )
    )
    classifiers.append(
        MLPClassifier(
            data["MLPClassifier"]["hidden_layer_sizes"],
            data["MLPClassifier"]["activation"],
            data["MLPClassifier"]["solver"],
            data["MLPClassifier"]["alpha"],
            data["MLPClassifier"]["batch_size"],
            data["MLPClassifier"]["learning_rate"],
            data["MLPClassifier"]["learning_rate_init"],
            data["MLPClassifier"]["power_t"],
            data["MLPClassifier"]["max_iter"],
            data["MLPClassifier"]["shuffle"],
            data["MLPClassifier"]["random_state"],
            data["MLPClassifier"]["tol"],
            data["MLPClassifier"]["verbose"],
            data["MLPClassifier"]["warm_start"],
            data["MLPClassifier"]["momentum"],
            data["MLPClassifier"]["nesterovs_momentum"],
            data["MLPClassifier"]["early_stopping"],
            data["MLPClassifier"]["validation_fraction"],
            data["MLPClassifier"]["beta_1"],
            data["MLPClassifier"]["beta_2"],
            data["MLPClassifier"]["epsilon"],
            data["MLPClassifier"]["n_iter_no_change"],
            data["MLPClassifier"]["max_fun"]
        )
    )



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


# load_params()
# get_all_stats("ProjectData/Data/train_multiclass.csv", "ProjectData/Data/test_multiclass.csv")
