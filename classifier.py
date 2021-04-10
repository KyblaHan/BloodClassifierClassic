import datetime
import json
import pickle
import timeit
from collections import Counter
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, \
    GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, \
    SGDClassifier, PassiveAggressiveClassifier, RidgeClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
# Перечень доступных моделей в Системе
from sklearn.tree import DecisionTreeClassifier


"""
classifiers - перечень всех классификаторов.
"""
classifiers = [
    # RadiusNeighborsClassifier(),
    # RidgeClassifierCV(),
    # GradientBoostingClassifier(),
    # BaggingClassifier(),
    # DummyClassifier(),
    # RidgeClassifier(),
    # LogisticRegression(max_iter=10000),
    # SGDClassifier(),
    # PassiveAggressiveClassifier(),
    # KNeighborsClassifier(),
    # ExtraTreesClassifier(),
    # GaussianNB(),
    # AdaBoostClassifier(),
    # GaussianProcessClassifier(),
    # DecisionTreeClassifier(),
    # RandomForestClassifier(),
    # MLPClassifier()
]


def load_params():
    """
    Загружает информацию о клаасификаторах из json файла: ProjectData/SystemFiles/params.json
    :return:
    """

    with open("ProjectData/SystemFiles/params.json") as json_file:
        data = json.load(json_file)
    classifiers.append(
        LinearSVC(
            penalty=data["LinearSVC"]["penalty"],
            loss=data["LinearSVC"]["loss"],
            dual=data["LinearSVC"]["dual"],
            tol=data["LinearSVC"]["tol"],
            C=data["LinearSVC"]["C"],
            multi_class=data["LinearSVC"]["multi_class"],
            fit_intercept=data["LinearSVC"]["fit_intercept"],
            intercept_scaling=data["LinearSVC"]["intercept_scaling"],
            class_weight=data["LinearSVC"]["class_weight"],
            verbose=data["LinearSVC"]["verbose"],
            random_state=data["LinearSVC"]["random_state"],
            max_iter=data["LinearSVC"]["max_iter"]
        )
    )
    classifiers.append(
        SVC(
            C=data["SVC"]["C"],
            kernel=data["SVC"]["kernel"],
            degree=data["SVC"]["degree"],
            gamma=data["SVC"]["gamma"],
            coef0=data["SVC"]["coef0"],
            shrinking=data["SVC"]["shrinking"],
            probability=data["SVC"]["probability"],
            tol=data["SVC"]["tol"],
            cache_size=data["SVC"]["cache_size"],
            class_weight=data["SVC"]["class_weight"],
            verbose=data["SVC"]["verbose"],
            max_iter=data["SVC"]["max_iter"],
            decision_function_shape=data["SVC"]["decision_function_shape"],
            break_ties=data["SVC"]["break_ties"],
            random_state=data["SVC"]["random_state"]
        )
    )
    classifiers.append(
        RadiusNeighborsClassifier(
            radius=data["RadiusNeighborsClassifier"]["radius"],
            weights=data["RadiusNeighborsClassifier"]["weights"],
            algorithm=data["RadiusNeighborsClassifier"]["algorithm"],
            leaf_size=data["RadiusNeighborsClassifier"]["leaf_size"],
            p=data["RadiusNeighborsClassifier"]["p"],
            metric=data["RadiusNeighborsClassifier"]["metric"],
            outlier_label=data["RadiusNeighborsClassifier"]["outlier_label"],
            metric_params=data["RadiusNeighborsClassifier"]["metric_params"],
            n_jobs=data["RadiusNeighborsClassifier"]["n_jobs"]
        )
    )
    classifiers.append(
        RidgeClassifierCV(
            alphas=data["RidgeClassifierCV"]["alphas"],
            fit_intercept=data["RidgeClassifierCV"]["fit_intercept"],
            normalize=data["RidgeClassifierCV"]["normalize"],
            scoring=data["RidgeClassifierCV"]["scoring"],
            cv=data["RidgeClassifierCV"]["cv"],
            class_weight=data["RidgeClassifierCV"]["class_weight"],
            store_cv_values=data["RidgeClassifierCV"]["store_cv_values"]
        )
    )
    classifiers.append(
        GradientBoostingClassifier(
            loss=data["GradientBoostingClassifier"]["loss"],
            learning_rate=data["GradientBoostingClassifier"]["learning_rate"],
            n_estimators=data["GradientBoostingClassifier"]["n_estimators"],
            subsample=data["GradientBoostingClassifier"]["subsample"],
            criterion=data["GradientBoostingClassifier"]["criterion"],
            min_samples_split=data["GradientBoostingClassifier"]["min_samples_split"],
            min_samples_leaf=data["GradientBoostingClassifier"]["min_samples_leaf"],
            min_weight_fraction_leaf=data["GradientBoostingClassifier"]["min_weight_fraction_leaf"],
            max_depth=data["GradientBoostingClassifier"]["max_depth"],
            min_impurity_decrease=data["GradientBoostingClassifier"]["min_impurity_decrease"],
            min_impurity_split=data["GradientBoostingClassifier"]["min_impurity_split"],
            init=data["GradientBoostingClassifier"]["init"],
            random_state=data["GradientBoostingClassifier"]["random_state"],
            max_features=data["GradientBoostingClassifier"]["max_features"],
            verbose=data["GradientBoostingClassifier"]["verbose"],
            max_leaf_nodes=data["GradientBoostingClassifier"]["max_leaf_nodes"],
            warm_start=data["GradientBoostingClassifier"]["warm_start"],
            validation_fraction=data["GradientBoostingClassifier"]["validation_fraction"],
            n_iter_no_change=data["GradientBoostingClassifier"]["n_iter_no_change"],
            tol=data["GradientBoostingClassifier"]["tol"],
            ccp_alpha=data["GradientBoostingClassifier"]["ccp_alpha"]
        )
    )
    classifiers.append(
        BaggingClassifier(
            base_estimator=data["BaggingClassifier"]["base_estimator"],
            n_estimators=data["BaggingClassifier"]["n_estimators"],
            max_samples=data["BaggingClassifier"]["max_samples"],
            max_features=data["BaggingClassifier"]["max_features"],
            bootstrap=data["BaggingClassifier"]["bootstrap"],
            bootstrap_features=data["BaggingClassifier"]["bootstrap_features"],
            oob_score=data["BaggingClassifier"]["oob_score"],
            warm_start=data["BaggingClassifier"]["warm_start"],
            n_jobs=data["BaggingClassifier"]["n_jobs"],
            random_state=data["BaggingClassifier"]["random_state"],
            verbose=data["BaggingClassifier"]["verbose"]
        )
    )
    classifiers.append(
        DummyClassifier(
            strategy=data["DummyClassifier"]["strategy"],
            random_state=data["DummyClassifier"]["random_state"],
            constant=data["DummyClassifier"]["constant"]
        )
    )
    classifiers.append(
        RidgeClassifier(
            alpha=data["RidgeClassifier"]["alpha"],
            fit_intercept=data["RidgeClassifier"]["fit_intercept"],
            normalize=data["RidgeClassifier"]["normalize"],
            copy_X=data["RidgeClassifier"]["copy_X"],
            max_iter=data["RidgeClassifier"]["max_iter"],
            tol=data["RidgeClassifier"]["tol"],
            class_weight=data["RidgeClassifier"]["class_weight"],
            solver=data["RidgeClassifier"]["solver"],
            random_state=data["RidgeClassifier"]["random_state"]
        )
    )
    classifiers.append(
        LogisticRegression(
            penalty=data["LogisticRegression"]["penalty"],
            dual=data["LogisticRegression"]["dual"],
            tol=data["LogisticRegression"]["tol"],
            C=data["LogisticRegression"]["C"],
            fit_intercept=data["LogisticRegression"]["fit_intercept"],
            intercept_scaling=data["LogisticRegression"]["intercept_scaling"],
            class_weight=data["LogisticRegression"]["class_weight"],
            random_state=data["LogisticRegression"]["random_state"],
            solver=data["LogisticRegression"]["solver"],
            max_iter=data["LogisticRegression"]["max_iter"],
            multi_class=data["LogisticRegression"]["multi_class"],
            verbose=data["LogisticRegression"]["verbose"],
            warm_start=data["LogisticRegression"]["warm_start"],
            n_jobs=data["LogisticRegression"]["n_jobs"],
            l1_ratio=data["LogisticRegression"]["l1_ratio"]
        )
    )
    classifiers.append(
        SGDClassifier(
            loss=data["SGDClassifier"]["loss"],
            penalty=data["SGDClassifier"]["penalty"],
            alpha=data["SGDClassifier"]["alpha"],
            l1_ratio=data["SGDClassifier"]["l1_ratio"],
            fit_intercept=data["SGDClassifier"]["fit_intercept"],
            max_iter=data["SGDClassifier"]["max_iter"],
            tol=data["SGDClassifier"]["tol"],
            shuffle=data["SGDClassifier"]["shuffle"],
            verbose=data["SGDClassifier"]["verbose"],
            epsilon=data["SGDClassifier"]["epsilon"],
            n_jobs=data["SGDClassifier"]["n_jobs"],
            random_state=data["SGDClassifier"]["random_state"],
            learning_rate=data["SGDClassifier"]["learning_rate"],
            eta0=data["SGDClassifier"]["eta0"],
            power_t=data["SGDClassifier"]["power_t"],
            early_stopping=data["SGDClassifier"]["early_stopping"],
            validation_fraction=data["SGDClassifier"]["validation_fraction"],
            n_iter_no_change=data["SGDClassifier"]["n_iter_no_change"],
            class_weight=data["SGDClassifier"]["class_weight"],
            warm_start=data["SGDClassifier"]["warm_start"],
            average=data["SGDClassifier"]["average"]
        )
    )
    classifiers.append(
        PassiveAggressiveClassifier(

            C=data["PassiveAggressiveClassifier"]["C"],
            fit_intercept=data["PassiveAggressiveClassifier"]["fit_intercept"],
            max_iter=data["PassiveAggressiveClassifier"]["max_iter"],
            tol=data["PassiveAggressiveClassifier"]["tol"],
            early_stopping=data["PassiveAggressiveClassifier"]["early_stopping"],
            validation_fraction=data["PassiveAggressiveClassifier"]["validation_fraction"],
            n_iter_no_change=data["PassiveAggressiveClassifier"]["n_iter_no_change"],
            shuffle=data["PassiveAggressiveClassifier"]["shuffle"],
            verbose=data["PassiveAggressiveClassifier"]["verbose"],
            loss=data["PassiveAggressiveClassifier"]["loss"],
            n_jobs=data["PassiveAggressiveClassifier"]["n_jobs"],
            random_state=data["PassiveAggressiveClassifier"]["random_state"],
            warm_start=data["PassiveAggressiveClassifier"]["warm_start"],
            class_weight=data["PassiveAggressiveClassifier"]["class_weight"],
            average=data["PassiveAggressiveClassifier"]["average"]
        )
    )
    classifiers.append(
        KNeighborsClassifier(
            n_neighbors=data["KNeighborsClassifier"]["n_neighbors"],
            weights=data["KNeighborsClassifier"]["weights"],
            algorithm=data["KNeighborsClassifier"]["algorithm"],
            leaf_size=data["KNeighborsClassifier"]["leaf_size"],
            p=data["KNeighborsClassifier"]["p"],
            metric=data["KNeighborsClassifier"]["metric"],
            metric_params=data["KNeighborsClassifier"]["metric_params"],
            n_jobs=data["KNeighborsClassifier"]["n_jobs"]
        )
    )
    classifiers.append(
        ExtraTreesClassifier(
            n_estimators=data["ExtraTreesClassifier"]["n_estimators"],
            criterion=data["ExtraTreesClassifier"]["criterion"],
            max_depth=data["ExtraTreesClassifier"]["max_depth"],
            min_samples_split=data["ExtraTreesClassifier"]["min_samples_split"],
            min_samples_leaf=data["ExtraTreesClassifier"]["min_samples_leaf"],
            min_weight_fraction_leaf=data["ExtraTreesClassifier"]["min_weight_fraction_leaf"],
            max_features=data["ExtraTreesClassifier"]["max_features"],
            max_leaf_nodes=data["ExtraTreesClassifier"]["max_leaf_nodes"],
            min_impurity_decrease=data["ExtraTreesClassifier"]["min_impurity_decrease"],
            min_impurity_split=data["ExtraTreesClassifier"]["min_impurity_split"],
            bootstrap=data["ExtraTreesClassifier"]["bootstrap"],
            oob_score=data["ExtraTreesClassifier"]["oob_score"],
            n_jobs=data["ExtraTreesClassifier"]["n_jobs"],
            random_state=data["ExtraTreesClassifier"]["random_state"],
            verbose=data["ExtraTreesClassifier"]["verbose"],
            warm_start=data["ExtraTreesClassifier"]["warm_start"],
            class_weight=data["ExtraTreesClassifier"]["class_weight"],
            ccp_alpha=data["ExtraTreesClassifier"]["ccp_alpha"],
            max_samples=data["ExtraTreesClassifier"]["max_samples"]
        )
    )
    classifiers.append(
        GaussianNB(
            priors=data["GaussianNB"]["priors"],
            var_smoothing=data["GaussianNB"]["var_smoothing"]
        )
    )
    classifiers.append(
        AdaBoostClassifier(
            base_estimator=data["AdaBoostClassifier"]["base_estimator"],
            n_estimators=data["AdaBoostClassifier"]["n_estimators"],
            learning_rate=data["AdaBoostClassifier"]["learning_rate"],
            algorithm=data["AdaBoostClassifier"]["algorithm"],
            random_state=data["AdaBoostClassifier"]["random_state"]
        )
    )
    classifiers.append(
        GaussianProcessClassifier(
            kernel=data["GaussianProcessClassifier"]["kernel"],
            optimizer=data["GaussianProcessClassifier"]["optimizer"],
            n_restarts_optimizer=data["GaussianProcessClassifier"]["n_restarts_optimizer"],
            max_iter_predict=data["GaussianProcessClassifier"]["max_iter_predict"],
            warm_start=data["GaussianProcessClassifier"]["warm_start"],
            copy_X_train=data["GaussianProcessClassifier"]["copy_X_train"],
            random_state=data["GaussianProcessClassifier"]["random_state"],
            multi_class=data["GaussianProcessClassifier"]["multi_class"],
            n_jobs=data["GaussianProcessClassifier"]["n_jobs"]
        )
    )
    classifiers.append(
        DecisionTreeClassifier(
            criterion=data["DecisionTreeClassifier"]["criterion"],
            splitter=data["DecisionTreeClassifier"]["splitter"],
            max_depth=data["DecisionTreeClassifier"]["max_depth"],
            min_samples_split=data["DecisionTreeClassifier"]["min_samples_split"],
            min_samples_leaf=data["DecisionTreeClassifier"]["min_samples_leaf"],
            min_weight_fraction_leaf=data["DecisionTreeClassifier"]["min_weight_fraction_leaf"],
            max_features=data["DecisionTreeClassifier"]["max_features"],
            random_state=data["DecisionTreeClassifier"]["random_state"],
            max_leaf_nodes=data["DecisionTreeClassifier"]["max_leaf_nodes"],
            min_impurity_decrease=data["DecisionTreeClassifier"]["min_impurity_decrease"],
            min_impurity_split=data["DecisionTreeClassifier"]["min_impurity_split"],
            class_weight=data["DecisionTreeClassifier"]["class_weight"],
            ccp_alpha=data["DecisionTreeClassifier"]["ccp_alpha"]
        )
    )
    classifiers.append(
        RandomForestClassifier(
            n_estimators=data["RandomForestClassifier"]["n_estimators"],
            criterion=data["RandomForestClassifier"]["criterion"],
            max_depth=data["RandomForestClassifier"]["max_depth"],
            min_samples_split=data["RandomForestClassifier"]["min_samples_split"],
            min_samples_leaf=data["RandomForestClassifier"]["min_samples_leaf"],
            min_weight_fraction_leaf=data["RandomForestClassifier"]["min_weight_fraction_leaf"],
            max_features=data["RandomForestClassifier"]["max_features"],
            max_leaf_nodes=data["RandomForestClassifier"]["max_leaf_nodes"],
            min_impurity_decrease=data["RandomForestClassifier"]["min_impurity_decrease"],
            min_impurity_split=data["RandomForestClassifier"]["min_impurity_split"],
            bootstrap=data["RandomForestClassifier"]["bootstrap"],
            oob_score=data["RandomForestClassifier"]["oob_score"],
            n_jobs=data["RandomForestClassifier"]["n_jobs"],
            random_state=data["RandomForestClassifier"]["random_state"],
            verbose=data["RandomForestClassifier"]["verbose"],
            warm_start=data["RandomForestClassifier"]["warm_start"],
            class_weight=data["RandomForestClassifier"]["class_weight"],
            ccp_alpha=data["RandomForestClassifier"]["ccp_alpha"],
            max_samples=data["RandomForestClassifier"]["max_samples"]
        )
    )
    classifiers.append(
        MLPClassifier(
            hidden_layer_sizes=data["MLPClassifier"]["hidden_layer_sizes"],
            activation=data["MLPClassifier"]["activation"],
            solver=data["MLPClassifier"]["solver"],
            alpha=data["MLPClassifier"]["alpha"],
            batch_size=data["MLPClassifier"]["batch_size"],
            learning_rate=data["MLPClassifier"]["learning_rate"],
            learning_rate_init=data["MLPClassifier"]["learning_rate_init"],
            power_t=data["MLPClassifier"]["power_t"],
            max_iter=data["MLPClassifier"]["max_iter"],
            shuffle=data["MLPClassifier"]["shuffle"],
            random_state=data["MLPClassifier"]["random_state"],
            tol=data["MLPClassifier"]["tol"],
            verbose=data["MLPClassifier"]["verbose"],
            warm_start=data["MLPClassifier"]["warm_start"],
            momentum=data["MLPClassifier"]["momentum"],
            nesterovs_momentum=data["MLPClassifier"]["nesterovs_momentum"],
            early_stopping=data["MLPClassifier"]["early_stopping"],
            validation_fraction=data["MLPClassifier"]["validation_fraction"],
            beta_1=data["MLPClassifier"]["beta_1"],
            beta_2=data["MLPClassifier"]["beta_2"],
            epsilon=data["MLPClassifier"]["epsilon"],
            n_iter_no_change=data["MLPClassifier"]["n_iter_no_change"],
            max_fun=data["MLPClassifier"]["max_fun"]
        )
    )


# генератор имен файлов
def generate_save_name(classifier, preprocess_method):
    """
    Функция для генерации названия сохранения для весов.
    :param classifier: используемый классификатор
    :param preprocess_method: используемый метод предобработки
    :return: строка с названием файла
    """
    classifier = str(classifier)
    name = "ProjectData/Weights/Classic/"
    file_format = ".pkl"
    now = datetime.datetime.now()
    name = name + classifier + "_" + str(preprocess_method) + "_" + str(now.strftime("%d-%m-%Y %H:%M")) + file_format

    return name



def load_and_preprocess_data(path_to_data, preprocess_method):
    """
    Функция загрузки данных из csv, также выполняет предобработку.
    :param path_to_data: путь к csv
    :param preprocess_method: метод предобработки, сейчас всегда нормализация.
    :return: (X,y), где X - данные, y - лейблы
    """
    data = pd.read_csv(path_to_data, sep=';', thousands=",", low_memory=False)
    del data["path_to_cell"]
    X = data.iloc[:, 1:]
    y = data["Label"]
    X = preprocessing.normalize(X)

    return (X, y)



def get_load_file_stats(y):
    """
    Функция подсчета классов их чостоты в файле.
    :param y: Лист с лейблами
    :return: Лист в формате: (название класса, количество)
    """
    stats = Counter(y)
    # print(len(stats))
    stats = pd.DataFrame.from_dict(stats, orient='index').reset_index()
    stats.rename(columns={"index": "Клетка", 0: "Количество клеток"}, inplace=True)
    stats_list = []
    for i in range(0, len(stats)):
        stats_list.append((stats.loc[i][0], str(stats.loc[i][1])))

    return stats_list



def control_classifiers(X, y, num_classifier, preprocess_method):
    """
    Функция обучения классификаторов
    :param X: данные
    :param y: лейблы
    :param num_classifier: номер классификатора
    :param preprocess_method: способ предобработки
    :return: (expected, predicted) - данные по обученной модели
    """
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
    """
    Функция тестирования моделей
    :param X: данные
    :param y: лейблы
    :param path: путь к весам,для загрузки модели
    :return: (expected, predicted) - данные по тестированию модели
    """
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
    """
    Эксперементальная функция поиска лучших X
    :param X: данные
    :param y: лейблы
    :param best: Лучшие параметры, передавать массив если уже известны индексы лучших
    :return: X, best. X - очищенные данные, best - индексы лучших параметров
    """
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
    """
    Функция прогона по всем классификаторам,вывод в консоль
    :param path_train: путь к обучающей выборке csv-файлу
    :param path_test: путь к тестовой выборке csv-файлу
    :return:
    """
    list_acc = []
    for m in classifiers:
        X, y = load_and_preprocess_data(path_train, 1)
        # X, best = get_best_x(X, y)
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
        # X = get_best_x(X, y, best)

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


def generate_test_report(path_to_data):
    """
    Генератор отчета по тестированию в разрезе каждого объекта, сохраняет csv-файл: ProjectData/OutputData/test_report.csv
    :param path_to_data: путь к тестовой выборке csv-файлу
    :return:
    """
    output_data = []

    data = pd.read_csv(path_to_data, sep=';', thousands=",", low_memory=False)
    paths = data["path_to_cell"]
    del data["path_to_cell"]
    X = data.iloc[:, 1:]
    y = data["Label"]
    X = preprocessing.normalize(X)
    with open("test.pkl", 'rb') as file:
        model = pickle.load(file)
    expected = y
    predicted = model.predict(X)

    for i in range(0, len(paths)):
        output_data.append((paths[i], expected[i], predicted[i]))

    output_df = pd.DataFrame(output_data,columns=["path_to_cell","expected","predicted"])
    output_df.to_csv("ProjectData/OutputData/test_report.csv")


# generate_test_report("ProjectData/Data/test_multiclass.csv")
# load_params()
# get_all_stats("ProjectData/Data/train_multiclass.csv", "ProjectData/Data/test_multiclass.csv")
