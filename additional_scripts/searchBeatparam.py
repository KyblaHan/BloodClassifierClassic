from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
import scipy.stats
import pandas as pd
from sklearn import metrics


def load(path_to_data):
    # path_to_data = r"C:\_Programming\BloodClassifierClassic\ProjectData\Data\test_multiclass.csv"
    data = pd.read_csv(path_to_data, sep=';', thousands=",", low_memory=False)
    del data["path_to_cell"]
    # Вытаскиваем данные
    X = data.iloc[:, 1:]
    # Вытаскиваем лейблы
    y = data["Label"]
    # X = preprocessing.scale(X)
    X = preprocessing.normalize(X)
    return (X, y)


test_best = 0
n_best = 1

best = []

neirons = [512, 256, 128, 64, 32, 16]

for n in neirons:
    print("--->", n)
    # path_to_data =

    X, y = load(r"/ProjectData/Data/train_multiclass.csv")
    # print(len(y))
    # , batch_size = int(len(y) / 1),
    mlp = MLPClassifier(hidden_layer_sizes=(512, 32, n,), max_iter=1000, solver="adam", activation="relu",
                        verbose=False, random_state=1)

    # batch_size = int(len(y) / 10)
    mlp.fit(X, y)

    expected = y
    predicted = mlp.predict(X)
    train_acc = metrics.accuracy_score(expected, predicted)

    X, y = load(r"/ProjectData/Data/test_multiclass.csv")
    expected = y
    predicted = mlp.predict(X)
    test_acc = metrics.accuracy_score(expected, predicted)
    if test_acc > test_best:
        test_best = test_acc
        n_best = n
        print("!! n=", n, "train_acc=", train_acc, "test_acc=", test_acc)
        print(metrics.classification_report(expected, predicted, zero_division=0))
        best.append((n_best, test_best))
    print("train_acc=", train_acc, "test_acc=", test_acc)
    print("best: ", best)

print(best)
# 1  [(700, 0.8190649666059502), (699, 0.8306010928961749), (657, 0.8312082574377656), (597, 0.833029751062538), (531, 0.8336369156041287)]
# 2  [(531, 0.8208864602307225), (529, 0.8221007893139041), (528, 0.8245294474802671), (525, 0.8263509411050395), (523, 0.8293867638129934), (491, 0.8312082574377656), (480, 0.8318154219793564), (363, 0.8324225865209471), (301, 0.8342440801457195)]
# best:  [(301, 0.7856709168184578), (300, 0.8142076502732241), (299, 0.8269581056466302), (298, 0.827565270188221), (290, 0.8299939283545841), (215, 0.8312082574377656), (190, 0.8318154219793564)]
