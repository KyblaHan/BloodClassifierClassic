import pandas as pd
import json

from sklearn.svm import LinearSVC

with open("..\params.json") as json_file:
    data = json.load(json_file)
    print(data["SVC"])
    # data =data["LinearSVC"]
    # LinearSVC=LinearSVC(
    #             penalty=data.get("penalty"),
    #             loss=data.get("loss"),
    #             dual=data.get("dual"),
    #             tol=data.get("tol"),
    #             C=data.get("C"),
    #             multi_class=data.get("multi_class"),
    #             fit_intercept=data.get("fit_intercept"),
    #             intercept_scaling=data.get("intercept_scaling"),
    #             class_weight=data.get("class_weight"),
    #             verbose=data.get("verbose"),
    #             random_state=data.get("random_state"),
    #             max_iter=data.get("max_iter")
    #         )
