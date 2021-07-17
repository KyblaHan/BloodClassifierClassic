import pathlib

import pandas as pd

path = "C:\_Programming\BloodClassifierClassic\ProjectData\OutputData\Classic\Train"



count_test = [1, 1475, 26, 1, 148, 1, 57, 56, 126, 257, 9, 7, 168, 156]
count_train = [3,5898,104,3,589,1,227,224,502,1025,34,24,672,623,]

def all_path(path):
    path = pathlib.Path(path)
    paths = path.glob('*/')

    for p in paths:
        acc_counter(p,count_test)

def acc_counter(path,count):
    print(path)
    file = pd.read_csv(path)
    expected = file["expected"]
    predicted = file["predicted"]
    start = 0
    accTotal=0
    for end in count:
        end = start + end
        right = 0
        for i in range(start, end):
            if expected[i] == predicted[i]:
                right = right + 1

        acc = right/end
        accTotal+=acc
        print(acc)

    print(accTotal/14)


all_path(path)
