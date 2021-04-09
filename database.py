# -*- coding: utf-8 -*- #
import sqlite3
import pandas as pd
from collections import Counter
from transliterate import translit

conn = sqlite3.connect('ProjectData/SystemFiles/knowledge_base.db')
cursor = conn.cursor()


def cells_types_append(file):
    data = pd.read_csv(file, sep=';', thousands=",", low_memory=False)
    classes = list(Counter(data["Label"]))
    input_db = []
    i = 0
    for cl in classes:
        input_db.append((str(i), cl))
        i += 1
    cursor.executemany("insert into CellTypes values  (? ,?);", input_db)
    conn.commit()


def sign_types_append(file):
    data = pd.read_csv(file, sep=';', thousands=",", low_memory=False)
    del data["path_to_cell"]
    del data["Label"]
    input_db = []
    i = 0
    for col in data.columns:
        input_db.append((str(i), col))
        i += 1
    cursor.executemany("insert into SignTypes values  (? ,?);", input_db)
    conn.commit()


def cells_append(file):
    data = pd.read_csv(file, sep=';', thousands=",", low_memory=False)
    paths = data["path_to_cell"]
    labels = data["Label"]
    input_db = []
    uniq_labels = list(Counter(labels))
    for i in range(0, len(paths)):
        input_db.append((str(i), paths[i], uniq_labels.index(labels[i])))
    cursor.executemany("insert into Cells values  (? ,?,?);", input_db)
    conn.commit()


def sings_append(file):
    data = pd.read_csv(file, sep=';', thousands=",", low_memory=False)
    del data["path_to_cell"]
    del data["Label"]
    input_db = []
    i = 0
    for index, row in data.iterrows():
        j = 0
        for r in row:
            input_db.append((i, index, j, r))
            i += 1
            j += 1
    cursor.executemany("insert into Sings values  (?, ?, ?, ?);", input_db)
    conn.commit()

# sings_append("ProjectData/Data/all_append_.csv")
conn.close()
