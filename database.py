import sqlite3
import pandas as pd
from collections import Counter

conn = sqlite3.connect('ProjectData/SystemFiles/knowledge_base.db')
cursor = conn.cursor()

def import_db(file):
    data = pd.read_csv(file, sep=';', thousands=",", low_memory=False)


def transliteration(text):
    cyrillic = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    latin = 'a|b|v|g|d|e|e|zh|z|i|i|k|l|m|n|o|p|r|s|t|u|f|kh|tc|ch|sh|shch||y||e|iu|ia|A|B|V|G|D|E|E|Zh|Z|I|I|K|L|M|N|O|P|R|S|T|U|F|Kh|Tc|Ch|Sh|Shch||Y||E|Iu|Ia'.split('|')
    return text.translate({ord(k):v for k,v in zip(cyrillic,latin)})

def cells_types_append(file):
    data = pd.read_csv(file, sep=';', thousands=",", low_memory=False)
    classes = list(Counter(data["Label"]))
    input_db = []
    i =0
    for cl in classes:
        input_db.append((str(i),cl))
        i+=1
    cursor.executemany("insert into CellTypes values  (? ,?);", input_db)
    conn.commit()

def sign_types_append(file):
    data = pd.read_csv(file, sep=';', thousands=",", low_memory=False,encoding="ascii")
    del data["path_to_cell"]
    del data["Label"]
    input_db = []
    i = 0
    for col in data.columns:
        input_db.append((str(i), transliteration(col)))
        print(transliteration(col))
        i+=1

    # cursor.executemany("insert into SignTypes values  (? ,?);", input_db)
    # conn.commit()

# import_db("ProjectData/Data/all_append.csv")
sign_types_append("ProjectData/Data/all_append.csv")

conn.close()
