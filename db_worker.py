from peewee import *
import pandas as pd


class UnknownField(object):
    def __init__(self, *_, **__): pass


class BaseModel(Model):
    class Meta:
        database = SqliteDatabase('ProjectData/SystemFiles/knowledge_base.db')


class CellTypes(BaseModel):
    id = AutoField(column_name='Id')
    name = CharField(column_name='Name')

    class Meta:
        table_name = 'CellTypes'


class Cells(BaseModel):
    cell_type = ForeignKeyField(column_name='CellTypeId', field='id', model=CellTypes, null=True)
    id = AutoField(column_name='Id')
    path = CharField(column_name='Path', unique=True)

    class Meta:
        table_name = 'Cells'


class SignTypes(BaseModel):
    id = AutoField(column_name='Id')
    name = CharField(column_name='Name')

    class Meta:
        table_name = 'SignTypes'


class Sings(BaseModel):
    cell = ForeignKeyField(column_name='CellId', field='id', model=Cells)
    id = AutoField(column_name='Id')
    sign_type = ForeignKeyField(column_name='SignTypeId', field='id', model=SignTypes)
    value = DecimalField(column_name='Value')

    class Meta:
        table_name = 'Sings'


# ТУТ БУДЕТ НАШ КОД РАБОТЫ С БАЗОЙ ДАННЫХ


def get_data():
    """
    Выгружает данные из БД в DataFrame
    :return: DataFrame - ["path", "CellType", "Sing", "Value"]
    """
    database = SqliteDatabase('ProjectData/SystemFiles/knowledge_base.db')
    cursor = database.cursor()
    cursor.execute(
        "SELECT  t3.Path, t4.Name, t2.Name, t1.Value FROM Sings AS t1 INNER JOIN SignTypes AS t2 ON (t1.SignTypeId = t2.Id) INNER JOIN Cells AS t3 ON (t1.CellId = t3.Id) INNER JOIN CellTypes AS t4 ON (t3.CellTypeId = t4.Id)")
    results = cursor.fetchall()
    database.close()
    df = pd.DataFrame(results, columns=["path", "CellType", "Sing", "Value"])
    return df


