from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import timeit

import tensorflow as tf
import os
import pathlib
import random
import shutil
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# PATH_TO_DATA = r"C:\_Programming\DataSets\First_png"
# PATH_TEMP = r"C:\_Programming\DataSets\temp"
# PATH_TO_TRAIN = "\TrainDataSet_20"
# PATH_TO_TEST = "\TestDataSet_20"
# LABEL_NAMES = ['Базофил', 'Бласты', 'Лимфоцит', 'Метамиелоцит', 'Миелоцит', 'Моноцит', 'Нормобласты',
#                'Палочкоядерный нейтрофил', 'Промиелоцит', 'Сегментноядерный нейтрофил', 'Эозинофил']
# EPOCHS = 10
# act = 'relu'
TEST_STEPS = 100
# checkpoint_path = "Weights/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/cp.ckpt"

checkpoint_path = "ProjectData//Weights/Neiron//cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)
# Создаем коллбек сохраняющий веса модели
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


def generate_checkpoint_callback():
    pass


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def initiate_dataset(path):
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = sorted(item.name for item in path.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_label_ds = image_label_ds.cache()
    return image_label_ds


def initiate_dataset_one_class(path):
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = sorted(item.name for item in path.glob('../*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    # print(all_image_labels)
    # print(all_image_paths)
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_label_ds = image_label_ds.cache()
    # print(image_label_ds)
    return image_label_ds


def count_images(path):
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*/*'))
    return len(all_image_paths)


def get_label_names(path):
    path = pathlib.Path(path)
    return sorted(item.name for item in path.glob('*/') if item.is_dir())


def initiate_model(PATH_TO_DATA):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(192, 192, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(14, activation='softmax')])
    return model


def activate_gpu(use_gpu):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def start_train(path, use_gpu, epochs):
    activate_gpu(use_gpu)

    image_label_ds = initiate_dataset(path)
    ds = image_label_ds.shuffle(buffer_size=count_images(path))
    # print(count_images(path))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    model = initiate_model(path)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    # print(count_images(path) / BATCH_SIZE)
    steps_per_epoch = tf.math.ceil(count_images(path) / BATCH_SIZE).numpy()

    model.fit(ds,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              callbacks=cp_callback,
              verbose=1)
    print("обучение завершенно")


def test_model(path, use_gpu, path_to_weights):
    activate_gpu(use_gpu)
    count = len(list(pathlib.Path(path).glob('*/*')))
    TEST_STEPS = count
    ds_test = initiate_dataset(path)
    # print("!!")
    ds = ds_test.shuffle(buffer_size=count_images(path))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    model = initiate_model(path)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()

    # print("Загрузка модели")
    model.load_weights(path_to_weights)
    # print("Анализ точности модели")
    test_loss, test_acc = model.evaluate(ds, verbose=1, steps=TEST_STEPS)
    print('\nТочность на проверочных данных:', test_acc)
    return test_acc


def test_model_one_class(path, use_gpu, path_to_weights):
    activate_gpu(use_gpu)

    pathh = pathlib.Path(path)
    paths = list(pathh.glob('*'))
    print(list(pathh.glob('*')))

    output_data = []
    for x in paths:

        count = len(list(pathlib.Path(x).glob('*')))
        print(count)
        if count >= 100:
            TEST_STEPS = 100
        else:
            TEST_STEPS = count

        TEST_STEPS = count

        ds_test = initiate_dataset_one_class(str(x))
        # print("!!")
        ds = ds_test.shuffle(buffer_size=count_images(path))
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        model = initiate_model(path)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=["accuracy"])

        # model.summary()

        # print("Загрузка модели")
        model.load_weights(path_to_weights)
        # print("Анализ точности модели")
        test_loss, test_acc = model.evaluate(ds, verbose=1, steps=TEST_STEPS)
        output_data.append((os.path.basename(x), test_acc))
    print(output_data)
    return output_data


def additional_train(path, use_gpu, epochs, path_to_weights):
    activate_gpu(use_gpu)

    image_label_ds = initiate_dataset(path)
    ds = image_label_ds.shuffle(buffer_size=count_images(path))
    print(count_images(path))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    model = initiate_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    print("Загрузка модели")
    # checkpoint_path = "Weights_20/cp.ckpt"
    model.load_weights(path_to_weights)

    print(count_images(path) / BATCH_SIZE)
    steps_per_epoch = tf.math.ceil(count_images(path) / BATCH_SIZE).numpy()

    model.fit(ds,
              epochs=epochs,
              teps_per_epoch=steps_per_epoch,
              # callbacks=[tensorboard_callback, cp_callback],
              callbacks=cp_callback,
              verbose=1)
    print("обучение завершенно")


def train_and_test(path_to_train, path_to_test, use_gpu, epochs, test_every_epoch="short"):
    """
    Генерация отчетов в разрезе каждой из эпох

    :param path_to_train:
    :param path_to_test:
    :param use_gpu:
    :param epochs:
    :param test_every_epoch: full - отчеты  cells+short,cells - точность в разрезе каждого класса, short - генереруется отчет по общей точности
    :return:
    """
    activate_gpu(use_gpu)
    cells_report = []
    short_report = []
    time_report = []
    for i in range(3, epochs, 1):
        train_time = timeit.default_timer()
        start_train(path_to_train, use_gpu, i)
        train_time = timeit.default_timer() - train_time
        if test_every_epoch == "full" or test_every_epoch == "cells":

            train_data = test_model_one_class(path_to_train, use_gpu,
                                              "ProjectData//Weights/Neiron//cp.ckpt")



            test_data = test_model_one_class(path_to_test, use_gpu,
                                             "ProjectData//Weights/Neiron//cp.ckpt")


            cells_report.append(get_values(i, "train", train_data))
            cells_report.append(get_values(i, "test", test_data))
            df = pd.DataFrame(cells_report,
                              columns=["epoch", "type", 'Basophil', 'Blasts', 'Eosinophil', 'Lymphoblast', 'Lymphocyte',
                                       'Megakaryocyte', 'Metamyelocyte', 'Monocyte', 'Myelocyte', 'Normoblasts',
                                       'Plasma cell',
                                       'Promyelocyte', 'Rod-shaped neutrophil', 'Segmented neutrophil'])
            df.to_csv(r"ProjectData/OutputData/Neiron/tensor_cells_report_" + str(i) + ".csv")

        if test_every_epoch == "full" or test_every_epoch == "short":
            test_train_time = timeit.default_timer()
            train_data = test_model(path_to_train, use_gpu,
                                    "ProjectData//Weights/Neiron//cp.ckpt")
            test_train_time = timeit.default_timer() - test_train_time
            test_test_time = timeit.default_timer()
            test_data = test_model(path_to_test, use_gpu,
                                   "ProjectData//Weights/Neiron//cp.ckpt")
            test_test_time = timeit.default_timer() - test_test_time

            time_report.append((i,train_time,test_train_time,test_test_time))
            df = pd.DataFrame(time_report, columns=["epoch", "train_time", 'test_train_time','test_test_time'])
            df.to_csv(r"ProjectData/OutputData/Neiron/tensor_time_report_" + str(i) + ".csv")

            short_report.append((i, "train", train_data))
            short_report.append((i, "test", test_data))
            df = pd.DataFrame(short_report,
                              columns=["epoch", "type", 'acc'])
            df.to_csv(r"ProjectData/OutputData/Neiron/tensor_short_report_" + str(i) + ".csv")



    if test_every_epoch == "full" or test_every_epoch == "cells":
        df = pd.DataFrame(cells_report,
                          columns=["epoch", "type", 'Basophil', 'Blasts', 'Eosinophil', 'Lymphoblast', 'Lymphocyte',
                                   'Megakaryocyte', 'Metamyelocyte', 'Monocyte', 'Myelocyte', 'Normoblasts',
                                   'Plasma cell',
                                   'Promyelocyte', 'Rod-shaped neutrophil', 'Segmented neutrophil'])
        df.to_csv(r"ProjectData/OutputData/Neiron/tensor_cells_report.csv")
    if test_every_epoch == "full" or test_every_epoch == "short":
        df = pd.DataFrame(short_report,
                          columns=["epoch", "type", 'acc'])
        df.to_csv(r"ProjectData/OutputData/Neiron/tensor_short_report.csv")


# initiate_dataset_one_class(r"C:\_Programming\_DataSets\Multiclass\png_devided_data\test\Myelocyte")
# test_model_one_class(r"C:\_Programming\_DataSets\Two class\png_devided_data\test", True, "ProjectData//Weights/Neiron//cp.ckpt")
# test_model(r"C:\_Programming\_DataSets\Multiclass\png_devided_data\test",True,"ProjectData//Weights/Neiron//cp.ckpt")


# start_train(r"C:\_Programming\_DataSets\Multiclass\png_devided_data_append\train", True,1)

def get_values(epoch, data_type, data_list):
    output_list = []
    output_list.append(epoch)
    output_list.append(data_type)
    for d in data_list:
        output_list.append(d[1])
    return output_list


def train_and_test_no_control():
    data_list = []
    for i in range(1, 100, 1):
        start_train(r"C:\_Programming\_DataSets\Multiclass\png_devided_data_append\train", True, i)
        train_data = test_model_one_class(r"C:\_Programming\_DataSets\Multiclass\png_devided_data_append\train", True,
                                          "ProjectData//Weights/Neiron//cp.ckpt")
        test_data = test_model_one_class(r"C:\_Programming\_DataSets\Multiclass\png_devided_data_append\test", True,
                                         "ProjectData//Weights/Neiron//cp.ckpt")

        data_list.append(get_values(i, "train", train_data))
        data_list.append(get_values(i, "test", test_data))
        df = pd.DataFrame(data_list,
                          columns=["epoch", "type", 'Basophil', 'Blasts', 'Eosinophil', 'Lymphoblast', 'Lymphocyte',
                                   'Megakaryocyte', 'Metamyelocyte', 'Monocyte', 'Myelocyte', 'Normoblasts',
                                   'Plasma cell',
                                   'Promyelocyte', 'Rod-shaped neutrophil', 'Segmented neutrophil'])
        df.to_csv(r"ProjectData/OutputData/tensor_report_" + str(i) + ".csv")

    df = pd.DataFrame(data_list,
                      columns=["epoch", "type", 'Basophil', 'Blasts', 'Eosinophil', 'Lymphoblast', 'Lymphocyte',
                               'Megakaryocyte', 'Metamyelocyte', 'Monocyte', 'Myelocyte', 'Normoblasts', 'Plasma cell',
                               'Promyelocyte', 'Rod-shaped neutrophil', 'Segmented neutrophil'])
    df.to_csv(r"ProjectData/OutputData/tensor_report.csv")


# train_and_test(r"C:\_Programming\_DataSets\Multiclass\png_devided_data_append\train",
#                 r"C:\_Programming\_DataSets\Multiclass\png_devided_data_append\test", False, 21,"short")

# train_and_test_no_control()
