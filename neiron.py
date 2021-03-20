from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import tensorflow as tf
import os
import pathlib
import random
import shutil
import cv2
import numpy as np

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
        tf.keras.layers.Dense(len(get_label_names(PATH_TO_DATA)), activation='softmax')])
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


def test_model(path, use_gpu,path_to_weights):
    activate_gpu(use_gpu)

    ds_test = initiate_dataset(path)
    ds = ds_test.shuffle(buffer_size=count_images(path))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    model = initiate_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()

    # print("Загрузка модели")
    model.load_weights(path_to_weights)
    # print("Анализ точности модели")
    test_loss, test_acc = model.evaluate(ds, verbose=1, steps=TEST_STEPS)
    print('\nТочность на проверочных данных:', test_acc)


def additional_train(path):
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
    checkpoint_path = "Weights_20/cp.ckpt"
    model.load_weights(checkpoint_path)

    print(count_images(path) / BATCH_SIZE)
    steps_per_epoch = tf.math.ceil(count_images(path) / BATCH_SIZE).numpy()

    model.fit(ds,
              epochs=EPOCHS,
              teps_per_epoch=steps_per_epoch,
              # callbacks=[tensorboard_callback, cp_callback],
              callbacks=cp_callback,
              verbose=1)
    print("обучение завершенно")
