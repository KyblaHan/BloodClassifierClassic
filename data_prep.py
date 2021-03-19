import math
import pathlib
import random
from PIL import Image
import sys

TEST_FOLDER = "test"
TRAIN_FOLDER = "train"
TEST_ONE_CLASS_FOLDER = "test_1class"
TRAIN_ONE_CLASS_FOLDER = "train_1class"


# статистика по количеству файлов в папках
def get_folder_stats(path):
    stats = []
    path = pathlib.Path(path)
    data = list(path.glob("*"))

    for folder in data:
        count_images = len(list(folder.glob("*")))
        stats.append((str(folder.name), str(count_images)))

    return stats


# автосоздание паопок
def create_folder(input_path, path_to_save, folder):
    path = pathlib.Path(input_path)
    data = list(path.glob("*"))

    folders = []
    for d in data:
        d = str(d).split("\\")
        folders.append(d[len(d) - 1])

    new_path = pathlib.Path(path_to_save + "\\" + folder)
    new_path.mkdir()

    for f in folders:
        path = new_path.joinpath(f)
        path.mkdir()


# делитель+конвертер в ПНГ
def devide_data(input_path, save_bpm_path, save_png_path, create_png, percent, progressbar):
    completed = 0
    progressbar.setValue(completed)
    max_value = len(list(pathlib.Path(input_path).glob("*/*")))
    step = 100 / max_value

    create_folder(input_path, save_bpm_path, TEST_FOLDER)
    create_folder(input_path, save_bpm_path, TRAIN_FOLDER)

    if create_png == True:
        create_folder(input_path, save_png_path, TEST_FOLDER)
        create_folder(input_path, save_png_path, TRAIN_FOLDER)

    path = pathlib.Path(input_path)
    data = list(path.glob("*"))

    for folder in data:
        images = list(folder.glob("*"))
        random.shuffle(images)
        # print(folder, int(len(images)))

        for img in images[:math.ceil(len(images) * percent / 100)]:
            im = Image.open(img)
            splitted = str(img).split("\\")
            save_path = save_bpm_path + "\\" + TEST_FOLDER + "\\" + splitted[len(splitted) - 2] + "\\" + splitted[
                len(splitted) - 1]
            im.save(save_path, "bmp")

            if create_png == True:
                save_path = save_png_path + "\\" + TEST_FOLDER + "\\" + splitted[len(splitted) - 2] + "\\" + splitted[
                    len(splitted) - 1]
                im.save(save_path+".png", "png")
            completed += step
            progressbar.setValue(completed)

        for img in images[math.ceil(len(images) * percent / 100):]:
            im = Image.open(img)
            splitted = str(img).split("\\")
            save_path = save_bpm_path + "\\" + TRAIN_FOLDER + "\\" + splitted[len(splitted) - 2] + "\\" + splitted[
                len(splitted) - 1]
            im.save(save_path, "bmp")

            if create_png == True:
                save_path = save_png_path + "\\" + TRAIN_FOLDER + "\\" + splitted[len(splitted) - 2] + "\\" + splitted[
                    len(splitted) - 1]
                im.save(save_path+".png", "png")
            completed += step
            progressbar.setValue(completed)

    completed = 100
    progressbar.setValue(completed)
