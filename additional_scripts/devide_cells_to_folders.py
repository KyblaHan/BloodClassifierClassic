import pathlib
from PIL import Image
from collections import Counter

path = pathlib.Path(r"C:\Users\kybla\Downloads\выделенные клетки от 24-03-2020\cells")
all_image_paths = list(path.glob('*'))

new_folders = []
for file in all_image_paths:
    new_folders.append(str(file).split("-")[3])

new_folders = list(Counter(new_folders))

for folder in new_folders:
    path_save = pathlib.Path(r"C:\Users\kybla\Downloads\выделенные клетки от 24-03-2020")
    path_save = path_save.joinpath(folder)
    path_save.mkdir()

for file in all_image_paths:
    print(file)
    folder = str(file).split("-")[3]
    path_save = pathlib.Path(r"C:\Users\kybla\Downloads\выделенные клетки от 24-03-2020")
    path_save = str(path_save.joinpath(folder))

    im = Image.open(file)
    im_name = str(file).split("\\")[6]
    path_save = path_save + "\\" + im_name
    im.save(path_save, "bmp")

# path = new_path.joinpath(f)
#         path.mkdir()
