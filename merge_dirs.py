from genericpath import isdir
from os import listdir
from os.path import isfile, isdir, join
from shutil import copyfile

ROOT_DIR_PATH = '/raid/asarkisov/MyProjects/stylegan2-ada/data'
DEST_DIR_PATH = '/raid/asarkisov/MyProjects/Image-Captioning/data/thenounproject_images'


if __name__ == "__main__":
    images_dir_paths = [join(ROOT_DIR_PATH, path) for path in listdir(ROOT_DIR_PATH) if isdir(join(ROOT_DIR_PATH, path)) and (path.split("_")[0] == "thenounproject")]
    for images_dir_path in images_dir_paths:
        file_names = [f for f in listdir(images_dir_path) if isfile(join(images_dir_path, f)) and (f.split(".")[1] == "png" or f.split(".")[1] == "jpg")]
        for file_name in file_names:
            copyfile(f'{images_dir_path}/{file_name}', f'{DEST_DIR_PATH}/{file_name}')