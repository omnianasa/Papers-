import os, shutil
from sklearn.model_selection import train_test_split
def split_data(source_folder, target_root):
    allFiles = os.listdir(source_folder)
    train, temp = train_test_split(allFiles, test_size=.1)
    val, test = train_test_split(temp, test_size=.111)

    for folder_name, files in zip(["train", "val", "test"], [train, val, test]):
        target_folder = os.path.join(target_root, folder_name, os.path.basename(source_folder))
        os.makedirs(target_folder, exist_ok=True)
        for file in files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))

split_data('dataset/cat', 'cats_and_dogs')
split_data('dataset/dog', 'cats_and_dogs')
