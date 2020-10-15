import pathlib
import matplotlib.pyplot as plt
import numpy as np
import imghdr
import joblib
import sys
import os

from sklearn import svm
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.transform import resize


"""
For simple, SVM-based image classifier training.
Originally developed for fish analysis (hence the name), but plenty of other use cases abound.
"""

# NOTE: At the time of writing, this project is a work in progress - please expect the unexpected.
# TODO: refactor save code
# TODO: test rigorously for save path bug (could not be replicated?)

def format_dataset(dir_path, dims):
    data_dir = pathlib.Path(dir_path)
    targets = [d for d in data_dir.iterdir() if d.is_dir()]

    data = []
    target = []
    target_names = [t.name for t in targets]
    images = []

    for t in targets:
        im_target = str(os.path.basename(t)) # Category - nothing to do with felines
        for f in t.iterdir():
            if imghdr.what(f) == "jpeg":
                im = resize(plt.imread(f), dims, anti_aliasing=True, mode='reflect')
                data.append(im.flatten())
                images.append(im)
                target.append(im_target)
    data = np.array(data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=data, target=target, target_names=target_names, images=images)


def create_dataset(data_path, dims):
    print("Creating dataset...")
    data = format_dataset(data_path, dims)
    print("Dataset created.")
    return data


def load_dataset(path):
    print("Loading dataset...")
    data = joblib.load(path)
    print("Dataset loaded.")

    return data


def train_clf(x, y, n_jobs, params ={'kernel': ('linear', 'rbf'), 'C': [1, 10]}):

    print("Training SVM. Please stand by...")
    svc = svm.SVC(class_weight='balanced')
    clf = GridSearchCV(svc, params, n_jobs=n_jobs, verbose=True)
    clf.fit(x, y)
    print("Training complete.")

    return clf


def save_ds(ds):
    while True:
        try:
            save_path = str(input("Save to path: "))
            if os.path.isdir(save_path):
                filename = str(input("Save as: "))
                joblib.dump(ds, str(save_path + "/" + filename))
                print("Model successfully saved.")
                return

        except:
            print("An error occured. Please try again.")


def save_clf(clf, dims):
    while True:
        try:
            save_path = str(input("Save to path: "))
            if os.path.isdir(save_path):
                filename = str(input("Save as: "))
                savefile = Bunch(clf=clf, dims=dims)
                joblib.dump(savefile, str(save_path + "/" + filename))
                print("Model successfully saved.")
                return
        except:
            print("An error occured. Please try again.")


def main():
    print("FishSVM v1.0.0\nby Richard Alexander Haydon")

    n_jobs = 8  # Max workers for SVM
    dims = (64, 64)
    test_size = 0.2

    while True:
        try:
            option = int(input("Press 1 to create a new dataset.\n"
                               "Press 2 to train from an existing dataset.\n"
                               "Press 3 to train directly from a folder hierarchy.\n\n> "))
            if option == 1:
                data_path = input("Path to data folder: ")
                dataset = create_dataset(data_path, dims)
                save_ds(dataset)

            elif option == 2:
                load_path = input("Path to dataset: ")
                dataset = load_dataset(load_path)
                x_train, x_val, y_train, y_val = train_test_split(dataset.data, dataset.target, test_size=test_size)
                del dataset
                clf = train_clf(x_train, y_train, n_jobs)
                save_clf(clf, dims)

            elif option == 3:
                data_path = input("Path to data folder: ")
                dataset = create_dataset(data_path, dims)
                x_train, x_val, y_train, y_val = train_test_split(dataset.data, dataset.target, test_size=test_size)
                del dataset
                clf = train_clf(x_train, y_train, n_jobs)
                save_clf(clf, dims)
            else:
                print("Invalid input.")
        except:
            print(sys.exc_info())


main()
