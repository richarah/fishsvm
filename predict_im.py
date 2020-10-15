import matplotlib.pyplot as plt
import numpy as np
import imghdr
import joblib
import sys
from skimage.transform import resize

import easygui

def read_file(file_path, dims):
    if imghdr.what(file_path) == "jpeg":
        im = plt.imread(file_path)
        im_resized = resize(im, dims, anti_aliasing=True, mode='reflect')
        im_flat = np.array(im_resized.flatten())
        im_reshaped = im_flat.reshape(1, -1)
        return im_reshaped

loaded = joblib.load(r"")
clf = loaded.clf

while True:
    try:
        path = easygui.fileopenbox()

        print("Prediction:", str(float(clf.predict(read_file(path, loaded.dims))) / 10))
    except:
        print(sys.exc_info())
