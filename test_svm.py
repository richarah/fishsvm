import joblib
import sys
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from skimage.transform import resize

"""
Calculates classifier accuracy by comparing SVC predictions to ground truth values specified in Excel sheet.
"""


tile_dims = 512
blank_threshold = 240  # Images with median brightness above threshold will be marked as blank
relevant_class = "atrium" # Class to search for using filter SVM

class_clf_file = joblib.load(r"") # Classifies tiles
organ_clf_file = joblib.load(r"") # Filters out tiles not containing atrium
test_dir = r""

organ_clf = organ_clf_file.clf
class_clf = class_clf_file.clf
clf_dims = class_clf_file.dims # Different from tile dims - tuple of dimensions used by clf

excel_path = r""
excel = pd.read_excel(excel_path)
excel_df = pd.DataFrame(excel)


def most_common(arr):
    return max(set(arr), key=arr.count)


def median_brightness(im):
    return ImageStat.Stat(im).median[0]


def not_blank(im):
    if median_brightness(im) < blank_threshold:
        return True
    else:
        return False

def slice_image(file, tile_size):
    tiles = []
    try:
        im = Image.open(file)

        # Crop to width & height divisible by tile size
        new_width = int(tile_size * (im.size[0] // tile_size))
        new_height = int(tile_size * (im.size[1] // tile_size))
        im = im.crop((0, 0, new_width, new_height))

        tiles = []

        x = 0
        y = 0

        while y < im.size[1]:
            while x < im.size[0]:
                tile = im.crop((x, y, x + tile_size, y + tile_size))
                if not_blank(tile):
                    tile = tile.convert("RGB")
                    tiles.append(tile)
                x += tile_size
            y += tile_size
            x = 0

    except:
        print(sys.exc_info())

    return tiles

def reshape_tile(tile, dims):
    arr = np.asarray(tile)
    im_resized = resize(arr, dims, anti_aliasing=True, mode='reflect')
    im_flat = np.array(im_resized.flatten())
    reshaped = im_flat.reshape(1, -1)
    return reshaped


def filter_tiles(tiles, clf, dims, desired_class):
    filtered_tiles = []
    for tile in tiles:
        try:
            reshaped = reshape_tile(tile, dims)
            prediction = clf.predict(reshaped)
            if desired_class == prediction:
                filtered_tiles.append(tile)
        except:
            print(sys.exc_info())
    return filtered_tiles


def field_index(df, field_name):
    return (list(df.columns.values)).index(field_name)


def main():

    total = 0
    correct = 0

    for index, row in excel_df.iterrows():

        try:
            atrium_old = row[field_index(excel_df, "atrium_org")]
            atrium_new = row[field_index(excel_df, "atrium_new")]
            entry = test_dir + r"/" + str(row[0]) + ".png"

            if os.path.isfile(entry) and (pd.notna(atrium_old) or pd.notna(atrium_new)):
                tiles = slice_image(entry, tile_dims)
                filtered_tiles = filter_tiles(tiles, organ_clf, clf_dims, relevant_class)

                predictions = []
                for tile in filtered_tiles:
                    reshaped = reshape_tile(tile, clf_dims)
                    # Since the class names are class x 10 (long story), they are divided by 10 here
                    predictions.append(float(class_clf.predict(reshaped)) / 10)

                # If no atrium tiles are found, skip to next image
                if len(filtered_tiles) == 0:
                    continue

                prediction = most_common(predictions)

                # Use new classification if available
                truth = None
                if pd.notna(atrium_new):
                    truth = atrium_new
                else:
                    truth = atrium_old

                if prediction == truth:
                    correct += 1
                total += 1

                print("Total predictions:", total, "\nCorrect predictions:", correct)
                print("Accuracy so far:", str('{0:.2f}'.format((correct / total) * 100) + "%\n"))

        except:
            print(sys.exc_info())
    print("\nTesting complete.\nFinal accuracy score:", str('{0:.2f}'.format((correct / total) * 100) + "%\n"))
main()