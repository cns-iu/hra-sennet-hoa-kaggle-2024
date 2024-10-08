import cv2
import gc
import matplotlib.pyplot as plot
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from collections import deque
from glob import glob
from itertools import starmap


PATCH_WIDTH = 128
PATCH_HEIGHT = 128
PATCH_DEPTH = 32

model = tf.keras.models.load_model('../data/models/model-0.h5')


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def submit(data_folder_path, submission_file):
    data_paths = tuple(sorted(glob(f"{data_folder_path}/images/*.tif")))

    depth = len(data_paths)
    height, width = np.shape(cv2.imread(data_paths[0], cv2.IMREAD_GRAYSCALE))

    def get_min_max():
        values = np.zeros((depth, height, width), dtype=np.uint8)
        for i in range(depth):
            values[i] = cv2.imread(data_paths[i], cv2.IMREAD_GRAYSCALE)

        values = np.ravel(values)

        index = -int(len(values) * 0.001)
        max = np.partition(values, index)[index]

        values[values > max] = max

        index = int(len(values) * 0.001)
        min = np.partition(values, index)[index]

        return min / 255, max / 255

    min, max = get_min_max()
    print(depth, height, width, min, max)

    def get_predict():
        result = np.zeros(tuple(starmap(lambda patch_size, size: int(patch_size * np.ceil(size / patch_size)),
                                        ((PATCH_DEPTH, depth), (PATCH_HEIGHT, height), (PATCH_WIDTH, width)))),
                          dtype=np.uint8)

        for i in range(0, depth, PATCH_DEPTH):
            print(i)
            volumetric_image = np.stack(tuple(map(lambda path: (cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 256).astype(np.float32) if path is not None else np.zeros((height, width), dtype=np.float32),
                                                  map(lambda index: data_paths[index] if index < depth else None,
                                                      range(i, i + PATCH_DEPTH)))))
            volumetric_image = np.pad(volumetric_image, ((0, PATCH_DEPTH - np.shape(volumetric_image)[0]), (0, np.shape(result)[1] - np.shape(volumetric_image)[1]), (0, np.shape(result)[2] - np.shape(volumetric_image)[2])))

            # クリッピング
            volumetric_image[volumetric_image > max] = max
            volumetric_image[volumetric_image < min] = min

            # TODO: 正則化する？

            for j in range(0, height, PATCH_HEIGHT):
                for k in range(0, width, PATCH_WIDTH):
                    x = np.reshape(volumetric_image[:, j: j + PATCH_HEIGHT, k: k + PATCH_WIDTH], (1, PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, 1))
                    y = model.predict_on_batch(x)

                    result[i: i + PATCH_DEPTH, j: j + PATCH_HEIGHT, k: k + PATCH_WIDTH] = (np.reshape(y, (PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH)) > 0.25)  # 後処理で削るので、ここでは、閾値を小さな値にしておきます。

                    del y, x
                    gc.collect()

            del volumetric_image
            gc.collect()

        return result[0: depth, 0: height, 0: width]

    def get_blood_vessel(predict):
        checked = np.zeros((depth, height, width), dtype=np.uint8)
        number_of_predict = np.count_nonzero(predict)
        deltas = tuple(filter(lambda x: x != (0, 0, 0),
                              map(lambda x: (x[0] - 1, x[1] - 1, x[2] - 1),
                                  zip(*np.where(np.ones((3, 3, 3)))))))

        for start_z, start_y, start_x in zip(*np.where(predict)):
            if checked[start_z, start_y, start_x]:
                continue

            result = np.zeros((depth, height, width), dtype=np.uint8)

            stack = deque()
            stack.append((start_z, start_y, start_x))

            result[start_z, start_y, start_z] = True

            n = 1
            while stack:
                z, y, x = stack.pop()

                for delta_x, delta_y, delta_z in deltas:
                    next_z = z + delta_z
                    next_y = y + delta_y
                    next_x = x + delta_x

                    if not (0 <= next_z < depth and 0 <= next_y < height and 0 <= next_x < width):
                        continue

                    if not predict[next_z, next_y, next_x]:
                        continue

                    if result[next_z, next_y, next_x]:
                        continue

                    result[next_z, next_y, next_x] = True
                    stack.append((next_z, next_y, next_x))

                    n += 1

            checked |= result

            print(f"{n}\t{number_of_predict}\t{n / number_of_predict}")

            if n < number_of_predict * 0.2:
                continue

            return result

        return predict

    predict = get_predict()

    # with open(f"{data_folder_path.split('/')[-1]}_predict.pickle", mode='wb') as f:
    #     pickle.dump(predict, f)

    blood_vessel = get_blood_vessel(predict)

    # with open(f"{data_folder_path.split('/')[-1]}.pickle", mode='wb') as f:
    #     pickle.dump(blood_vessel, f)

    for i in range(np.shape(blood_vessel)[0]):
        image = blood_vessel[i]

        print(f"{data_folder_path.split('/')[-1]}_{data_paths[i].split('/')[-1].split('.')[0]},{rle_encode(image) if np.count_nonzero(image) else '1 0'}", file=submission_file)


with open('submission.csv', mode='w') as f:
    print('id,rle', file=f)

    for data_folder_path in sorted(glob('../data/blood-vessel-segmentation/test/*')):
        submit(data_folder_path, f)

    # submit('../data/blood-vessel-segmentation/train/kidney_1_dense', f)
    # submit('../data/blood-vessel-segmentation/train/kidney_3_sparse', f)


# submission_data_frame = pd.read_csv('/kaggle/working/submission.csv')
# submission_data_frame.head()
