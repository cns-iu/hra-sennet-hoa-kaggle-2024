import os
import glob
import numpy as np
import cv2
import tqdm
import argparse  # Import argparse


TESTING = False



def load_volume(dataset):
    path = os.path.join(dataset, "labels", "*.tif")
    dataset = sorted(glob.glob(path))
    volume = None
    target = None

    for z, path in enumerate(tqdm.tqdm(dataset)):
        label = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        label = np.array(label, dtype=np.uint8)  # / 255.
        if target is None:
            target = np.zeros((len(dataset), *label.shape[-2:]), dtype=np.uint8)
        if volume is None:
            volume = np.zeros((len(dataset), *label.shape[-2:]), dtype=np.uint16)
        target[z] = label

        path = path.replace("/labels/", "/images/")
        if "/kidney_3_dense/" in path:
            path = path.replace("kidney_3_dense", "kidney_3_sparse")

        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        image = np.array(image, dtype=np.uint16)
        volume[z] = image
    return volume, target


def save_volume(volume, target, dataset):
    os.makedirs(dataset, exist_ok=True)
    os.makedirs(os.path.join(dataset, "labels"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "images"), exist_ok=True)

    for z in tqdm.tqdm(range(volume.shape[0])):
        cv2.imwrite(os.path.join(dataset, "images", f"{z:04d}.tif"), volume[z])
        cv2.imwrite(os.path.join(dataset, "labels", f"{z:04d}.tif"), target[z])

def main():
    for dataset in glob.glob(os.path.join(dataset_path, "train", "kidney*")):
        dataset_xz = dataset + "_xz"
        dataset_zy = dataset + "_zy"
        skip_xz = os.path.exists(dataset_xz)
        skip_zy = os.path.exists(dataset_zy)
        if not skip_xz or not skip_zy:
            print("load", dataset)
            volume, target = load_volume(dataset)
        if not skip_xz:
            print("save", dataset_xz)
            volume_xz, target_xz = volume.transpose((1, 2, 0)), target.transpose((1, 2, 0))
            if not TESTING:
                save_volume(volume_xz, target_xz, dataset_xz)
        else:
            print("skipping", dataset_xz)
        if not skip_zy:
            print("save", dataset_zy)
            volume_zy, target_zy = volume.transpose(2, 0, 1), target.transpose(2, 0, 1)
            if not TESTING:
                save_volume(volume_zy, target_zy, dataset_zy)
        else:
            print("skipping", dataset_zy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process blood vessel segmentation data.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    main()
