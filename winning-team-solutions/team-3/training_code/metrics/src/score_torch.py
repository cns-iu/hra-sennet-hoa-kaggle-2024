import numpy as np
import pandas as pd
import argparse

import time
import torch
import torch.nn as nn
from PIL import Image
from official_metric import create_table_neighbour_code_to_surface_area

di = '/kaggle/input/blood-vessel-segmentation'
device = torch.device('cuda')


def rle_decode(mask_rle: str, shape: tuple) -> np.array:
    """
    Decode rle string
    https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script
    https://www.kaggle.com/stainsby/fast-tested-rle

    Args:
      mask_rle: run length (rle) as string
      shape: (height, width) of the mask

    Returns:
      array[uint8], 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def compute_area(y: list, unfold: nn.Unfold, area: torch.Tensor) -> torch.Tensor:
    """
    Args:
      y (list[Tensor]): A pair of consecutive slices of mask
      unfold: nn.Unfold(kernel_size=(2, 2), padding=1)
      area (Tensor): surface area for 256 bit patterns (256, )

    Returns:
      Surface area of surface in 2x2x2 cube
    """
    # Two layers of segmentation masks -> (2, H, W)
    yy = torch.stack(y, dim=0).to(torch.float16)  # bit (0/1) but unfold requires float

    # unfold slides through the volume like a convolution
    # 2x2 kernel returns 8 values (2 channels * 2x2)
    cubes_float = unfold(yy)  # (8, n_cubes)

    # Each of the 8 values are either 0 or 1
    # Convert those 8 bits to one uint8
    # but indices are required to be int32 or long for area[cube_byte] below, not uint8
    cubes_byte = torch.zeros(cubes_float.size(1), dtype=torch.int32, device=device)

    for k in range(8):
        cubes_byte += cubes_float[k, :].to(torch.int32) << k

    # Area is defined for each of 256 bit patterns
    cubes_area = area[cubes_byte]

    return cubes_area


def compute_surface_dice_score(submit: pd.DataFrame, label: pd.DataFrame) -> float:
    """
    Compute surface Dice score for one 3D volume

    submit (pd.DataFrame): submission file with id and rle
    label (pd.DataFrame): ground truth id, rle, and also image height, width
    """
    # submit and label must contain exact same id in same order
    assert (submit['id'] == label['id']).all()
    assert len(label) > 0

    # All height, width must be the same
    len(label['height'].unique()) == 1
    len(label['width'].unique()) == 1

    # 256 patterns of area: Tensor (256, )
    area = create_table_neighbour_code_to_surface_area((1, 1, 1))
    area = torch.from_numpy(area).to(device)  # torch.float32

    # Slide through the volume like a convolution
    unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)

    r = label.iloc[0]
    h, w = r['height'], r['width']
    n_slices = len(label)

    # Padding before first slice
    y0 = y0_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

    num = 0     # numerator of surface Dice
    denom = 0   # denominator
    for i in range(n_slices + 1):
        if i < n_slices:
            r = label.iloc[i]
            y1 = rle_decode(r['rle'], (h, w))
            y1 = torch.from_numpy(y1).to(device)

            r = submit.iloc[i]
            y1_pred = rle_decode(r['rle'], (h, w))
            y1_pred = torch.from_numpy(y1_pred).to(device)
        else:
            # Padding after the last slice
            y1 = y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

        area_pred = compute_area([y0_pred, y1_pred], unfold, area)
        area_true = compute_area([y0, y1], unfold, area)

        idx = torch.logical_and(area_pred > 0, area_true > 0)

        num += area_pred[idx].sum() + area_true[idx].sum()
        denom += area_pred.sum() + area_true.sum()

        # Next slice
        y0 = y1
        y0_pred = y1_pred

    dice = num / denom.clamp(min=1e-8)
    return dice.item()


def add_size_columns(df: pd.DataFrame):
    """
    df (DataFrame): including id column kidney_1_dense_0000
    """
    widths = []
    heights = []
    subdirs = []
    nums = []
    for i, r in df.iterrows():
        file_id = r['id']
        subdir = file_id[:-5]    # kidney_1_dense
        file_num = file_id[-4:]  # 0000

        filename = '%s/train/%s/images/%s.tif' % (di, subdir, file_num)
        img = Image.open(filename)
        w, h = img.size
        widths.append(w)
        heights.append(h)
        subdirs.append(subdir)
        nums.append(file_num)

    df['width'] = widths
    df['height'] = heights
    df['image_id'] = subdirs
    df['slice_id'] = nums


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', default='submission1.csv')
    parser.add_argument('--label', default='kidney_1_dense')
    arg = parser.parse_args()

    # Submission file
    submit = pd.read_csv(arg.submit)

    # Label
    label = pd.read_csv(di + '/train_rles.csv')
    idx = label['id'].str.startswith(arg.label)
    label = label[idx]
    assert len(label) > 0

    # Add columns for height, width
    add_size_columns(label)

    tb = time.time()

    # Compute surface Dice score
    dice = compute_surface_dice_score(submit, label)
    print('Score: %.4f' % dice)

    dt = time.time() - tb
    print('%.1f sec' % dt)
