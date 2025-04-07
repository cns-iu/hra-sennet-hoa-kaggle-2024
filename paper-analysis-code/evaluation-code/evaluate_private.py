import os
import numpy as np
import pandas as pd
import shutil
# import cldice
import nibabel as nib
import torch
import monai
from monai.metrics import compute_average_surface_distance, compute_surface_dice, compute_dice
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d, skeletonize

import gc, time, json, glob
import nibabel as nib

data_dir = '/teradata/hra_data/k4_data/nnunet-test-set-preds'
list_files_pred = glob.glob(os.path.join(data_dir, 'predsTs', '*.nii.gz'))
list_files_gt = [f.replace('predsTs', 'labelsTs') for f in list_files_pred]

import sys
sys.path.append('../')

from codes.metrics.sennet_metrices import (
    rle_decode,
    rle_encode,
    create_table_neighbour_code_to_surface_area,
    compute_area,
    compute_surface_dice_score
)
from codes.metrics.src.official_metric import score

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)


device = 'cuda'
if os.path.exists('performance_private.json'):
    performance = json.load(open('performance_private.json', 'r'))
else:
    performance = {}

for f_pred, f_gt in zip(list_files_pred, list_files_gt):
    start = time.time()
    case_id = os.path.basename(f_pred).split('.nii.gz')[0]
    performance.setdefault(case_id, {})
    # load data
    pred = nib.load(f_pred).get_fdata().astype(bool)
    gt = nib.load(f_gt).get_fdata().astype(bool)
    # calculate scores
    ### 2. cldice
    try:
        cldice = clDice(pred, gt)
        performance[case_id]['cldice'] = cldice
    except:
        performance[case_id]['cldice'] = 0
    print(performance[case_id])
    gc.collect()
    ### 3. monai's assd
    true = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
    pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    assd = compute_average_surface_distance(pred, true, spacing = 1, symmetric = True)
    performance[case_id]['assd'] = assd.item()
    gc.collect()
    ### 4. monai's dice
    try:
        dice = compute_dice(pred, true)
        performance[case_id]['dice'] = dice.item()
    except:
        performance[case_id]['dice'] = 0
    print(performance[case_id])
    gc.collect()
    ### surface dice - monai
    try:
        nsd_monai_0 = compute_surface_dice(pred, true, [0], True, use_subvoxels = False)
        performance[case_id]['nsd_monai_0'] = nsd_monai_0.item()
    except:
        performance[case_id]['nsd_monai_0'] = 0
    try:
        nsd_monai_0_subvoxel = compute_surface_dice(pred, true, [0], True, use_subvoxels = True)
        performance[case_id]['nsd_monai_0_subvoxel'] = nsd_monai_0_subvoxel.item()
    except:
        performance[case_id]['nsd_monai_0_subvoxel'] = 0
    try:
        nsd_monai_0_subvoxel = compute_surface_dice(pred, true, [1], True, use_subvoxels = True)
        performance[case_id]['nsd_monai_1_subvoxel'] = nsd_monai_0_subvoxel.item()
    except:
        performance[case_id]['nsd_monai_1_subvoxel'] = 0
    print(performance[case_id])
    gc.collect()
    # collect time
    end = time.time()
    performance[case_id]['time_took'] = end - start
    # save performance
    json.dump(performance, open('performance_private.json', 'w'))
    print(f'took {end-start:.2f} seconds...')
    gc.collect()
