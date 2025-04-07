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

import gc, time, json, argparse

parser = argparse.ArgumentParser(description="This is my program description")

# Add the arguments
parser.add_argument('--team_id', type=str, help='team id to proceed', default="1")
# Execute the parse_args() method
args = parser.parse_args()

data_dir = '../k4_data'
comp_dir = os.path.join(data_dir, 'competition-data')
sol_dir = os.path.join(data_dir, 'winning-team-solutions')
sub_dir = os.path.join(data_dir, 'k4-top-50-submissions')
result_dir = os.path.join('./performance_top50')
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok = True)

df_sol = pd.read_csv(os.path.join(comp_dir, 'solution.csv'))

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
team_id = args.team_id
team_performance = {} if not os.path.exists(os.path.join(result_dir, f'team_{team_id}.json')) else json.load(open(os.path.join(result_dir, f'team_{team_id}.json')))
filename = f'team_{team_id}.csv'
filepath = os.path.join(sub_dir, filename)
df_team = pd.read_csv(filepath)
# merge with GT dataframe (make sure the alignment)
df_merge = df_team.merge(df_sol, on = 'id', suffixes = ('_pred', '_sol'))
df_merge['slice'] = df_merge['slice'].astype(int)
df_merge.loc[df_merge['rle_pred'].apply(lambda x: len(str(x)) == 3), 'rle_pred'] = '1 0'
# run evaluation for each image_id
# list_metrics = ['nsd_0', 'nsd_1', 'cldice', 'assd', 'dice', 'nsd_mcube', 'nsd_monai_0', 'nsd_monai_0_subvoxel', 'nsd_monai_1_subvoxel']
list_metrics = ['nsd_0', 'nsd_1', 'cldice', 'assd', 'dice', 'nsd_mcube']
for image_id in df_merge['image'].unique():
    if (image_id in team_performance) and all([metric_name in team_performance[image_id] for metric_name in list_metrics]):
        print(f"image_id {image_id} already calculated. Skipping this file.")
        continue
    team_performance[image_id] = {}
    # if str(team_id) == '2' and image_id == 'kidney_5':
    #     continue
    df_image = df_merge.loc[df_merge['image'] == image_id].sort_values('slice')
    # separate dataframes for target image_id
    df_pred = df_image.loc[:, ['id', 'rle_pred']].rename(columns = {'rle_pred': 'rle'})
    df_real = df_image.loc[:, ['id', 'rle_sol', 'width', 'height', 'image', 'slice', 'Usage']].rename(columns = {'rle_sol': 'rle'})
    w = df_real['width'].iloc[0]
    h = df_real['height'].iloc[0]
    shape = (df_pred.shape[0], h,w)
    pred = np.zeros(shape).astype(bool)
    true = np.zeros(shape).astype(bool)
    for slice_idx, (_, row) in enumerate(df_image.iterrows()):
        # slice_idx = row['slice'] # this cannot be used for slice_idx
        p_rle = rle_decode(row['rle_pred'], shape[1:])
        t_rle = rle_decode(row['rle_sol'], shape[1:])
        pred[slice_idx] = p_rle.astype(bool)
        true[slice_idx] = t_rle.astype(bool)
    print('Annotation summation (pred vs. true):', pred.sum(), true.sum())
    start = time.time()
    print(f'Running for team {team_id}, image_id {image_id}...')
    df_image = df_merge.loc[df_merge['image'] == image_id].sort_values('slice')
    # separate dataframes for target image_id
    df_pred = df_image.loc[:, ['id', 'rle_pred']].rename(columns = {'rle_pred': 'rle'})
    df_real = df_image.loc[:, ['id', 'rle_sol', 'width', 'height', 'image', 'slice', 'Usage']].rename(columns = {'rle_sol': 'rle'})
    #################
    # calculate scores
    #################
    ### 1.surface dice score from deepmind
    print('calculating nsd_0')
    if 'nsd_0' not in team_performance[image_id]:
        try:
            nsd_0 = score(df_real, df_pred, row_id_column_name = 'id', rle_column_name = 'rle', image_id_column_name='image', slice_id_column_name = 'slice', tolerance = 0)
            team_performance[image_id]['nsd_0'] = nsd_0
            json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
        except:
            team_performance[image_id]['nsd_0'] = 0
    print('calculating nsd_1')
    if 'nsd_1' not in team_performance[image_id]:
        try:
            nsd_1 = score(df_real, df_pred, row_id_column_name = 'id', rle_column_name = 'rle', image_id_column_name='image', slice_id_column_name = 'slice', tolerance = 1)
            team_performance[image_id]['nsd_1'] = nsd_1
            json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
        except:
            team_performance[image_id]['nsd_1'] = 0
    gc.collect()
    ### 2. cldice
    # 1. make 3d iamge
    w = df_real['width'].iloc[0]
    h = df_real['height'].iloc[0]
    shape = (df_pred.shape[0], h,w)
    pred = np.zeros(shape).astype(bool)
    true = np.zeros(shape).astype(bool)
    for slice_idx, (_, row) in enumerate(df_image.iterrows()):
        # slice_idx = row['slice'] # this cannot be used for slice_idx
        p_rle = rle_decode(row['rle_pred'], shape[1:])
        t_rle = rle_decode(row['rle_sol'], shape[1:])
        pred[slice_idx] = p_rle.astype(bool)
        true[slice_idx] = t_rle.astype(bool)
    gc.collect()
    # 2. calculate cldice
    print('calculating cldice')
    if 'cldice' not in team_performance[image_id]:
        try:
            cldice = clDice(pred, true)
            team_performance[image_id]['cldice'] = cldice
            json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
        except:
            pass
    gc.collect()
    ### 3. monai's assd
    true = torch.from_numpy(true).unsqueeze(0).unsqueeze(0)
    pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    if 'assd' not in team_performance[image_id]:
        try:
            assd = compute_average_surface_distance(pred, true, spacing = 1, symmetric = True)
            team_performance[image_id]['assd'] = assd.item()
            json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
        except:
            pass
        
    gc.collect()
    ### 4. monai's dice
    print('calculating dice')
    if 'dice' not in team_performance[image_id]:
        try:
            dice = compute_dice(pred, true)
            team_performance[image_id]['dice'] = dice.item()
            json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
        except:
            pass
    gc.collect()
    ### (optional) surface dice - marching cube
    print('calculating nsd_mcube')
    if 'nsd_mcube' not in team_performance[image_id]:
        try:
            nsd_mcube = compute_surface_dice_score(df_pred, df_real, device = 'cuda')
            team_performance[image_id]['nsd_mcube'] = nsd_mcube
            json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
        except:
            pass
    gc.collect()
    # ### surface dice - monai
    # print('calculating nsd_monai_0')
    # if 'nsd_monai_0' not in team_performance[image_id]:
    #     try:
    #         nsd_monai_0 = compute_surface_dice(pred, true, [0], True, use_subvoxels = False)
    #         team_performance[image_id]['nsd_monai_0'] = nsd_monai_0.item()
    #         json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
    #     except:
    #         pass
    # print('calculating nsd_monai_0_subvoxel')
    # if 'nsd_monai_0_subvoxel' not in team_performance[image_id]:
    #     try:
    #         nsd_monai_0_subvoxel = compute_surface_dice(pred, true, [0], True, use_subvoxels = True)
    #         team_performance[image_id]['nsd_monai_0_subvoxel'] = nsd_monai_0_subvoxel.item()
    #         json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
    #     except:
    #         pass
    # print('calculating nsd_monai_1_subvoxel')
    # if 'nsd_monai_1_subvoxel' not in team_performance[image_id]:
    #     try:
    #         nsd_monai_0_subvoxel = compute_surface_dice(pred, true, [1], True, use_subvoxels = True)
    #         team_performance[image_id]['nsd_monai_1_subvoxel'] = nsd_monai_0_subvoxel.item()
    #         json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
    #     except:
    #         pass
    gc.collect()
    # collect time
    end = time.time()
    team_performance[image_id]['time_took'] = end - start
    # save performance
    json.dump(team_performance, open(os.path.join(result_dir, f'team_{team_id}.json'), 'w'))
    print(f'took {end-start:.2f} seconds...')
    gc.collect()