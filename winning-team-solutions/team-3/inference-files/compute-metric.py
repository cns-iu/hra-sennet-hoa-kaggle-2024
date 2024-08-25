import sys
import os
sys.path.append(f'{os.getcwd()}/sennet-metrics')
sys.path.append(f'{os.getcwd()}/sennet-metrics/src')

from sennet_metrices import *
import pandas as pd

SPLIT = 'test'
BASE_DIR = f'/teradata/hra_data/k4_data/competition-data/{SPLIT}'

# Compute competition metric.

submit_df = pd.read_csv('/u/yashjain/kaggle_4/winning-team-solutions/team-1/evaluation/EnsembleModel_ImageSize_3072/submission.csv')
label_df = pd.read_csv(f'{BASE_DIR}/solution.csv')

# Check the id column of the dataframe and separate rows into two dataframes based on if the values contains "kidney_5" or "kidney_6".
kidney_5_submit_df = submit_df[submit_df['id'].str.contains('kidney_5')]
kidney_6_submit_df = submit_df[submit_df['id'].str.contains('kidney_6')]
print(f'kidney_5_submit_df shape: {kidney_5_submit_df.shape}')
print(f'kidney_6_submit_df shape: {kidney_6_submit_df.shape}')

kidney_5_label_df = label_df[label_df['id'].str.contains('kidney_5')]
kidney_6_label_df = label_df[label_df['id'].str.contains('kidney_6')]
print(f'kidney_5_label_df shape: {kidney_5_label_df.shape}')
print(f'kidney_6_label_df shape: {kidney_6_label_df.shape}')

kidney_5_submit_df.reset_index(inplace=True)
kidney_6_submit_df.reset_index(inplace=True)
kidney_5_label_df.reset_index(inplace=True)
kidney_6_label_df.reset_index(inplace=True)

## -------------- Surface Dice --------------
surface_dice_kidney_5 = compute_surface_dice_score(kidney_5_submit_df, kidney_5_label_df)
print(f'Surface dice for public test (kidney_5) set is: {surface_dice_kidney_5}')

surface_dice_kidney_6 = compute_surface_dice_score(kidney_6_submit_df, kidney_6_label_df)
print(f'Surface dice for private test (kidney_6) set is: {surface_dice_kidney_6}')