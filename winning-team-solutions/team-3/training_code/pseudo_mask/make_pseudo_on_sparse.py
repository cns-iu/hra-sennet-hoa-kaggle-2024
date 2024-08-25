import os
import numpy as np
from glob import glob
import cv2

MERGE_MASKS = True
COUNT_VALUES = True
SAVE_MASK = False
SPLIT = 'train'
BASE_DIR = f'/home/v-honsong/dataset/senet/{SPLIT}'
PSEUDO_THR = 0.4
pseudo_mask_dir = '/home/v-honsong/workspace/sennet/pseudo_mask/k2'
pseudo_mask_path = f'{pseudo_mask_dir}/maxViT512_effv2s512_k1dk3dp.npy'


kidney = 'kidney_2'
density = 'sparse'
if kidney == 'kidney_2':
    tmp_label_ls = glob(os.path.join(f"/home/v-honsong/dataset/senet/train/kidney_2/labels", '*.tif'))
    tmp_label_ls.sort()
    tmp_label_ls = tmp_label_ls[900:]
elif kidney == 'kidney_3':
    path1=f"{BASE_DIR}/kidney_3_sparse"
    path2=f"{BASE_DIR}/kidney_3_dense"
    tmp_label_ls=glob(f"{path2}/labels/*")
    tmp_img_ls=[x.replace("labels","images").replace("dense","sparse") for x in tmp_label_ls]
    if density == 'dense':
        tmp_img_ls.sort()
        tmp_label_ls.sort()        
    else:
        all_img_ls = glob(f"{path1}/images/*")
        sparse_img_ls = list(set(all_img_ls) - set(tmp_img_ls))
        tmp_label_ls = [x.replace('images', 'labels') for x in sparse_img_ls]
        tmp_img_ls = sparse_img_ls
        tmp_img_ls.sort()
        tmp_label_ls.sort()

ori_labels = [cv2.imread(x,cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :] for x in tmp_label_ls]
ori_labels = np.concatenate(ori_labels, axis=0).astype(np.uint8)
pse_labels = np.load(pseudo_mask_path)
ori_labels = ori_labels>127
if MERGE_MASKS:
    for i in [0.1, 0.15, 0.2, 0.3]:
        pseudo_sparse_mask_dir = f'{pseudo_mask_dir}/mask{i}'
        os.makedirs(pseudo_sparse_mask_dir, exist_ok=True)
        pse_labels4save = (pse_labels>i).astype(np.uint8)
        pse_ori_label = pse_labels4save + ori_labels
        # pse_ori_label = pse_ori_label>0
        # pse_ori_label = pse_ori_label * 255
        if COUNT_VALUES:
            flat_arr = pse_ori_label.flatten()  
            unique_values, occurrence_counts = np.unique(flat_arr, return_counts=True)
            for value, count in zip(unique_values, occurrence_counts):  
                print(f"数值 {value} 出现了 {count} 次")  
        if SAVE_MASK:
            for label_path, label in zip(tmp_label_ls, pse_ori_label):
                label_name = os.path.basename(label_path).replace('.tif', '.png')
                pseudo_sparse_mask_path = f'{pseudo_sparse_mask_dir}/{label_name}'
                cv2.imwrite(pseudo_sparse_mask_path, label)
else:
    print(np.sum(ori_labels))
    for i in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        print(i, np.sum(pse_labels>i))

