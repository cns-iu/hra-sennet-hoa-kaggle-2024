model_ls = [ ## model_name, backbone, weight, image_size, stride, batch_size
    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-tf_efficientnetv2_s",
    #     "weight":'/home/v-honsong/workspace/sennet/result/UNet3c_effv2s_900onwards_Scale03/valonK3_best_dice.pt', 
    #     "image_size":512, 
    #     "stride":512, 
    #     "batch_size":256,
    #     "in_chans": 3
    # },
    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-tf_efficientnetv2_s",
    #     "weight":'/home/v-honsong/workspace/sennet/result/UNeteffv2s900onwards_512_moreBriConGammaaug/valonK3_best_dice.pt', 
    #     "largeRes_weight": "/home/v-honsong/workspace/sennet/result/0202UNeteffv2s832_900onwards_Scale55-105/valonK3_best_dice.pt",
    #     "large_size":832,
    #     "large_stride":832,
    #     "image_size":512, 
    #     "stride":512, 
    #     "batch_size":64,
    #     "in_chans": 1,
    #     "have_large_res":True
    # },
    {
        "model_name":"Unet", 
        "backbone":"tu-maxvit_large_tf_512",
        # "weight":'/home/v-honsong/workspace/sennet/result/UNetmaxvitlarge512_900onwards/valonK1_best_loss.pt', 
        # "weight": '/home/v-honsong/workspace/sennet/result/Denseonly_maxViTlarge/valonK2_last.pt',
        "weight": '/home/v-honsong/workspace/sennet/result/k1_k3pseu_maxViTlarge/valonK2_best_dice.pt',
        "image_size":512, 
        "stride":512, 
        "batch_size":32,
        "in_chans": 1,
        "have_large_res":False,
    },
    {
        "model_name":"Unet", 
        "backbone":"tu-tf_efficientnetv2_s",
        # "weight":'/home/v-honsong/workspace/sennet/result/Denseonly_Effv2s/valonK2_best_loss.pt', 
        "weight": '/home/v-honsong/workspace/sennet/result/k1_k3pseu_Effv2s/valonK2_best_loss.pt',
        "image_size":512, 
        "stride":512, 
        "batch_size":256,
        "in_chans": 1,
        "have_large_res":False,
    },
    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-tf_efficientnetv2_s",
    #     "weight":'/home/v-honsong/workspace/sennet/result/UNeteffv2s900onwards_512_moreBriConGammaaug/valonK3_best_dice.pt', 
    #     "image_size":448, 
    #     "stride":448, 
    #     "batch_size":256,
    #     "in_chans": 1,
    #     "have_large_res":False,
    # },
    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-tf_efficientnetv2_s",
    #     "weight":'/home/v-honsong/workspace/sennet/result/UNeteffv2s900onwards_512_moreBriConGammaaug/valonK3_best_dice.pt', 
    #     "image_size":512, 
    #     "stride":512, 
    #     "batch_size":256,
    #     "in_chans": 1,
    # },
    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-tf_efficientnetv2_s",
    #     "weight":'/home/v-honsong/workspace/sennet/result/UNeteffv2s900onwards_512_moreBriConGammaaug/valonK3_best_dice.pt', 
    #     "image_size":256, 
    #     "stride":256, 
    #     "batch_size":256,
    #     "in_chans": 1,
    # },
    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-maxvit_large_tf_384",
    #     "weight":'/home/v-honsong/workspace/sennet/result/UNetmaxvitlarge384_900onwards_resume_Scale05/valonK3_best_dice.pt', 
    #     "image_size":384, 
    #     "stride":384, 
    #     "batch_size":64,
    #     "in_chans": 1
    # },
    # {
    #     "model_name":"UnetPlusPlus", 
    #     "backbone":"tu-tf_efficientnetv2_l",
    #     "weight":'/home/v-honsong/workspace/sennet/result/UNetpp_effv2l_900onwards_Scale05/valonK3_best_dice.pt', 
    #     "image_size":512, 
    #     "stride":512, 
    #     "batch_size":64,
    #     "in_chans": 1,
    #     "have_large_res":False,
    # },
]


refine_model_ls = [
    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-tf_efficientnetv2_s",
    #     "weight":'/home/v-honsong/workspace/sennet/result/NonemptyMask_UNeteffv2s_900onwards_Scale55-105_valtrans/valonK3_best_dice.pt', 
    #     "image_size":512, 
    #     "stride":512, 
    #     "batch_size":256,
    #     "in_chans": 1
    # },

    # {
    #     "model_name":"Unet", 
    #     "backbone":"tu-tf_efficientnetv2_l",
    #     "weight":'/home/v-honsong/workspace/sennet/result/NonemptyMask_UNeteffv2l_900onwards_Scale55-105_valtrans/valonK3_best_dice.pt', 
    #     "image_size":512, 
    #     "stride":512, 
    #     "batch_size":64,
    #     "in_chans": 1
    # },
]


import torch as tc 
import torch.nn as nn  
import numpy as np
from tqdm import tqdm
import os,sys,cv2
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from glob import glob
import random
import torch
import pandas as pd
import argparse  
import sys
from PIL import Image
import gc
sys.path.append('metrics')
from sennet_metrices import *

def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)
CHOPPING_PER =1e-3

valid_aug_list = [
    ToTensorV2(),
]
valid_aug = A.Compose(valid_aug_list)

class CustomModel(nn.Module):
    def __init__(self, model_name, backbone, in_chans=1, target_size=1, weight=None):
        super().__init__()
        if model_name == 'Unet':
            self.model = smp.Unet(
                encoder_name=backbone, 
                encoder_weights=weight,
                in_channels=in_chans,
                classes=target_size,
                activation=None,
            )
        elif model_name == 'UnetPlusPlus':
            self.model = smp.UnetPlusPlus(
                encoder_name=backbone, 
                encoder_weights=weight,
                in_channels=in_chans,
                classes=target_size,
                activation=None,
            )            

    def forward(self, image):
        output = self.model(image)
        return output[:,0]


def build_model(model_name, backbone, in_chans=1):
    print('model_name', model_name)
    print('backbone', backbone)
    model = CustomModel(model_name, backbone, in_chans=in_chans)

    return model.cuda()

def normalize_img(in_img, eps=1e-9):
    min_ = in_img.min()
    max_ = in_img.max()
    return (255 * (in_img - min_) / (max_ - min_ + eps)).astype(np.uint8)


def add_noise(x:tc.Tensor,max_randn_rate=0.1,randn_rate=None,x_already_normed=False):
    """input.shape=(batch,f1,f2,...) output's var will be normalizate  """
    ndim=x.ndim-1
    if x_already_normed:
        x_std=tc.ones([x.shape[0]]+[1]*ndim,device=x.device,dtype=x.dtype)
        x_mean=tc.zeros([x.shape[0]]+[1]*ndim,device=x.device,dtype=x.dtype)
    else: 
        dim=list(range(1,x.ndim))
        x_std=x.std(dim=dim,keepdim=True)
        x_mean=x.mean(dim=dim,keepdim=True)
    if randn_rate is None:
        randn_rate=max_randn_rate*np.random.rand()*tc.rand(x_mean.shape,device=x.device,dtype=x.dtype)
    cache=(x_std**2+(x_std*randn_rate)**2)**0.5
    return (x-x_mean+tc.randn(size=x.shape,device=x.device,dtype=x.dtype)*randn_rate*x_std)/(cache+1e-7)

def filter_noise(x):
    TH=x.reshape(-1)
    index = -int(len(TH) * CHOPPING_PER)
    TH:int = np.partition(TH, index)[index]
    x[x>TH]=int(TH)
    ########################################################################
    TH=x.reshape(-1)
    index = -int(len(TH) * CHOPPING_PER)
    TH:int = np.partition(TH, -index)[-index]
    x[x<TH]=int(TH)
    return x

def get_indexes_along_axis0(num_slice, height, width, image_size, stride, slice_offset, nonempty_slices):
    indexes = []
    cur_h, cur_w = 0, 0
    flag = 0
    while(cur_h < height or cur_w < width):
        if (cur_h + image_size <= height) and (cur_w + image_size <= width):
            indexes.append((cur_h, cur_w))
            cur_w += stride
        elif (cur_h + image_size <= height) and (cur_w + image_size > width):
            indexes.append((cur_h, width - image_size)) 
            cur_h += stride
            cur_w = 0

        if (cur_h + image_size > height):
            if flag == 0:
                cur_h = height - image_size 
                flag = 1
            else:
                break

    indexes_3d = []
    if nonempty_slices is None: slice_ls = range(slice_offset, num_slice-slice_offset)
    else: slice_ls = nonempty_slices
    for slice_idx in slice_ls:
        for h, w in indexes:
            indexes_3d.append((slice_idx, h, w))
    return indexes_3d

def norm_with_clip(x:torch.Tensor,smooth=1e-5):
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    x=(x-mean)/(std+smooth)
    x[x>5]=(x[x>5]-5)*1e-3 +5
    x[x<-3]=(x[x<-3]+3)*1e-3-3
    return x

def min_max_normalization(x:tc.Tensor)->tc.Tensor:
    """input.shape=(batch,f1,...)"""
    shape=x.shape
    if x.ndim>2:
        x=x.reshape(x.shape[0],-1)
    
    min_=x.min(dim=-1,keepdim=True)[0]
    max_=x.max(dim=-1,keepdim=True)[0]
    if min_.mean()==0 and max_.mean()==1:
        return x.reshape(shape)
    
    x=(x-min_)/(max_-min_+1e-9)
    return x.reshape(shape)

def pad_hw(images, pad_h, pad_w):
    pad_width = [(0, 0),  
                 (pad_h, pad_h),  
                 (pad_w, pad_w)]  
    images = np.pad(images, pad_width=pad_width, mode='constant')
    return images

class Dataset3D(Dataset):
    def __init__(self, image_list, label_list=None, trans_axis=0,image_size=512, in_chans=1,stride=512,aug=False, pad_h=0, pad_w=0, nonempty_slices=None):
        super(Dataset,self).__init__()

        self.image_size=image_size
        self.in_chans=in_chans
        assert self.in_chans % 2 == 1
        self.slice_offset = (self.in_chans - 1)//2
        images = [cv2.imread(x,cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :] for x in image_list]
        images = np.concatenate(images, axis=0)## N, H, W

        self.label_list = label_list
        images = torch.tensor(filter_noise(images))
        images = (min_max_normalization(images.to(tc.float16)[None])[0]*255).to(tc.uint8).numpy()
        if label_list is not None:
            labels = [cv2.imread(x,cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :] for x in label_list]
            labels = np.concatenate(labels, axis=0).astype(np.uint8)
        else:
            labels = None      

        if trans_axis == 1:
            images = np.transpose(images, (1,2,0))
            if label_list is not None:
                labels = np.transpose(labels, (1,2,0))
        if trans_axis == 2:
            images = np.transpose(images, (2,0,1))
            if label_list is not None:
                labels = np.transpose(labels, (2,0,1))
        self.trans_axis = trans_axis       
        images = pad_hw(images, pad_h, pad_w)
        
        self.images = [images]
        self.labels = [labels]
        num_slice, height, width = images.shape
        slide_image_size = min(image_size, height-1, width-1)
        slide_stride = min(stride, height-1, width-1)
        self.coors = get_indexes_along_axis0(num_slice, height, width, slide_image_size, slide_stride, self.slice_offset, nonempty_slices)
        self.idx_3d = [0] * len(self.coors)
        self.transform=valid_aug

    def __len__(self):
        return len(self.coors)

    def __getitem__(self,index):
        n, h, w = self.coors[index]
        idx_3d = self.idx_3d[index]
        if self.slice_offset == 0:
            image = self.images[idx_3d][n, h:h+self.image_size, w:w+self.image_size] # (1, H, W)
        else:
            image = self.images[idx_3d][n-self.slice_offset:n+self.slice_offset+1, h:h+self.image_size, w:w+self.image_size] # (3(5), H, W)
            image = np.transpose(image, (1, 2, 0))            
        data = self.transform(image=image)
        return data['image'], idx_3d, n, h, w
    
    def get_labels(self):
        return self.labels
    

def permute_axis(pred, trans_axis=0, direction=0):
    """
    direction: 
    0: from original shape to transposed shape
    1: from transposed shape to original shape
    """
    if trans_axis == 0: 
        return pred  ## N, H, W
    elif trans_axis == 1:
        if direction == 0:
            pred = pred.permute(1, 2, 0) ## H, W, N
        else:
            pred = pred.permute(2, 0, 1)  ## N, H, W
    elif trans_axis == 2:
        if direction == 0:
            pred = pred.permute(2, 0, 1) ## W, N, H
        else:
            pred = pred.permute(1, 2, 0) ## N, H, W
    return pred  ## N, H, W

def get_nhw(trans_axis, NUM_SLICES, HEIGHT, WIDTH):
    if trans_axis == 0:
        num_slices, height, width = NUM_SLICES, HEIGHT, WIDTH
    elif trans_axis == 1:
        num_slices, height, width = HEIGHT, WIDTH, NUM_SLICES
    elif trans_axis == 2:
        num_slices, height, width = WIDTH, NUM_SLICES, HEIGHT
    return num_slices, height, width

def get_pad_hw(height, width, image_size, edge_size=16):
    if height <= image_size:
        pad_h = (image_size - height) // 2 + edge_size
    else:
        pad_h = 0
    if width <= image_size:
        pad_w = (image_size - width) // 2 + edge_size
    else:
        pad_w = 0
    return pad_h, pad_w

def perd_add2map(preds, preds_cnt, n, h, w, height, width, pad_h, pad_w, image_size, pred, sample_idx):
    ### w=0, 4 是pad后在整个slice上的坐标
    ### 
    w_start = max(w-pad_w, 0)
    w_end = min(width, w_start+image_size)
    h_start = max(h-pad_h, 0)
    h_end = min(height, h_start+image_size)
    
    
    if pad_h != 0:
        h_pred_start = pad_h - h
        h_pred_end = h_pred_start + h_end - h_start
    else:
        h_pred_start, h_pred_end = 0, image_size
    if pad_w != 0:
        w_pred_start = pad_w - w
        w_pred_end = w_pred_start + w_end - w_start
    else:
        w_pred_start, w_pred_end = 0, image_size

    preds[n, h_start:h_end, w_start:w_end] += pred[sample_idx][h_pred_start:h_pred_end, w_pred_start:w_pred_end]
    preds_cnt[n, h_start:h_end, w_start:w_end] += 1
    
    return preds, preds_cnt


def search_thr(preds, labels, image_ids, width, height, min_thr=0.01, max_thr=0.5, interval=0.01, early_stop_max=5):
    thr_list = np.arange(min_thr, max_thr, interval)
    thr_list = [round(x, 3) for x in thr_list]
    best_dice, best_thr, early_stop_cnt = 0, 0, 0
    for thr in tqdm(thr_list, total=len(thr_list)):
        if early_stop_cnt >= early_stop_max: break
        ## -------------- 1. Use the thre to get binary pred ---------------
        bin_preds = preds > thr
        ## -------------- 2. RLE Encode label and preds for each slice---------------
        tmp_preds, tmp_labels = [], []
        for pred, label in zip(bin_preds, labels):
            tmp_preds.append(rle_encode(pred))
            tmp_labels.append(rle_encode(label))
        ## -------------- 3. Get df---------------
        submit = pd.DataFrame({'id': image_ids, 'rle': tmp_preds, 'width':[width]*len(image_ids), 'height': [height]*len(image_ids)})  
        label_df = pd.DataFrame({'id': image_ids, 'rle': tmp_labels, 'width':[width]*len(image_ids), 'height': [height]*len(image_ids)})
        ## -------------- 4. Surface Dice --------------
        surface_dice = compute_surface_dice_score(submit, label_df)
        print(f'Surface dice at threshold {thr} is: {surface_dice}')
        if surface_dice > best_dice:
            best_dice, best_thr = surface_dice, thr
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

    print(f'Best Surface dice at threshold {best_thr} is: {best_dice}')
    
def filter_empty_slice(preds, threshold=50):
    num_slice, h, w = preds.shape
    for i in range(num_slice):
        if np.sum(preds[i, :, :]) < threshold:
            preds[i, :, :] = np.zeros((h, w))
    return preds

from glob import glob
SPLIT = 'train'
BASE_DIR = f'/home/v-honsong/dataset/senet/{SPLIT}'
if SPLIT == 'test':
    kidney_ls = os.listdir(BASE_DIR)
    image_ls, label_ls = [], []
    for kidney in kidney_ls:
        tmp_img_ls = glob(os.path.join(f'{BASE_DIR}/{kidney}/images', '*.tif'))
        tmp_img_ls.sort()
        image_ls.append(tmp_img_ls)
        label_ls.append(None)
else:

    kidney = 'kidney_2'
    density = 'sparse'
    if kidney == 'kidney_2':
        tmp_img_ls = glob(os.path.join(f"/home/v-honsong/dataset/senet/train/kidney_2/images", '*.tif'))
        tmp_label_ls = glob(os.path.join(f"/home/v-honsong/dataset/senet/train/kidney_2/labels", '*.tif'))
        tmp_img_ls.sort()
        tmp_label_ls.sort()
        tmp_img_ls = tmp_img_ls[900:]
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
        
    image_ls = [tmp_img_ls]    
    label_ls = [tmp_label_ls]

SAVE_MASK = False
MASK_SAVE_DIR = "pseudo_mask/k2"
os.makedirs(MASK_SAVE_DIR, exist_ok=True)
MASK_SAVE_PATH = f'{MASK_SAVE_DIR}/maxViT512_effv2s512_k1dk3dp.npy'
REFINE = False
TTA = False
TTA_LS = [[2,3]]
TRANS_AXIS = [0,1,2]  #0,1,2
if (len(image_ls[0]) == 3): TRANS_AXIS = [0]
Stage1_Thres = 0.2
Threshold = 0.3  # 0.15
Discard_thres = 30
EMPTY_THRESS = [50, 100, 200]
EARLY_MAX = 20

sample_submission = {'id':[], 'rle':[]}

def infer_kidney(NUM_SLICES, HEIGHT, WIDTH,  tmp_image_ls, tmp_label_ls, preds, preds_cnt, nonempty_dict, model_ls=model_ls, trans_axiss=TRANS_AXIS):
    for model_idx, model_card in enumerate(model_ls):
        
        for trans_axis in trans_axiss:
                    
            num_slices, height, width = get_nhw(trans_axis, NUM_SLICES, HEIGHT, WIDTH)   
            if  model_card['have_large_res'] and height > model_card["large_size"] and width > model_card["large_size"]:
                image_size = model_card["large_size"]
                stride = model_card["large_stride"]
                weight_path = model_card['largeRes_weight']
            else:
                image_size = model_card["image_size"]
                stride = model_card["stride"]
                weight_path = model_card["weight"]
            pad_h, pad_w = get_pad_hw(height, width, image_size, edge_size=2)
            
            val_dataset = Dataset3D(tmp_image_ls, label_list=tmp_label_ls, trans_axis=trans_axis, image_size=image_size, 
                                    stride=stride, aug=False, pad_h=pad_h, pad_w=pad_w, 
                                    in_chans=model_card["in_chans"], nonempty_slices=nonempty_dict[trans_axis])
            
            if SPLIT == 'train':
                if trans_axis==0 and model_idx==len(model_ls)-1:
                    labels = val_dataset.get_labels()[0]
            else:
                labels = None
                
            val_dataset = DataLoader(val_dataset, batch_size=model_card["batch_size"] ,num_workers=4, shuffle=False, drop_last=False)

            model=build_model(model_card["model_name"], model_card["backbone"], in_chans=model_card["in_chans"])
            model.load_state_dict(tc.load(weight_path,"cuda:0"), strict=True)
            model.eval()

            preds = permute_axis(preds, trans_axis, 0)
            preds_cnt = permute_axis(preds_cnt, trans_axis, 0)
            for batch_idx, (x,idx_3d, n_bs, h_bs, w_bs) in enumerate(tqdm(val_dataset, total=len(val_dataset))):

                x=x.cuda().to(tc.float32)
                x=norm_with_clip(x.reshape(-1,*x.shape[2:])).reshape(x.shape)

                with autocast():
                    with tc.no_grad():
                        pred=torch.sigmoid(model(x))
                        if TTA:
                            for axis in TTA_LS:  # [2],[3],
                                tmp_pred = torch.sigmoid(model(torch.flip(x, dims=axis)))
                                axis = [tmp_x-1 for tmp_x in axis]
                                tmp_pred = torch.flip(tmp_pred, dims=axis)
                                pred += tmp_pred
                            pred = pred / (len(TTA_LS)+1)
                            
                
                for sample_idx, (n, h, w) in enumerate(zip(n_bs, h_bs, w_bs)):
                    if pad_h == 0 and pad_w == 0:
                        preds[n, h:h+image_size, w:w+image_size] += pred[sample_idx]
                        preds_cnt[n, h:h+image_size, w:w+image_size] += 1
                    else:
                        perd_add2map(preds, preds_cnt, n, h, w, height, width, pad_h, pad_w, image_size, pred, sample_idx)
            preds = permute_axis(preds, trans_axis, 1)
            preds_cnt = permute_axis(preds_cnt, trans_axis, 1)
            del model, val_dataset
            gc.collect()
            torch.cuda.empty_cache()
            
    return preds, preds_cnt, labels

def get_empty_slice_dict_with_seg(preds, preds_cnt, pos_thres=0.2, empty_thres=50):
    tmp_preds = torch.div(preds.cpu(), preds_cnt.cpu()).numpy()
    tmp_preds = (tmp_preds > pos_thres).astype(np.int8)
    nonempty_slices = {0:[], 1:[], 2:[]}
    n, h, w = tmp_preds.shape
    for i in range(n):
        if np.sum(tmp_preds[i, :, :]) > empty_thres:
            nonempty_slices[0].append(i)

    for i in range(h):
        if np.sum(tmp_preds[:, i, :]) > empty_thres:
            nonempty_slices[1].append(i)

    for i in range(w):
        if np.sum(tmp_preds[:, :, i]) > empty_thres:
            nonempty_slices[2].append(i)
    del tmp_preds
    gc.collect()
    return nonempty_slices

for tmp_image_ls, tmp_label_ls in zip(image_ls, label_ls):
    image_ids = [image_path.split('/')[-3] + '_' +image_path.split('/')[-1].split('.')[0] for image_path in tmp_image_ls]
    sample_submission['id'].extend(image_ids)
    tmp_img = cv2.imread(tmp_image_ls[0],cv2.IMREAD_GRAYSCALE)
    HEIGHT, WIDTH = tmp_img.shape
    NUM_SLICES = len(image_ids)
    preds, preds_cnt = torch.zeros((NUM_SLICES, HEIGHT, WIDTH), dtype=torch.float16).cuda(), torch.zeros((NUM_SLICES, HEIGHT, WIDTH), dtype=torch.int8).cuda()
    del tmp_img
    gc.collect()
    preds, preds_cnt, labels = infer_kidney(NUM_SLICES, HEIGHT, WIDTH,  tmp_image_ls, tmp_label_ls, preds, preds_cnt, 
                                            nonempty_dict=[None, None, None], model_ls=model_ls, trans_axiss=TRANS_AXIS)
    if SAVE_MASK:
        preds4save = torch.div(preds.cpu(), preds_cnt.cpu()).numpy()
        np.save(MASK_SAVE_PATH, preds4save)
        del preds4save
        gc.collect()
    if REFINE:
        for EMPTY_THRES in EMPTY_THRESS:
            print(f'*************** Searching Empty threshold {EMPTY_THRES} *******************')
            nonempty_dict = get_empty_slice_dict_with_seg(preds, preds_cnt, pos_thres=Stage1_Thres, empty_thres=EMPTY_THRES)
            preds2, preds_cnt2, _ = infer_kidney(NUM_SLICES, HEIGHT, WIDTH,  tmp_image_ls, tmp_label_ls, preds, preds_cnt, nonempty_dict=nonempty_dict, model_ls=refine_model_ls, trans_axiss=TRANS_AXIS)
            preds2 = torch.div(preds2.cpu(), preds_cnt2.cpu()).numpy()
            search_thr(preds2, labels, image_ids, WIDTH, HEIGHT, min_thr=0.1, max_thr=0.6, interval=0.01)
            print('*******************************************')
    else:
        preds = torch.div(preds.cpu(), preds_cnt.cpu()).numpy()
        search_thr(preds, labels, image_ids, WIDTH, HEIGHT, min_thr=0.3, max_thr=0.9, interval=0.01, early_stop_max=EARLY_MAX)