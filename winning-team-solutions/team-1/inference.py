
import os
import sys
import cv2
import cc3d
import timm
import shutil
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import albumentations as A
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Download from https://www.kaggle.com/datasets/clevert/segmentation-models-pytorch-extra-stem-2-5d 
sys.path.append("/u/yashjain/kaggle_4/winning-team-solutions/team-1/segmentation-models-pytorch-extra-stem-2-5d")
import segmentation_models_pytorch as smp

############################################### helper functions ##################################################
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    if rle=='':
        rle = '1 0'
    return rle

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def build_model(backbone, in_channels, num_classes):
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights=None,
        encoder_args={"in_channels": in_channels},
        decoder_norm_type="GN",
        decoder_act_type="GeLU",
        decoder_upsample_method="nearest",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    return model


def filter_checkpoint(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    return new_state_dict


def load_model(backbone, in_channels, num_classes, path):
    model = build_model(backbone, in_channels, num_classes)
    state_dict = torch.load(path, map_location="cpu")
    state_dict = filter_checkpoint(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class Ensemble(object):
    def __init__(self, backbone, in_channels, num_classes, ckpts, device):
        self.models = []
        for ckpt_path in ckpts:
            model = load_model(backbone, in_channels, num_classes, ckpt_path).to(device)
            model = torch.compile(model)
            self.models.append(model)
            
    def __call__(self, x):
        out = None
        for model in self.models:
            if out is None:
                out = model(x).sigmoid()
            else:
                out += model(x).sigmoid()
        out /= len(self.models)
        return out


class InferenceDataset(torch.utils.data.Dataset):
    
    axis2dim = {"z": 0, "y": 1, "x": 2}
    
    def __init__(self, volume_path, volume_shape, local_rank, world_size, in_channels=3, image_size=512, axis="z"):
        self.volume_path = volume_path
        self.volume_shape = volume_shape
        self.axis = axis
        self.in_channels = in_channels
        self.image_size = image_size
        block = self.volume_shape[self.axis2dim[self.axis]] // world_size
        if local_rank < world_size-1:
            self.indexs = range(self.volume_shape[self.axis2dim[self.axis]])[local_rank*block:(local_rank+1)*block]
        else:
            self.indexs = range(self.volume_shape[self.axis2dim[self.axis]])[local_rank*block:]
        
    def __len__(self):
        return len(self.indexs)
    
    def load_image(self, idx):
        idx = self.indexs[idx]
        volume = np.memmap(self.volume_path, shape=self.volume_shape, dtype=np.uint16, mode="r")
        idxs = np.clip(range(idx-self.in_channels//2, idx+self.in_channels//2+1), 0, self.volume_shape[self.axis2dim[self.axis]]-1)
        if self.axis == "z":
            image =  volume[idxs].transpose(1, 2, 0)
        elif self.axis == "x":
            image =  volume[:, :, idxs]
        else:
            image =  volume[:, idxs, :].transpose(0, 2, 1)
        image = image.astype(np.float32)
        image = image / 65535.0
        return image
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        orig_size = image.shape
        area = self.image_size**2
        orig_area = orig_size[0]*orig_size[1]
        scale = np.sqrt(area/orig_area)
        new_h = int(orig_size[0]*scale) if int(orig_size[0]*scale) % 32 == 0 else int(orig_size[0]*scale) - (int(orig_size[0]*scale)%32) + 32
        new_w = int(orig_size[1]*scale) if int(orig_size[1]*scale) % 32 == 0 else int(orig_size[1]*scale) - (int(orig_size[1]*scale)%32) + 32
        # LANCZOS4 is slighter better than bilinear and bicubic
        image = cv2.resize(image, (new_w, new_h), cv2.INTER_LANCZOS4)
        image = torch.tensor(np.transpose(image, (2, 0, 1)))
        return self.indexs[idx], image, torch.tensor(np.array([orig_size[0], orig_size[1]]))
    

############################################### main ##################################################

def main_worker(rank, args, queue):
    
    torch.backends.cudnn_benchmark = True
    
    # set device
    device = torch.device(f"cuda:{rank}")
    
    # meta info
    volume_shape = args.volume_shape
    volume_path = args.volume_path
        
    # build model
    model = Ensemble(args.backbone, args.in_channels, args.num_classes, args.ckpt_path, device)
    
    # inference 
    with torch.no_grad():
        for size in args.image_size:
            for axis in args.axis:
                test_dataset = InferenceDataset(volume_path, volume_shape, rank, args.num_processes, args.in_channels, image_size=size, axis=axis)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
                max_len = test_dataset.volume_shape[test_dataset.axis2dim[axis]]
                pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Inference {args.group} {axis}', ncols=150)
                for step, (idx, images, shapes) in pbar:
                    shape = shapes[0].numpy()
                    idx = idx.numpy()
                    images = images.to(device, non_blocking=True)
                    bsz = images.size(0)
                    batch_pred_mask = torch.zeros(bsz, args.num_classes, shape[0], shape[1]).to(device)
                    for aug, flip in zip([torch.flip]*len(args.flip)+[partial(torch.rot90, dims=[2, 3])]*len(args.rot), args.flip+args.rot):
                        with torch.cuda.amp.autocast(enabled=True):
                            preds = model(aug(images, flip))
                            flip = -flip if not isinstance(flip, list) else flip
                            preds = F.interpolate(aug(preds, flip).float(), (int(shape[0]), int(shape[1])), mode='bicubic')
                        batch_pred_mask += preds
                    
                    batch_pred_mask /= len(args.axis) * (len(args.flip)+len(args.rot)) * len(args.image_size)
                    if args.overlap:
                        batch_pred_mask /= args.num_classes
                    masks = batch_pred_mask.to(torch.float16).cpu().numpy()
                    queue.put((axis, idx, masks, max_len))
                    pbar.set_postfix(shape=images.shape)
                        
                        
def write_worker(args, queue, write_lock):
    while True:
        axis, idx, masks, max_len = queue.get()
        if idx is None:
            break
        with write_lock:
            pred_masks = np.memmap(args.mask_path, shape=args.volume_shape, dtype=np.float16, mode="r+")
            if args.overlap:
                for i in range(args.num_classes):
                    mask = masks[:, i, ...]
                    offset = i - args.num_classes // 2
                    idxs = np.clip(idx+offset, 0, max_len-1)
                    if axis == "z":
                        pred_masks[idxs, :, :] += mask
                    elif axis == "y":
                        pred_masks[:, idxs, :] += mask.transpose(1, 0, 2)
                    else:
                        pred_masks[:, :, idxs] += mask.transpose(1, 2, 0)
            else:
                mask = masks[:, args.num_classes//2, ...]
                idxs = np.clip(idx, 0, max_len-1)
                if axis == "z":
                    pred_masks[idxs, :, :] += mask
                elif axis == "y":
                    pred_masks[:, idxs, :] += mask.transpose(1, 0, 2)
                else:
                    pred_masks[:, :, idxs] += mask.transpose(1, 2, 0)
            pred_masks.flush()
            del pred_masks, masks, axis, idx, max_len
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", type=str, default="kidney_5")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=2560)
    parser.add_argument("--axis", type=str, default="z|y|x")
    parser.add_argument("--flip", type=int, default=3)
    parser.add_argument("--rot", type=int, default=3)
    parser.add_argument("--overlap", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    args.image_size = [args.image_size]
    args.axis = args.axis.split("|")
    args.flip = [[], [1], [2], [3], [2,3]][:args.flip]
    args.rot = [1, 2, 3][:args.rot]
    args.ckpt_path = args.ckpt_path.split("|")
    args.num_processes = torch.cuda.device_count()
    
    ls_images = sorted(glob(os.path.join("/u/yashjain/kaggle_4/competition-data/full-test-dataset", args.group, "images", "*.tif")))
    h, w = cv2.imread(ls_images[-1], cv2.IMREAD_UNCHANGED).shape
    volume_shape = (len(ls_images), h, w)
    volume_path = f"/dev/shm/{args.group}.mmap"
    mask_path = f"/dev/shm/{args.group}_mask.mmap"
    if not os.path.exists(volume_path):
        volume = np.memmap(volume_path, shape=volume_shape, dtype=np.uint16, mode="w+")
        for i, path in enumerate(tqdm(ls_images, total=len(ls_images), desc=f"Caching {args.group} images")):
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            volume[i] = image
        volume.flush()
        del volume
    if not os.path.exists(mask_path):
        mask = np.memmap(mask_path, shape=volume_shape, dtype=np.float16, mode="w+")
        mask.fill(0.0)
        mask.flush()
        del mask
    
    args.volume_shape = volume_shape
    args.volume_path = volume_path
    args.mask_path = mask_path
    
    # inference
    queue = mp.Queue()
    write_lock = mp.Lock()
    inference_processes = []
    write_processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=main_worker, args=(rank, args, queue))
        p.start()
        inference_processes.append(p)
    for rank in range(args.num_processes*2):
        p = mp.Process(target=write_worker, args=(args, queue, write_lock))
        p.start()
        write_processes.append(p)
    for p in inference_processes:
        p.join()
    for _ in range(args.num_processes*2):
        queue.put((None, None, None, None))
    for p in write_processes:
        p.join()
        
    # write to csv
    rles, ids = [], []
    pred_masks = np.memmap(args.mask_path, shape=args.volume_shape, dtype=np.float16, mode="r")
    for i in tqdm(range(len(ls_images)), total=len(ls_images)):
        pred_mask = pred_masks[i, :, :]
        pred_mask = (pred_mask > args.threshold).astype(np.uint8)
        rle = rle_encode(pred_mask)
        path = ls_images[i].split(os.path.sep)
        dataset = path[-3]
        slice_id, _ = os.path.splitext(path[-1])
        rles.append(rle)
        ids.append(f"{dataset}_{slice_id}")
        
    df = pd.DataFrame.from_dict({
        "id": ids,
        "rle": rles
    })
    df.to_csv(f"{args.group}.csv", index=False)
    del pred_masks
    
    # clean up memmap files
    os.remove(args.volume_path)
    os.remove(args.mask_path)
