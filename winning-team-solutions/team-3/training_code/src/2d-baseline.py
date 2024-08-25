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
import wandb  
import random
import torch
import gc
from torch.nn.utils import clip_grad_value_  
wandb.login()  
import pickle
import argparse  
  
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():  
    parser = argparse.ArgumentParser(description='Configuration for the model training.')  
      
    parser.add_argument('--weight_base', type=str, default='result', help='Base directory for weights')  
    parser.add_argument('--exp_name', type=str, default='effv2s_2d_baseline', help='Experiment name')  
    parser.add_argument('--target_size', type=int, default=1, help='Prediction target size')  
      
    parser.add_argument('--model_name', type=str, default='Unet', help='Model name')  
    parser.add_argument('--backbone', type=str, default='tu-tf_efficientnetv2_s', help='Backbone model')  
      
    parser.add_argument('--in_chans', type=int, default=1, help='Number of input channels')  # 1, 3, 5
      
    parser.add_argument('--image_size', type=int, default=512, help='Image size for training')  
    parser.add_argument('--stride', type=int, default=512, help='Input size for the model')
    parser.add_argument('--input_size', type=int, default=512, help='Input size for the model')  
      
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')  
    parser.add_argument('--valid_batch_size', type=int, default=32, help='Batch size for validation')  
      
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')  
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')  
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  
    parser.add_argument('--chopping_percentile', type=float, default=1e-3, help='Chopping percentile')  
    parser.add_argument("--ce", action="store_true")
    parser.add_argument('--valid_id', type=int, default=3, help='Validation fold ID')  
    parser.add_argument("--val_trans_axis", action="store_true")
    parser.add_argument('--resume', type=str, default=None, help='Resume Path')  
    parser.add_argument('--scale_limit', type=float, default=0.3, help='Learning rate')  
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Learning rate')
    parser.add_argument('--k3_sparse', action="store_true")
    parser.add_argument('--non_empty_only', action="store_true")
    parser.add_argument('--k3_pseudo_mask_dir', type=str, default=None)
    parser.add_argument('--k2_pseudo_mask_dir', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=2, help=' ')  
    parser.add_argument('--save_start', type=int, default=2, help=' ')  
    args = parser.parse_args()  
      
    return args  
CFG = parse_args()

seed_everything(CFG.seed)

train_aug_list = [
                A.RandomCrop(CFG.input_size, CFG.input_size,p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.3,
                                   scale_limit=(-CFG.scale_limit, 0.5-CFG.scale_limit),  
                                   rotate_limit=45,
                                   # value=0,
                                   border_mode=4,
                                   p=0.95),
                A.OneOf([
                    A.MedianBlur(
                        blur_limit=3,
                        p=0.5
                    ),
                    A.GaussianBlur(
                        blur_limit=3,
                        p=0.5
                    ),
                    A.GaussNoise(
                        var_limit=(3.0, 9.0),
                        p=0.5
                    ),
                ], p=0.1),
                # A.OneOf([
                #     A.RandomBrightnessContrast(p=1.0),
                #     A.RandomGamma(p=0.5),
                # ], p=1.0),
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=0.8),
                A.OneOf([
                    # 畸变相关操作
                    A.ElasticTransform(
                        # alpha=3,
                        # sigma=50,
                        # alpha_affine=50,
                        # value=0,
                        # border_mode=cv2.BORDER_CONSTANT,
                        p= 1.0
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=1.,
                        # distort_limit=0.1,
                        # value=0,
                        # border_mode=cv2.BORDER_CONSTANT,
                        p= 1.0
                    ),
                    A.OpticalDistortion(
                        distort_limit=1.,
                        # distort_limit=0.05,
                        # shift_limit=0.05,
                        # value=0,
                        # border_mode=cv2.BORDER_CONSTANT,
                        p= 1.0
                    ),
                ],p=0.8),

                # A.CoarseDropout(
                #     max_height=int(self.cfg.image_size[0] * 0.1),
                #     max_width=int(self.cfg.image_size[1] * 0.1),
                #     max_holes = 8,
                #     # fill_value=0,
                #     # always_apply=True,
                #     p=aug.p_Dropout),
                ]
if CFG.in_chans in [1,3]:
    train_aug_list.append(
        A.OneOf([
                A.Sharpen(p=1.0),
                A.CLAHE(p=1.0),
            ], p=0.5),
    )
                    

train_aug_list.append(ToTensorV2(transpose_mask=True),)

train_aug = A.Compose(train_aug_list)
valid_aug_list = [
    ToTensorV2(transpose_mask=True),
]
valid_aug = A.Compose(valid_aug_list)

os.makedirs(CFG.weight_base, exist_ok=True)
SAVE_BASE = f'{CFG.weight_base}/{CFG.exp_name}'
os.makedirs(SAVE_BASE, exist_ok=True)

wandb.init(  
    project="SenNet",   
    group=CFG.exp_name,   
    name=f"{CFG.exp_name}/val_kidney{CFG.valid_id}",   
    dir=SAVE_BASE,  
    settings=wandb.Settings(start_method="fork")  
)  


class CustomModel(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        if CFG.model_name == 'Unet':
            self.model = smp.Unet(
                encoder_name=CFG.backbone, 
                encoder_weights=weight,
                in_channels=CFG.in_chans,
                classes=CFG.target_size,
                activation=None,
            )
        elif CFG.model_name == 'UnetPlusPlus':
            self.model = smp.UnetPlusPlus(
                encoder_name=CFG.backbone, 
                encoder_weights=weight,
                in_channels=CFG.in_chans,
                classes=CFG.target_size,
                activation=None,
            )            

    def forward(self, image):
        output = self.model(image)
        return output[:,0]


def build_model(weight="imagenet"):
    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)
    model = CustomModel(CFG, weight)

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
    index = -int(len(TH) * CFG.chopping_percentile)
    TH:int = np.partition(TH, index)[index]
    x[x>TH]=int(TH)
    ########################################################################
    TH=x.reshape(-1)
    index = -int(len(TH) * CFG.chopping_percentile)
    TH:int = np.partition(TH, -index)[-index]
    x[x<TH]=int(TH)
    return x

#https://www.kaggle.com/code/kashiwaba/sennet-hoa-train-unet-simple-baseline
def dice_coef(y_pred:tc.Tensor,y_true:tc.Tensor, thr=0.5, dim=(-1,-2), epsilon=0.001):
    y_pred=y_pred.sigmoid()
    y_true = y_true.to(tc.float32)
    y_pred = (y_pred>thr).to(tc.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()
    return dice


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.sigmoid()   
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def get_indexes_along_axis0(num_slice, height, width, image_size, stride, slice_offset, slice_ls=None):
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
    if slice_ls is None: slice_ls = range(slice_offset, num_slice-slice_offset)
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

def find_slice_ls(image_list):
    image_name = image_list[0]
    for i in range(3):
        kidney_name = f'kidney_{i+1}'
        if kidney_name in image_name:
            return i+1


class Dataset3D(Dataset):
    def __init__(self, images_list, labels_list, trans_axis=False,cfg=CFG, aug=True, non_empty_dict=None):
        super(Dataset,self).__init__()

        self.image_size=cfg.image_size
        self.in_chans=cfg.in_chans
        assert self.in_chans % 2 == 1
        self.slice_offset = (self.in_chans - 1)//2
        self.images = []
        self.labels = []
        self.coors, self.idx_3d = [], []
        for kidney_index, (image_list, label_list) in enumerate(zip(images_list, labels_list)):
            if non_empty_dict is not None: mask_dict = non_empty_dict[find_slice_ls(image_list)]
            images = [cv2.imread(x,cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :] for x in image_list]
            labels = [cv2.imread(x,cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :] for x in label_list]
            images = np.concatenate(images, axis=0)## N, H, W
            labels = np.concatenate(labels, axis=0)

            images = torch.tensor(filter_noise(images))
            images = (min_max_normalization(images.to(tc.float16)[None])[0]*255).to(tc.uint8).numpy()

            num_slice, height, width = images.shape
            # if trans_axis and num_slice < cfg.image_size:
            self.images.append(images)
            self.labels.append(labels)
            tmp_coors = get_indexes_along_axis0(num_slice, height, width, cfg.image_size, cfg.stride, self.slice_offset, slice_ls=None if non_empty_dict is None else mask_dict[0])
            self.coors.extend(tmp_coors)
            if trans_axis:
                self.idx_3d.extend([kidney_index*3] * len(tmp_coors)) 
            else:
                self.idx_3d.extend([kidney_index] * len(tmp_coors)) 

            if trans_axis:
                self.images.append(np.transpose(images, (1,2,0))) # H, W, N
                self.labels.append(np.transpose(labels, (1,2,0))) # H, W, N    
                width, num_slice, height = images.shape
                tmp_coors = get_indexes_along_axis0(num_slice, height, width, min(cfg.image_size, width-1), min(cfg.stride, width-1), self.slice_offset, slice_ls=None if non_empty_dict is None else mask_dict[1])
                self.coors.extend(tmp_coors)
                self.idx_3d.extend([kidney_index*3+1]*len(tmp_coors))

                self.images.append(np.transpose(images, (2,0,1))) # W, N, H
                self.labels.append(np.transpose(labels, (2,0,1))) # W, N, H    
                height, width, num_slice = images.shape
                tmp_coors = get_indexes_along_axis0(num_slice, height, width, min(cfg.image_size, height-1), min(cfg.stride, height-1), self.slice_offset, slice_ls=None if non_empty_dict is None else mask_dict[2])
                self.coors.extend(tmp_coors)
                self.idx_3d.extend([kidney_index*3+2]*len(tmp_coors))


        if aug:
            self.transform=train_aug
        else: 
            self.transform=valid_aug

    def __len__(self):
        return len(self.coors)

    def __getitem__(self,index):
        n, h, w = self.coors[index]
        idx_3d = self.idx_3d[index]
        label = self.labels[idx_3d][n, h:h+self.image_size, w:w+self.image_size] # (H, W)
        if self.slice_offset == 0:
            image = self.images[idx_3d][n, h:h+self.image_size, w:w+self.image_size] # (H, W)
        else:
            image = self.images[idx_3d][n-self.slice_offset:n+self.slice_offset+1, h:h+self.image_size, w:w+self.image_size] # (3(5), H, W)
            image = np.transpose(image, (1, 2, 0))
        if h < self.image_size or w < self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))
            label = cv2.resize(label, (self.image_size, self.image_size))
        data = self.transform(image=image, mask=label)
        image = data['image']
        label = data['mask']>=127
        return image,label

model=build_model()
if CFG.resume is not None: model.load_state_dict(torch.load(CFG.resume,"cuda:0"), strict=True)

k1_image_path_list = glob(os.path.join(f"/home/v-honsong/dataset/senet/train/kidney_1_dense/images", '*.tif'))
k1_label_path_list = glob(os.path.join(f"/home/v-honsong/dataset/senet/train/kidney_1_dense/labels", '*.tif'))
k1_image_path_list.sort()
k1_label_path_list.sort()


k2_image_path_list = glob(os.path.join(f"/home/v-honsong/dataset/senet/train/kidney_2/images", '*.tif'))
k2_image_path_list.sort()
k2_image_path_list = k2_image_path_list[900:]
if CFG.k2_pseudo_mask_dir is None:
    k2_label_path_list = glob(os.path.join(f"/home/v-honsong/dataset/senet/train/kidney_2/labels", '*.tif'))
    k2_label_path_list.sort()
    k2_label_path_list = k2_label_path_list[900:]
else:
    k2_label_path_list = []
    for k2_image_path in k2_image_path_list:
        k2_image_name = os.path.basename(k2_image_path).replace('.tif', '.png')
        k2_label_path_list.append(f'{CFG.k2_pseudo_mask_dir}/{k2_image_name}')


path1="/home/v-honsong/dataset/senet/train/kidney_3_sparse"
path2="/home/v-honsong/dataset/senet/train/kidney_3_dense"
if CFG.k3_sparse and (CFG.k3_pseudo_mask_dir is not None):
    print('Can not assign both k3_sparse and k3_pseudo!')
    exit(0)
if CFG.k3_pseudo_mask_dir is not None:
    k3_dense_label_list = set(glob(f"{path2}/labels/*"))
    k3_image_path_list=glob(f"{path1}/images/*")
    k3_image_path_list.sort()
    k3_label_path_list = []
    for k3_image_path in k3_image_path_list:
        k3_label_path = k3_image_path.replace('images', 'labels')
        k3_dense_label_path = k3_label_path.replace('sparse', 'dense')
        if k3_dense_label_path in k3_dense_label_list:
            k3_label_path_list.append(k3_dense_label_path)
        else:
            label_name = os.path.basename(k3_dense_label_path).replace('.tif', '.png')
            pseudo_path = f'{CFG.k3_pseudo_mask_dir}/{label_name}'
            k3_label_path_list.append(pseudo_path)
else:
    if CFG.k3_sparse:
        k3_dense_label_list = set(glob(f"{path2}/labels/*"))
        k3_sparse_label_list = glob(f"{path1}/labels/*")

        k3_image_path_list=glob(f"{path1}/images/*")
        k3_image_path_list.sort()
        k3_label_path_list = []
        for k3_image_path in k3_image_path_list:
            k3_label_path = k3_image_path.replace('images', 'labels')
            k3_dense_label_path = k3_label_path.replace('sparse', 'dense')
            if k3_dense_label_path in k3_dense_label_list:
                k3_label_path_list.append(k3_dense_label_path)
            else:
                k3_label_path_list.append(k3_label_path)
    else:
        k3_label_path_list=glob(f"{path2}/labels/*")
        k3_image_path_list=[x.replace("labels","images").replace("dense","sparse") for x in k3_label_path_list]
        k3_image_path_list.sort()
        k3_label_path_list.sort()

if CFG.valid_id == 1:
    train_image_path_list = [k2_image_path_list, k3_image_path_list]
    train_label_path_list = [k2_label_path_list, k3_label_path_list]
    val_image_path_list = [k1_image_path_list]
    val_label_path_list = [k1_label_path_list]
elif CFG.valid_id == 2:
    train_image_path_list = [k1_image_path_list, k3_image_path_list]
    train_label_path_list = [k1_label_path_list, k3_label_path_list]
    val_image_path_list = [k2_image_path_list]
    val_label_path_list = [k2_label_path_list]
elif CFG.valid_id == 3:
    train_image_path_list = [k1_image_path_list, k2_image_path_list]
    train_label_path_list = [k1_label_path_list, k2_label_path_list]
    val_image_path_list = [k3_image_path_list]
    val_label_path_list = [k3_label_path_list]
else:
    train_image_path_list = [k1_image_path_list, k2_image_path_list, k3_image_path_list]
    train_label_path_list = [k1_label_path_list, k2_label_path_list, k3_label_path_list]   
    val_image_path_list, val_label_path_list = [], []

if CFG.non_empty_only:
    with open('test_scripts/nonempty_masks.pkl', 'rb') as f:
        non_empty_dict = pickle.load(f)
else:
    non_empty_dict = None

train_dataset = Dataset3D(train_image_path_list, train_label_path_list, trans_axis=True, non_empty_dict=non_empty_dict)
train_dataset = DataLoader(train_dataset, batch_size=CFG.train_batch_size ,num_workers=24, shuffle=True, pin_memory=True)

if len(val_image_path_list) > 0:
    val_dataset = Dataset3D(val_image_path_list, val_label_path_list, trans_axis=True if CFG.val_trans_axis else False, aug=False)
    val_dataset = DataLoader(val_dataset, batch_size=CFG.valid_batch_size ,num_workers=24, shuffle=False, pin_memory=True)

# tc.backends.cudnn.enabled = True
# tc.backends.cudnn.benchmark = True
# model=DataParallel(model)

loss_fc=DiceLoss()
if CFG.ce: loss_fn=nn.BCEWithLogitsLoss()
optimizer=tc.optim.AdamW(model.parameters(),lr=CFG.lr)
scaler=tc.cuda.amp.GradScaler()
scheduler = tc.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr,
                                                steps_per_epoch=len(train_dataset), epochs=CFG.epochs+1,
                                                pct_start=(1/CFG.epochs),)

best_score, best_loss = 0., 100.
for epoch in range(CFG.epochs):
    model.train()
    time=tqdm(range(len(train_dataset)))
    losss=0
    scores=0
    for batch_idx,(x,y) in enumerate(train_dataset):
        x=x.cuda().to(tc.float32)
        x=norm_with_clip(x.reshape(-1,*x.shape[2:])).reshape(x.shape)
        x=add_noise(x,max_randn_rate=0.5,x_already_normed=True)
        y=y.cuda().to(tc.float32)
        
        with autocast():
            pred=model(x)
            loss=loss_fc(pred,y)
            if CFG.ce: loss += loss_fn(pred,y)
        scaler.scale(loss).backward()
        clip_grad_value_(model.parameters(), clip_value=CFG.grad_clip) 
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        score=dice_coef(pred.detach(),y)
        losss=(losss*batch_idx+loss.item())/(batch_idx+1)
        scores=(scores*batch_idx+score)/(batch_idx+1)
        time.set_description(f"epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}")
        time.update()
        del loss,pred
    time.close()
    wandb.log({"epoch": epoch, "train_loss": losss, "train_dice": scores, "lr": optimizer.param_groups[0]['lr']})
    if len(val_image_path_list) > 0:
        model.eval()
        time=tqdm(range(len(val_dataset)))
        val_losss=0
        val_scores=0
        for i,(x,y) in enumerate(val_dataset):
            x=x.cuda().to(tc.float32)
            x=norm_with_clip(x.reshape(-1,*x.shape[2:])).reshape(x.shape)
            y=y.cuda().to(tc.float32)

            with autocast():
                with tc.no_grad():
                    pred=model(x)
                    loss=loss_fc(pred,y)
            score=dice_coef(pred.detach(),y)
            val_losss=(val_losss*i+loss.item())/(i+1)
            val_scores=(val_scores*i+score)/(i+1)
            time.set_description(f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}")
            time.update()
        wandb.log({"epoch": epoch, "val_loss": val_losss, "val_dice": val_scores})  
        if val_scores > best_score:
            best_score = val_scores
            tc.save(model.state_dict(),f"./{SAVE_BASE}/valonK{CFG.valid_id}_best_dice.pt")
        if val_losss < best_loss:
            best_loss = val_losss
            tc.save(model.state_dict(),f"./{SAVE_BASE}/valonK{CFG.valid_id}_best_loss.pt")
        time.close()
    else:
        if epoch >= CFG.save_start and (epoch+1)%CFG.save_interval == 0:
            try:
                tc.save(model.state_dict(),f"./{SAVE_BASE}/valonK{str(CFG.valid_id)}_{str(epoch+1)}.pt")
            except:
                print(f'Warning: Failed to save checkpoint in every 5 epochs!')
tc.save(model.state_dict(),f"./{SAVE_BASE}/valonK{str(CFG.valid_id)}_last.pt")
time.close()