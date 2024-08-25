python src/2d-baseline.py --epochs 10 --exp_name Denseonly_maxViTlarge --backbone tu-maxvit_large_tf_512 --model_name Unet --train_batch_size 20 --valid_batch_size 20 --lr 1e-5 --valid_id 2 --scale_limit 0.45 --val_trans_axis --resume result/UNetmaxvitlarge512_900onwards/valonK2_last.pt
python src/2d-baseline.py --epochs 20 --exp_name Denseonly_Effv2s --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 2e-4 --valid_id 2 --scale_limit 0.45 --val_trans_axis

# python src/2d-baseline.py --epochs 10 --exp_name NonemptyMask_UNeteffv2s_900onwards_Scale55-105_noval --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 2e-5 --valid_id -1 --k3_sparse --scale_limit 0.45 --val_trans_axis --non_empty_only --resume result/0201UNeteffv2s_900onwards_Scale55-105_noval/valonK-1_last.pt
# python src/2d-baseline.py --epochs 20 --exp_name 0201UNetSeRes101_900onwards_Scale55-105_noval --backbone tu-seresnext101d_32x8d --model_name Unet --train_batch_size 64 --valid_batch_size 64 --lr 1e-4 --valid_id -1 --k3_sparse --scale_limit 0.45 --val_trans_axis
# 
# python src/2d-baseline.py --epochs 12 --exp_name NonemptyMask_UNetmaxViTLarge_900onwards_Scale55-105_noval --backbone tu-maxvit_large_tf_512 --model_name Unet --train_batch_size 20 --valid_batch_size 20 --lr 1e-5 --valid_id 3 --k3_sparse --scale_limit 0.45 --val_trans_axis --non_empty_only --resume result/UNetmaxvitlarge512_900onwards/valonK3_best_loss.pt
# python src/2d-baseline.py --epochs 20 --exp_name NonemptyMask_UNeteffv2l_900onwards_Scale55-105_noval --backbone tu-tf_efficientnetv2_l --model_name Unet --train_batch_size 48 --valid_batch_size 48 --lr 1e-4 --valid_id -1 --k3_sparse --scale_limit 0.45 --val_trans_axis --non_empty_only

# python src/2d-baseline.py --epochs 20 --exp_name NonemptyMask_UNeteffv2l_900onwards_Scale55-105_valtrans --backbone tu-tf_efficientnetv2_l --model_name Unet --train_batch_size 48 --valid_batch_size 48 --lr 1e-4 --valid_id 3 --k3_sparse --scale_limit 0.45 --val_trans_axis --non_empty_only

# python src/2d-baseline.py --epochs 30 --exp_name UNet3c_effv2s_900onwards_Scale03 --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 2e-4 --valid_id 3 --scale_limit 0.3 --in_chans 3
# python src/2d-baseline.py --epochs 20 --exp_name UNetpp_effv2l_900onwards_Scale05 --backbone tu-tf_efficientnetv2_l --model_name UnetPlusPlus --train_batch_size 32 --valid_batch_size 32 --lr 1e-4 --valid_id 3 --scale_limit 0.5

# python src/2d-baseline.py --epochs 20 --exp_name UNeteffv2s900onwards_768 --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 64 --valid_batch_size 64 --lr 1e-4 --valid_id 3 --image_size 768 --stride 768 --input_size 768
# python src/2d-baseline.py --epochs 30 --exp_name UNeteffv2s900onwards_512_moreBriConGammaaug --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 2e-4 --valid_id 3 
# python src/2d-baseline_general_norm.py --epochs 20 --exp_name GenNorm_UNeteffv2s900onwards_morebriaug --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 2e-4 --valid_id 3
# python src/2d-baseline_general_norm.py --epochs 20 --exp_name GenNorm_UNeteffv2s900onwards_morebriaug --backbone tu-tf_efficientnetv2_l --model_name Unet --train_batch_size 48 --valid_batch_size 48 --lr 1e-4 --valid_id 3
# python src/2d-baseline.py --epochs 20 --exp_name UNet_maxvit_tiny_512 --backbone tu-maxvit_tiny_tf_512 --model_name Unet --train_batch_size 64 --valid_batch_size 64 --lr 1e-4

# python src/2d-baseline.py --epochs 10 --exp_name UNetmaxvitlarge512_noval_k3sparse_resume5_Scale55-105 --backbone tu-maxvit_large_tf_512 --model_name Unet --train_batch_size 20 --valid_batch_size 20 --lr 1e-5 --valid_id -1 --k3_sparse --resume result/UNetmaxvitlarge512_noval/valonK-1_last.pt --scale_limit 0.45