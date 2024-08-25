## step1: train on dense mask only
python src/2d-baseline.py --epochs 20 --exp_name Denseonly_maxViTlarge --backbone tu-maxvit_large_tf_512 --model_name Unet --train_batch_size 20 --valid_batch_size 20 --lr 1e-5 --valid_id 2 --scale_limit 0.45 --val_trans_axis
python src/2d-baseline.py --epochs 20 --exp_name Denseonly_Effv2s --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 2e-4 --valid_id 2 --scale_limit 0.45 --val_trans_axis

## step2: make pseudo mask for k3 sparse (You may modify some config and path in the code)
python src/valid_2stage.py  ## Inference and save the float pseudo mask
python pseudo_mask/make_pseudo_on_sparse.py  ## Save the binarized pseudo mask with different thresdhold

## step3: train on k1+k3
python src/2d-baseline.py --epochs 10 --exp_name k1_k3pseu_Effv2s --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 1e-4 --valid_id 2 --scale_limit 0.45 --val_trans_axis --k3_pseudo_mask_dir pseudo_mask/k3sparse/mask0.15 --resume result/Denseonly_Effv2s/valonK2_best_loss.pt
python src/2d-baseline.py --epochs 10 --exp_name k1_k3pseu_maxViTlarge --backbone tu-maxvit_large_tf_512 --model_name Unet --train_batch_size 20 --valid_batch_size 20 --lr 2e-5 --valid_id 2 --scale_limit 0.45 --k3_pseudo_mask_dir pseudo_mask/k3sparse/mask0.15 --resume result/Denseonly_maxViTlarge/valonK2_last.pt

## step4: make pseudo mask for k2 (You may modify some config and path in the code)
python src/valid_2stage.py
python pseudo_mask/make_pseudo_on_sparse.py

## step5: train on k1+k3+k2
python src/2d-baseline.py --epochs 10 --exp_name allpse_UnetEffv2s --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 128 --valid_batch_size 128 --lr 1e-4 --valid_id -1 --scale_limit 0.45 --k3_pseudo_mask_dir pseudo_mask/k3sparse/mask0.15 --k2_pseudo_mask_dir pseudo_mask/k2/mask0.1 --resume result/k1_k3pseu_Effv2s/valonK2_best_loss.pt
python src/2d-baseline.py --epochs 10 --exp_name allpse_UnetppEffv2l --backbone tu-tf_efficientnetv2_l --model_name UnetPlusPlus --train_batch_size 32 --valid_batch_size 32 --lr 1e-4 --valid_id -1 --scale_limit 0.45 --k3_pseudo_mask_dir pseudo_mask/k3sparse/mask0.15 --k2_pseudo_mask_dir pseudo_mask/k2/mask0.2
python src/2d-baseline.py --epochs 12 --exp_name allpse_UnetSeRes --backbone tu-seresnext101d_32x8d --model_name Unet --train_batch_size 64 --valid_batch_size 64 --lr 1e-4 --valid_id -1 --scale_limit 0.45 --k3_pseudo_mask_dir pseudo_mask/k3sparse/mask0.15 --k2_pseudo_mask_dir pseudo_mask/k2/mask0.15
python src/2d-baseline.py --epochs 8 --exp_name allpse_maxViTlarge --backbone tu-maxvit_large_tf_512 --model_name Unet --train_batch_size 20 --valid_batch_size 20 --lr 2e-5 --valid_id -1 --scale_limit 0.45 --k3_pseudo_mask_dir pseudo_mask/k3sparse/mask0.15 --k2_pseudo_mask_dir pseudo_mask/k2/mask0.1 --resume result/k1_k3pseu_maxViTlarge/valonK2_best_dice.pt
python src/2d-baseline.py --epochs 5 --exp_name allpse_UnetEffv2s832 --backbone tu-tf_efficientnetv2_s --model_name Unet --train_batch_size 48 --valid_batch_size 48 --lr 2e-5 --valid_id -1 --scale_limit 0.45 --k3_pseudo_mask_dir pseudo_mask/k3sparse/mask0.15 --k2_pseudo_mask_dir pseudo_mask/k2/mask0.1 --resume result/allpse_UnetEffv2s/valonK-1_last.pt --input_size 832 --stride 832 --image_size 832