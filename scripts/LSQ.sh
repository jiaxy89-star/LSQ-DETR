coco_path=$1
torchrun --nproc_per_node=3 main_aitod.py \
  --output_dir logs/LSQDETR_ver1 -c config/LSQ_5scale.py --coco_path $coco_path \
  --pretrain_model_path /root/autodl-tmp/project/pretrain_model.pth \
  --options dn_scalar=100 embed_init_tgt=False \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0