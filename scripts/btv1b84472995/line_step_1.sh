# put cuda to 1
# cd /home/vlachoum/learnable-DTLR
# CUDA_VISIBLE_DEVICES=1 python reconstruction.py \
#   --dataset_file dataset \
#   --data_folder btv1b84472995 \
#   --space_index 0 \
#   --max_e 120 \
#   --num_fine_classes 2 \
#   --batch_size 24 \
#   --step 1 \
#   --loss L1 \
#   --wandb \
#   --resume \
#   --learning_rate 1e-4 \
#   --weight_loss_reconstruction 10 \
#   --tag ICDAR_btv1b84472995_pretrain2_without_mask_lr-1e-4-res48x48_batchsize24-epochs60-old_checkpoint \
#   --model_checkpoint_path /home/vlachoum/learnable-DTLR/logs_reconstruction/ICDAR_btv1b84472995_pretrain1_without_mask_lr-1e-4-res48x48_batchsize24-epochs20-old_checkpoint/model.pth \
#   --reconstructor_path /home/vlachoum/learnable-DTLR/logs_reconstruction/ICDAR_btv1b84472995_pretrain1_without_mask_lr-1e-4-res48x48_batchsize24-epochs20-old_checkpoint/reconstructor.pth

#   # put cuda to 1
# cd /home/vlachoum/learnable-DTLR
CUDA_VISIBLE_DEVICES=1 python reconstruction.py \
    --dataset_file dataset \
    --data_folder btv1b84472995 \
    --space_index 0 \
    --model_config_path config/Latin_accent.py \
    --max_e 100 \
    --num_fine_classes 2 \
    --step 1 \
    --batch_size 8 \
    --wandb \
    --learning_rate 1e-4 \
    --weight_loss_reconstruction 3 \
    --tag btv1b84472995_step_1_weight_loss_reconstruction_3_bis \
    --loss L1 \
    --model_checkpoint_path /home/rbaena/projects/learnable-DTLR/logs_reconstruction/btv1b84472995_step_0/model.pth \
    --reconstructor_path /home/rbaena/projects/learnable-DTLR/logs_reconstruction/btv1b84472995_step_0/reconstructor.pth
