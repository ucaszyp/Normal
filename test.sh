CUDA_VISIBLE_DEVICES=3 python main.py test \
-d /mnt/yuhang/dataset/ade20k \
--classes 150 \
-s 512 \
--resume /mnt/yuhang/Cerberus/checkpoint_fea128_ade20k_180.pth.tar \
--phase test \
--batch-size 1 \
--ms \
--workers 10
