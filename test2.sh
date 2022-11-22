CUDA_VISIBLE_DEVICES=2 python -m pdb main.py test \
-d /mnt/yuhang/dataset/datasets \
--classes 60 \
-s 512 \
--resume /mnt/yuhang/Cerberus/model_best_fea128.pth.tar \
--phase test \
--batch-size 1 \
--ms \
--workers 10