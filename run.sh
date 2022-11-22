CUDA_VISIBLE_DEVICES=2 \
python main.py single \
--crop-size 512 \
--classes 40 \
--batch-size 4 \
--random-scale 2 \
--random-rotate 10 \
--epochs 200 \
--lr 0.01 \
--momentum 0.9 \
--lr-mode poly \
--workers 12 
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=23453 main.py train \
                # "train",
                # "--classes", "40",
                # "--crop-size", "512",
                # "--batch-size", "4",
                # "--phase", "train",
                # "--random-scale", "2",
                # "--random-rotate", "10",
                # "--epochs", "200",
                # "--lr", "0.007",
                # "--momentum", "0.9",
                # "--lr-mode", "poly",
                # "--workers", "12",