#python -W ignore -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 train.py \
#    --network MobileFaceNet_128 \
#    --embedding_size 128 \
#    --model_parallel False \
#    --sample_ratio 0.1 \
#    --loss ArcFace \
#    --batch_size 64 \
#    --dataset emore \
#    --num_classes 85742 \
#    --data_dir /wangguoxia/plsc/MS1M_v2/ \
#    --label_file /wangguoxia/plsc/MS1M_v2/label.txt \
#    --is_bin False \
#    --log_interval_step 100


python -m paddle.distributed.launch --gpus=4,5,6,7 train_static.py \
    --backbone FresResNet100 \
    --classifier LargeScaleClassifier \
    --embedding_size 512 \
    --model_parallel False \
    --sample_ratio 0.1 \
    --loss ArcFace \
    --batch_size 128 \
    --dataset emore \
    --num_classes 85742 \
    --data_dir /wangguoxia/plsc/MS1M_v2/ \
    --label_file /wangguoxia/plsc/MS1M_v2/label.txt \
    --is_bin False \
    --log_interval_step 100 \
    --validation_interval_step 100
