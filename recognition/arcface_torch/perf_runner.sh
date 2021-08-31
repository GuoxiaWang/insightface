set -ex

gpus=${1:-0,1,2,3,4,5,6,7}
config_file=${2:-configs/ms1mv3_r50}
mode=${3:-dynamic}
num_classes=${4:-93431}
dtype=${5:-fp16}
num_nodes=${6:-1}
batch_size_per_device=${7:-128}
sample_ratio=${8:-0.1}
test_id=${9:-1}

sed -i 's/config.sample_rate = 1.0/config.sample_rate = 0.1/g' configs/ms1mv3_benchmark.py
sed -i "s/config.batch_size = [[:digit:]]*/config.batch_size = ${batch_size_per_device}/g" configs/ms1mv3_benchmark.py
sed -i "s/config.num_classes = [[:digit:]]*/config.num_classes = ${num_classes}/g" configs/ms1mv3_benchmark.py

if [ $dtype = "fp16" ]; then
    fp16=True
    sed -i 's/config.fp16 = False/config.fp16 = True/g' configs/ms1mv3_benchmark.py
else
    fp16=False
    sed -i 's/config.fp16 = True/config.fp16 = False/g' configs/ms1mv3_benchmark.py
fi

if [[ $config_file =~ r50 ]]; then
    backbone=r50
    sed -i 's/config.network = "r100"/config.network = "r50"/g' configs/ms1mv3_benchmark.py
else
    backbone=r100
    sed -i 's/config.network = "r50"/config.network = "r100"/g' configs/ms1mv3_benchmark.py
fi

gpu_num_per_node=`expr ${#gpus} / 2 + 1`

log_dir=./logs/arcface_torch_${backbone}_${mode}_${dtype}_r${sample_ratio}_bz${batch_size_per_device}_${num_nodes}n${gpu_num_per_node}g_id${test_id}
mkdir -p $log_dir
log_file=$log_dir/workerlog.0

    
CUDA_VISIBLE_DEVICES=${gpus} python -m torch.distributed.launch --max_restarts 0 --nproc_per_node=${gpu_num_per_node} --nnodes=${num_nodes} --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/ms1mv3_benchmark.py 2>&1 | tee $log_file
