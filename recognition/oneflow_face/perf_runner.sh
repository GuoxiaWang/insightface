#！/bin/bash
#export ONEFLOW_DEBUG_MODE=True
export PYTHONUNBUFFERED=1

set -ex

gpus=${1:-0,1,2,3,4,5,6,7}
network=${2:-r50}
mode=${3:-static}
num_classes=${4:-93432}
dtype=${5:-fp16}
num_nodes=${6:-1}
batch_size_per_device=${7:-128}
sample_ratio=${8:-0.1}
test_id=${9:-1}


loss=${10:-arcface}
train_unit=${11:-batch}
train_iter=${12:-200}
gpu_num_per_node=${13:-8}
model_parallel=${14:-1}
partial_fc=${15:-1}
sample_ratio=${16:-0.1}
use_synthetic_data=${17:-True}
do_validation_while_train=${18:-False}
dataset=${19:-ms1mv3}

gpu_num_per_node=`expr ${#gpus} / 2 + 1`

sed -i "s/${dataset}.num_classes = [[:digit:]]*/${dataset}.num_classes = ${num_classes}/g" config.py

PREC=""
if [ "$dtype" = "fp16" ]; then
   PREC=" --use_fp16=True"
elif [ "$dtype" = "fp32" ]; then
   PREC=" --use_fp16=False"
else
   echo "Unknown <precision> argument"
   exit -2
fi

IPS=""
if [ "$num_nodes" -gt 1 ]; then
   IPS=" --node_ips=${TRIANER_IP_LIST}"
fi

time=$(date "+%Y-%m-%d %H:%M:%S")
echo $time

log_dir=./logs/arcface_oneflow_${network}_${mode}_${dtype}_r${sample_ratio}_bz${batch_size_per_device}_${num_nodes}n${gpu_num_per_node}g_id${test_id}
mkdir -p $log_dir
log_file=$log_dir/workerlog.0

CMD="insightface_train.py"
CMD+=" --network=${network}"
CMD+=" --dataset=${dataset}"
CMD+=" --loss=${loss}"
CMD+=" --num_nodes=${num_nodes}"
CMD+=" --train_batch_size=$(expr $num_nodes '*' $gpu_num_per_node '*' $batch_size_per_device)"
CMD+=" --train_unit=${train_unit}"
CMD+=" --train_iter=${train_iter}"
CMD+=" --device_num_per_node=${gpu_num_per_node}"
CMD+=" --model_parallel=${model_parallel}"
CMD+=" --partial_fc=${partial_fc}"
CMD+=" --sample_ratio=${sample_ratio}"
CMD+=" --log_dir=${log_dir}"
CMD+=" $PREC"
CMD+=" $IPS"
CMD+=" --sample_ratio=${sample_ratio}"
CMD+=" --use_synthetic_data=${use_synthetic_data}"
CMD+=" --do_validation_while_train=${do_validation_while_train}"
CMD+=" --iter_num_in_snapshot=5000"
CMD+=" --validation_interval=5000"

CMD="python $CMD "
flag_exit=0
while [ $flag_exit -ne 1 ]
do
  rm -rf ${workspace}/new_models
  set -x
  if [ -z "$log_file" ]; then
     $CMD
  else
     (
        $CMD
     ) |& tee $log_file
  fi
  FIND_STR="Check failure stack trace"
  if [ `grep -c "$FIND_STR" $log_file` -ne '0' ];then
    echo "Error!"
    flag_exit=0
    rm -rf $log_file
  else
    flag_exit=1
  fi
done

echo "Writing log to ${log_file}"
