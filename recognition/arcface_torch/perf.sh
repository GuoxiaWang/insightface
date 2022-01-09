set -e

num_test=5
num_nodes=2

configs=(configs/ms1mv3_r50 configs/ms1mv3_r100)
dtypes=(fp16 fp32)
gpus=("0" "0,1,2,3" "0,1,2,3,4,5,6,7")
if [ "$num_nodes" -gt 1 ]; then
   gpus=("0,1,2,3,4,5,6,7")
fi

for config in "${configs[@]}"
do
    for dtype in "${dtypes[@]}"
    do
        for gpu in "${gpus[@]}"
        do
            i=1
            while [ $i -le ${num_test} ]
            do
                bash perf_runner.sh $gpu $config dynamic 93431 $dtype $num_nodes 128 0.1 ${i}
                echo " >>>>>>Finished Test Case $config, $dtype, $gpu, ${i} <<<<<<<"
                let i++
                sleep 20s
            done
        done
    done
done
