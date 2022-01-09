set -e

num_test=5
num_nodes=2

networks=(r50 r100)
dtypes=(fp16 fp32)
gpus=("0" "0,1,2,3" "0,1,2,3,4,5,6,7")
num_class=93432

if [ "$num_nodes" -gt 1 ]; then
   gpus=("0,1,2,3,4,5,6,7")
   num_class=93424
fi

for network in "${networks[@]}"
do
    for dtype in "${dtypes[@]}"
    do
        for gpu in "${gpus[@]}"
        do
            i=1
            while [ $i -le ${num_test} ]
            do
                bash perf_runner.sh $gpu $network static $num_class $dtype $num_nodes 128 0.1 ${i}
                echo " >>>>>>Finished Test Case $network, $dtype, $gpu, ${i} <<<<<<<"
                let i++
                sleep 20s
            done
        done
    done
done
