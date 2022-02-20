#!/bin/bash
# sets=(mnist fmnist celeba)
source venv/bin/activate


sets=($1)
# sets=(celeba)
for set in ${sets[*]};
do
    echo "Start training on $set."
#    timeout 30m python train.py --cfg experiments/cfgs/gans/$set-long-measure.yml --is_train
    # python train_and_measure_gan.py \
    #     --cfg experiments/cfgs/gans/$set-long-measure.yml \
    #     --results_dir train_and_measure_gan_$set \
    #     --probe_size 50000 \
    #     --is_train \
    #     --iters 100

    for i in $(seq 1 100);
    do
        mkdir -p debug3/gans/$set/$i
        mkdir -p debug3/debug/gans/$set/$i
        mv debug/gans/$set-long-measure/* debug3/gans/$set/$i/
        mv debug/debug/gans/$set-long-measure/* debug3/debug/gans/$set/$i/
        echo "Running $i turn of training."
        # timeout 30m python train.py --cfg experiments/cfgs/gans/$set-long-measure.yml --is_train
        
        # if ["$set" = celeba ]
        # then
        #     probe_size=20000
        # else
        #     probe_size=50000
        # fi


        probe_size=2000


        python generate_measurement_samples.py --cfg output/gans/$set-long-measure --results_dir train_and_measure_gan_$set --probe_size $probe_size --iter $i
        python measure_gan.py --cfg output/gans/$set-long-measure --results_dir train_and_measure_gan_$set --probe_size $probe_size --iter $i

        echo "Finished $i turn of training."

    done
    i=51
    mkdir -p debug3/gans/$set/$i
    mkdir -p debug3/debug/gans/$set/$i
    mv debug/gans/$set-long-measure/* debug3/gans/$set/$i/
    mv debug/debug/gans/$set-long-measure/* debug3/debug/gans/$set/$i/
    
    echo "Training on $set finished."
done
