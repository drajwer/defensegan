#!/bin/bash
# sets=(mnist fmnist celeba)
source venv/bin/activate


sets=(mnist fmnist celeba)
for set in ${sets[*]};
do
    echo "Start training on $set."
    for i in $(seq 1 200);
    do
        mkdir -p debug3/gans/$set/$i
        mkdir -p debug3/debug/gans/$set/$i
        mv debug/gans/$set/* debug3/gans/$set/$i/
        mv debug/debug/gans/$set/* debug3/debug/gans/$set/$i/
        echo "Running $i turn of training."
        timeout 30m python train.py --cfg experiments/cfgs/gans/$set-long-measure.yml --is_train
        python measure_gan.py --cfg output/gans/$set-long-measure --results_dir train_and_measure_gan_long --probe_size 20000 --iter $i
        echo "Finished $i turn of training."

    done
    i=201
    mkdir -p debug3/gans/$set/$i
    mkdir -p debug3/debug/gans/$set/$i
    mv debug/gans/$set/* debug3/gans/$set/$i/
    mv debug/debug/gans/$set/* debug3/debug/gans/$set/$i/
    
    echo "Training on $set finished."
done
