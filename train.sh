#!/bin/bash
# sets=(mnist fmnist celeba)
source venv/bin/activate

# python blackbox.py --cfg output/gans/celeba-2 \
#     --results_dir celeba-1-2 \
#     --bb_model A \
#     --sub_model B \
#     --fgsm_eps 0.01 \
#     --defense_type defense_gan --debug True --debug_dir debug/blackbox-celeba-1-2


python blackbox.py --cfg output/gans/celeba-3 \
    --results_dir celeba-1-3 \
    --bb_model A \
    --sub_model B \
    --fgsm_eps 0.01 \
    --defense_type defense_gan --debug True --debug_dir debug/blackbox-celeba-1-3


python blackbox.py --cfg output/gans/celeba-4 \
    --results_dir celeba-1-4 \
    --bb_model A \
    --sub_model B \
    --fgsm_eps 0.01 \
    --defense_type defense_gan --debug True --debug_dir debug/blackbox-celeba-1-4




sets=(celeba)
for set in ${sets[*]};
do
    echo "Start training on $set."
    for i in $(seq 85 100);
    do
        mkdir -p debug2/gans/$set/$i
        mkdir -p debug2/debug/gans/$set/$i
        mv debug/gans/$set/* debug2/gans/$set/$i/
        mv debug/debug/gans/$set/* debug2/debug/gans/$set/$i/
        echo "Running $i turn of training."
        timeout 2h python train.py --cfg experiments/cfgs/gans/$set.yml --is_train
        echo "Finished $i turn of training."
        sleep 10m;

    done
    i=5
    mkdir -p debug2/gans/$set/$i
    mkdir -p debug2/debug/gans/$set/$i
    mv debug/gans/$set/* debug2/gans/$set/$i/
    mv debug/debug/gans/$set/* debug2/debug/gans/$set/$i/
    
    echo "Training on $set finished."
done

shutdown -h now