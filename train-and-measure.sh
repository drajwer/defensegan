#!/bin/bash
# sets=(mnist fmnist celeba)
source venv/bin/activate

# python whitebox.py --cfg output/gans/celeba-1 \
#     --results_dir whitebox-celeba-fgsm \
#     --attack_type fgsm \
#     --fgsm_eps 0.1 \
#     --defense_type defense_gan --debug True --debug_dir debug/whitebox-celeba-fgsm

# python whitebox.py --cfg output/gans/celeba-1 \
#     --results_dir whitebox-celeba-rand_fgsm \
#     --attack_type rand_fgsm \
#     --fgsm_eps 0.1 \
#     --defense_type defense_gan --debug True --debug_dir debug/whitebox-celeba-rand_fgsm

# python whitebox.py --cfg output/gans/celeba-1 \
#     --results_dir whitebox-celeba-cw \
#     --attack_type cw \
#     --defense_type defense_gan --debug True --debug_dir debug/whitebox-celeba-cw

# python whitebox.py --cfg output/gans/mnist \
#     --results_dir whitebox-mnist-fgsm \
#     --attack_type fgsm \
#     --fgsm_eps 0.1 \
#     --defense_type defense_gan --debug True --debug_dir debug/whitebox-mnist-fgsm

# python whitebox.py --cfg output/gans/mnist \
#     --results_dir whitebox-mnist-rand_fgsm \
#     --attack_type rand_fgsm \
#     --fgsm_eps 0.1 \
#     --defense_type defense_gan --debug True --debug_dir debug/whitebox-mnist-rand_fgsm

# python whitebox.py --cfg output/gans/mnist \
#     --results_dir whitebox-mnist-cw \
#     --attack_type cw \
#     --defense_type defense_gan --debug True --debug_dir debug/whitebox-mnist-cw

# python blackbox.py --cfg output/gans/celeba-2 \
#     --results_dir celeba-1-11 \
#     --bb_model A \
#     --sub_model B \
#     --fgsm_eps 0.0001 \
#     --defense_type defense_gan --debug True --debug_dir debug/blackbox-celeba-1-11


# python blackbox.py --cfg output/gans/celeba-3 \
#     --results_dir celeba-1-12 \
#     --bb_model A \
#     --sub_model B \
#     --fgsm_eps 0.001 \
#     --defense_type defense_gan --debug True --debug_dir debug/blackbox-celeba-1-12

# python blackbox.py --cfg output/gans/celeba-4 \
#     --results_dir celeba-1-13 \
#     --bb_model A \
#     --sub_model B \
#     --fgsm_eps 0.01 \
#     --defense_type defense_gan --debug True --debug_dir debug/blackbox-celeba-1-13




sets=(celeba, fmnist, mnist)
for set in ${sets[*]};
do
    echo "Start training on $set."
    for i in $(seq 1 100);
    do
        mkdir -p debug3/gans/$set/$i
        mkdir -p debug3/debug/gans/$set/$i
        mv debug/gans/$set/* debug3/gans/$set/$i/
        mv debug/debug/gans/$set/* debug3/debug/gans/$set/$i/
        echo "Running $i turn of training."
        timeout 30m python train.py --cfg experiments/cfgs/gans/$set-measure_gan.yml --is_train
        python measure_gan.py --cfg output/gans/$set-measure --results_dir train_and_measure_gan --probe_size 20000 --iter $i
        echo "Finished $i turn of training."

    done
    i=100
    mkdir -p debug3/gans/$set/$i
    mkdir -p debug3/debug/gans/$set/$i
    mv debug/gans/$set/* debug3/gans/$set/$i/
    mv debug/debug/gans/$set/* debug3/debug/gans/$set/$i/
    
    echo "Training on $set finished."
done
