#!/bin/bash
sets=(celeba-long mnist-long fmnist-long)
# source venv/bin/activate

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

for set in ${sets[*]};
do
    echo "Start training on $set."
    python train.py --cfg experiments/cfgs/gans/$set.yml --is_train
    echo "Training on $set finished."
done
