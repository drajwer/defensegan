#!/bin/bash
# sets=(mnist fmnist celeba)
# source venv/bin/activate

echo \n\n\n\n\n
echo "Running celebA lbfgs..."
python whitebox.py --cfg output/gans/celeba-long-run \
    --results_dir whitebox-celeba-lbfgs \
    --attack_type lbfgs \
    --defense_type defense_gan --debug True --debug_dir debug/whitebox-celeba-lbfgs

echo \n\n\n\n\n
echo "Running celebA deepfool..."
python whitebox.py --cfg output/gans/celeba-long-run \
    --results_dir whitebox-celeba-deepfool \
    --attack_type deepfool \
    --fgsm_eps 0.1 \
    --defense_type defense_gan --debug True --debug_dir debug/whitebox-celeba-deepfool

echo "Running celebA MIM..."
python whitebox.py --cfg output/gans/celeba-long-run \
    --results_dir whitebox-celeba-mim \
    --attack_type mim \
    --fgsm_eps 0.1 \
    --defense_type defense_gan --debug True --debug_dir debug/whitebox-celeba-mim

echo \n\n\n\n\n
echo "Running mnist lbfgs..."
python whitebox.py --cfg output/gans/mnist \
    --results_dir whitebox-mnist-lbfgs \
    --attack_type lbfgs \
    --fgsm_eps 0.1 --debug True --debug_dir debug/whitebox-mnist-lbfgs

echo \n\n\n\n\n
echo "Running mnist deepfool..."
python whitebox.py --cfg output/gans/mnist \
    --results_dir whitebox-mnist-deepfool \
    --attack_type deepfool \
    --fgsm_eps 0.1 --debug True --debug_dir debug/whitebox-mnist-deepfool

echo \n\n\n\n\n
echo "Running mnist mim..."
python whitebox.py --cfg output/gans/mnist \
    --results_dir whitebox-mnist-mim \
    --attack_type mim \
    --fgsm_eps 0.1 --debug True --debug_dir debug/whitebox-mnist-mim

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




# sets=(celeba)
# for set in ${sets[*]};
# do
#     echo "Start training on $set."
#     for i in $(seq 85 100);
#     do
#         mkdir -p debug2/gans/$set/$i
#         mkdir -p debug2/debug/gans/$set/$i
#         mv debug/gans/$set/* debug2/gans/$set/$i/
#         mv debug/debug/gans/$set/* debug2/debug/gans/$set/$i/
#         echo "Running $i turn of training."
#         timeout 2h python train.py --cfg experiments/cfgs/gans/$set.yml --is_train
#         echo "Finished $i turn of training."
#         sleep 10m;

#     done
#     i=5
#     mkdir -p debug2/gans/$set/$i
#     mkdir -p debug2/debug/gans/$set/$i
#     mv debug/gans/$set/* debug2/gans/$set/$i/
#     mv debug/debug/gans/$set/* debug2/debug/gans/$set/$i/
    
#     echo "Training on $set finished."
# done
