#!/bin/bash
# sets=(mnist fmnist celeba)
sets=(celeba)
for set in ${sets[*]};
do
    echo "Start training on $set."
    #for i in $(seq 1 3);
    #do
        # echo "Running $i turn of training."
        # timeout 2h 
    python train.py --cfg experiments/cfgs/gans/$set.yml --is_train
        #echo "Finished $i turn of training."
        #sleep 20m;

    #done
    #echo "Training on $set finished."
done

shutdown -h now