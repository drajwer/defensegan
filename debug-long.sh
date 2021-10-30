#!/bin/bash
sets=(mnist) # fmnist celeba)
#sets=(fmnist celeba)
attacks=(bpda-fgsm) # bpda-pgd bpda-mim bpda-bim fgsm pgd mim)
defenses=(defense_gan)
epses=(0.1) # 0.05 0.2 0.3)
# Missing attacks
#sets=(mnist fmnist)
#attacks=(spsa)
dateNow=$(date '+%Y%m%d%H%M')  


for dataset in ${sets[*]};
do
    for attack in ${attacks[*]};
    do
        for defense in ${defenses[*]};
        do
        for eps in ${epses[*]};
        do
        echo ""
        echo ""
        echo ""
        echo "Running $dataset eval. Attack: $attack eps: $eps, defense: $defense..."
        python whitebox.py \
            --cfg output/gans/$dataset-long \
            --results_dir debug$dateNow-$dataset-$attack-$eps-$defense \
            --bb_model A \
            --fgsm_eps $eps \
            --defense_type $defense \
            --attack_type $attack \
            --nb_epochs 1 \
            --nb_epochs_s 1 \
            --data_aug 1 

            #--debug False \
            #--debug_dir debug/debug$dateNow-$dataset-$attack-$eps-$defense \
        done
        done
    done
done

