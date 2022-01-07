#!/bin/bash

sets=(celeba)
attacks=(bpda-fgsm bpda-pgd bpda-mim bpda-bim fgsm pgd mim cw)
defenses=(defense_gan)
epses=(0.05 0.1 0.2 0.3)
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
        echo "Running $dataset eval. Attack: $attack eps: $eps, defense: $defense)..."
        python whitebox.py \
            --cfg output/gans/$dataset-long \
            --results_dir eval-$dateNow-$dataset-$attack-$eps-$defense \
            --bb_model A \
            --fgsm_eps $eps \
            --defense_type $defense \
            --attack_type $attack
        done
        done
    done
done

