#!/bin/bash
sets=(mnist)
attacks=(bpda1 bpda2 bpda3)
defenses=(defense_gan)
epses=(0.3)
# Missing attacks
#sets=(mnist fmnist)
#attacks=(spsa)

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
            --results_dir eval-$dataset-$attack-$eps-$defense \
            --bb_model A \
            --fgsm_eps $eps \
            --defense_type $defense \
            --attack_type $attack
        done
        done
    done
done

