#!/bin/bash
model=$1
sets=(mnist fmnist celeba)
attacks=(cw-trim fgsm pgd mim bim cw) 
#defenses=(adv_tr)
defense=adv_tr
epses=(0.05 0.1 0.2 0.3)
adv_tr_epses=(0.05 0.1 0.2 0.3)
dateNow=$(date '+%Y%m%d%H%M')  

echo "Running $model model..."

for dataset in ${sets[*]};
do
    for attack in ${attacks[*]};
    do
        for adv_tr_eps in ${adv_tr_epses[*]};
        do
        for eps in ${epses[*]};
        do
        echo ""
        echo ""
        echo ""
        echo "Running $dataset eval. Attack: $attack eps: $eps, defense: $defense, defense-eps: $adv_tr_eps)..."
        python whitebox.py \
            --cfg output/gans/$dataset-long \
            --results_dir eval-$dateNow-$dataset-$attack-$eps-$defense-$adv_tr_eps-$model \
            --model $model \
            --fgsm_eps $eps \
            --fgsm_eps_tr $adv_tr_eps \
            --defense_type $defense \
            --attack_type $attack
        done
        done
    done
done

