#!/bin/bash
model=$1
sets=(mnist fmnist celeba)
attacks=(pgd mim bim cw fgsm)# deepfool lbfgs) # bpda-fgsm bpda-pgd bpda-mim bpda-bim 
defenses=(defense_gan)
epses=(0.05 0.1 0.2 0.3)
dateNow=$(date '+%Y%m%d%H%M')  

echo "Running $model model..."

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
            --results_dir eval-$dateNow-$dataset-$attack-$eps-$defense-$model \
            --model $model \
            --fgsm_eps $eps \
            --defense_type $defense \
            --attack_type $attack
        done
        done
    done
done

