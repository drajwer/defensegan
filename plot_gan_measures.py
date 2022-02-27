import glob
import os
import sys
from pathlib import Path

dataset = sys.argv[1] if len(sys.argv) == 2 else "mnist"
dir = os.path.join('results', 'train_and_measure_gan_%s' % dataset, '0_*.txt')

paths = sorted(glob.iglob(dir), key=os.path.getctime)
#paths = [paths[-1]] + paths[:-1]

print("IS_GEN_MEAN, IS_GEN_SD, IS_REAL_MEAN, IS_REAL_SD, FID, DATASET")

for filepath in paths:
    with open(filepath) as f:
        content = f.read()[:-1]
        values = tuple(content.split()) + (dataset,)

        print("%s, %s, %s, %s, %s, %s" % values)