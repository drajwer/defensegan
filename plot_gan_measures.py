import glob
import os
from pathlib import Path

dir = os.path.join('results', 'train_and_measure_gan', '0_*.txt')

paths = sorted(glob.iglob(dir), key=os.path.getctime)
paths = [paths[-1]] + paths[:-1]
for filepath in paths:
    with open(filepath) as f:
        #print(filepath)
        print(f.read()[:-1])