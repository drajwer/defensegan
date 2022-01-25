import glob
import os
import sys
from pathlib import Path

dir = sys.argv[1]
print_header = len(sys.argv) >= 3 and sys.argv[2] == 'True' 
full_path = os.path.join('results', dir, 'eval-*', '0_*.txt')

paths = sorted(glob.iglob(full_path), key=os.path.getctime)
paths = [paths[-1]] + paths[:-1]

if print_header:
    print("dataset, model, attack, eps, defense, date, acc")

for filepath in paths:
    with open(filepath) as f:
        acc = f.read()[:-1].split()[0]

        dir = os.path.dirname(filepath).split('/')[-1]
        splitted_dir = dir.split('-')[1:]
        if len(splitted_dir) <= 6:
            date, dataset, attack, eps, defense, model = tuple(splitted_dir)
        else:
            date, dataset, bpda, attack, eps, defense, model = tuple(splitted_dir)
            attack = "%s_%s" % (bpda, attack)
        date = "%s-%s-%s %s:%s" % (date[0:4], date[4:6], date[6:8], date[8:10], date[10:12])
        print("%s, %s, %s, %s, %s, %s, %s" % (dataset, model, attack, eps, defense, date, acc)) 
