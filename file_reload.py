import glob
from pathlib2 import Path
import os
from os import listdir


root = r'D:\repository\pycharm\datasets\imageClef'

for domain in listdir(root):
    lines = []
    for clz in listdir(os.path.join(root, domain)):
        lines += [domain + os.sep + clz + os.sep + file + ' ' + clz for file in listdir(os.path.join(root, domain, clz))]

    f = open(domain + '.txt', "w+")
    for line in lines:
        f.write(line + '\n')
    f.close()
