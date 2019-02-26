import numpy as np
import scipy.io as scio

tag_matrix = np.zeros([16980, 46391], dtype=np.int8)
files = open('item-tag.dat')
i = 0
for line in files:
    line = line.strip().split(' ')
    for l in line[1:]:
        citations_matrix[i,int(l)] = 1
    i += 1

scio.savemat('item_tag.mat',{'X':citations_matrix})