import numpy as np
import scipy.io as scio

citations_matrix = np.zeros([16980, 16980], dtype=np.int8)
files = open('citations.dat')
i = 0
for line in files:
    line = line.strip().split(' ')
    for l in line[1:]:
        citations_matrix[i,int(l)] = 1
    i += 1

scio.savemat('citations.mat',{'X':citations_matrix})