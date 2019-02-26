#coding=utf-8
import numpy as np
import scipy.io as scio

userss = scio.loadmat('pmf.mat')['m_U']
itemss = np.transpose(scio.loadmat('pmf.mat')['m_V'][:1000])
print(np.shape(itemss))
user_item = np.dot(userss * itemss)
print(np.shape(user_item))