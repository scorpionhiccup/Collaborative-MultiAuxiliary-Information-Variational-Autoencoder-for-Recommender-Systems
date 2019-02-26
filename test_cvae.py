from lib.cvae import *
import numpy as np
import tensorflow as tf
import scipy.io
from scipy import sparse
from lib.utils import *

np.random.seed(0)
tf.set_random_seed(0)
init_logging("cvae.log")

def load_cvae_data():
  data = {}
  data_dir = "data/citeulike-a/"
  link_variables = scipy.io.loadmat("data/citeulike-a/citations.mat")
  link_data = link_variables['X']
  content_variables = scipy.io.loadmat("data/citeulike-a/mult_nor.mat")
  content_data = content_variables['X']
  tfidf_variables = scipy.io.loadmat("data/citeulike-a/item_tag.mat")
  tfidf_data = np.array(tfidf_variables['X'])
  print(type(link_data))
  print(type(content_data))
  print(type(tfidf_data))

  data['content_data'] = content_data
  data['link_data'] = link_data
  data['tfidf_data'] = tfidf_data

  data["train_users"] = load_rating(data_dir + "cf-train-1-users.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users.dat")
  data["test_items"] = load_rating(data_dir + "cf-test-1-items.dat")

  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a[0]==0:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr

params = Params()
params.lambda_u = 0.1
params.lambda_v = 10
params.lambda_r = 1
params.a = 1
params.b = 0.01
params.M = 300
params.n_epochs = 100
params.max_iter = 1

data = load_cvae_data()
num_factors = 50
model = CVAE(num_users=5551, num_items=16980, num_factors=num_factors, params=params, 
    input_dim_content=8000, input_dim_link=16980, input_dim_tfidf=46391, dims=[200, 100], n_z=num_factors, activations=['sigmoid', 'sigmoid'],
    loss_type='cross-entropy', lr=0.001, random_seed=0, print_step=10, verbose=False)
model.load_model(weight_path="model/pretrain")
model.run(data["train_users"], data["train_items"], data["test_users"], data["test_items"], data["content_data"], data['link_data'], data['tfidf_data'], params)
model.save_model(weight_path="model/cvae", pmf_path="model/pmf")