import numpy as np
import tensorflow as tf
import scipy.io
import logging
from scipy import sparse
from lib.vae import VariationalAutoEncoder
from lib.utils import *

np.random.seed(0)
tf.set_random_seed(0)
init_logging("vae.log")

logging.info('loading data')

link_variables = scipy.io.loadmat("data/citeulike-a/citations.mat")
link_data = link_variables['X']
# link_data = sparse.lil_matrix(sparse.load_npz('data/citeulike-a/citation.npz'))
idx = np.random.rand(16980) < 0.8
link_train_X = link_data[idx]
link_test_X = link_data[~idx]

content_variables = scipy.io.loadmat("data/citeulike-a/mult_nor.mat")
content_data = content_variables['X']
# idx = np.random.rand(content_data.shape[0]) < 0.8
content_train_X = content_data[idx]
content_test_X = content_data[~idx]

tfidf_variables = scipy.io.loadmat("data/citeulike-a/item_tag.mat")
tfidf_data = tfidf_variables['X']
tfidf_train_X = tfidf_data[idx]
tfidf_test_X = tfidf_data[~idx]

logging.info('initializing sdae model')
model = VariationalAutoEncoder(input_dim_content=8000, input_dim_link=16980, input_dim_tfidf=46391, dims=[200, 100], z_dim=50,
	activations=['sigmoid','sigmoid'], epoch=[50, 50],
	noise='mask-0.3' ,loss='cross-entropy', lr=0.01, batch_size=128, print_step=1)
logging.info('fitting data starts...')
model.fit(content_train_X, link_train_X, tfidf_train_X, content_test_X, link_test_X, tfidf_test_X)
# feat = model.transform(data)
# scipy.io.savemat('feat-dae.mat',{'feat': feat})
# np.savez("sdae-weights.npz", en_weights=model.weights, en_biases=model.biases,
# 	de_weights=model.de_weights, de_biases=model.de_biases)