import numpy as np
import lib.utils as utils
import tensorflow as tf
import sys
import math
import scipy
import scipy.io
import logging

class Params:
    """Parameters for DMF
    """
    def __init__(self):
        self.a = 1
        self.b = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 10
        self.lambda_r = 1
        self.max_iter = 10
        self.M = 300

        # for updating W and b
        self.lr = 0.001
        self.batch_size = 128
        self.n_epochs = 10

class CVAE:
    def __init__(self, num_users, num_items, num_factors, params, input_dim_content, input_dim_link, input_dim_tfidf, dims, activations, n_z=50, loss_type='cross-entropy', lr=0.1,
        wd=1e-4, dropout=0.1, random_seed=0, print_step=50, verbose=True):
        self.m_num_users = num_users
        self.m_num_items = num_items
        self.m_num_factors = num_factors

        self.m_U = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)

        self.input_dim_content = input_dim_content
        self.input_dim_link = input_dim_link
        self.input_dim_tfidf = input_dim_tfidf
        self.dims = dims
        self.activations = activations
        self.lr = lr
        self.params = params
        self.print_step = print_step
        self.verbose = verbose
        self.loss_type = loss_type
        self.n_z = n_z
        self.weights = []
        self.reg_loss = 0
        self.kongzhi_1 = 0.7
        self.kongzhi_2 = 0.1
        self.belta = 0.5

        self.content_x = tf.placeholder(tf.float32, [None, self.input_dim_content], name='content_x')
        self.link_x = tf.placeholder(tf.float32, [None, self.input_dim_link], name='link_x')
        self.tfidf_x = tf.placeholder(tf.float32, [None, self.input_dim_tfidf], name='tfidf_x')
        self.v = tf.placeholder(tf.float32, [None, self.m_num_factors])

        x_recon_content, x_recon_link, x_recon_tfidf = self.inference_generation(self.content_x, self.link_x, self.tfidf_x)

        # loss
        # reconstruction loss
        if loss_type == 'rmse':
            self.gen_loss = tf.reduce_mean(tf.square(tf.sub(self.x, x_recon)))
        elif loss_type == 'cross-entropy':
            x_recon_content = tf.nn.sigmoid(x_recon_content, name='x_recon_content')
            self.gen_loss_content = -tf.reduce_mean(tf.reduce_sum(self.content_x * tf.log(tf.maximum(x_recon_content, 1e-10))
                + (1-self.content_x) * tf.log(tf.maximum(1 - x_recon_content, 1e-10)),1))
            x_recon_link = tf.nn.sigmoid(x_recon_link, name='x_recon_link')
            self.gen_loss_link = -tf.reduce_mean(tf.reduce_sum(self.link_x * tf.log(tf.maximum(x_recon_link, 1e-10))
                + (1 - self.link_x) * tf.log(tf.maximum(1 - x_recon_link, 1e-10)), 1))
            x_recon_tfidf = tf.nn.sigmoid(x_recon_tfidf, name='x_recon_tfidf')
            self.gen_loss_tfidf = -tf.reduce_mean(tf.reduce_sum(self.tfidf_x * tf.log(tf.maximum(x_recon_tfidf, 1e-10))
                                                               + (1 - self.tfidf_x) * tf.log(tf.maximum(1 - x_recon_tfidf, 1e-10)), 1))

        self.latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq) - self.z_log_sigma_sq - 1, 1))
        self.v_loss = 1.0*params.lambda_v/params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v - self.z), 1))

        self.loss = self.gen_loss_content + self.gen_loss_link + self.gen_loss_tfidf + self.belta*self.latent_loss + self.v_loss + 2e-4*self.reg_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Initializing the tensor flow variables
        self.saver = tf.train.Saver(self.weights)
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def inference_generation(self, content_x, link_x, tfidf_x):
        with tf.variable_scope("inference"):
            # biases 统统以0开头，
            # tf.contrib.layers.xavier_initializer()：这个初始化器是用来保持每一层的梯度大小都差不多相同。
            # 后面跟VAE 中差不多
            rec = {'W1_content': tf.get_variable("W1_content", [self.input_dim_content, self.dims[0]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b1_content': tf.get_variable("b1_content", [self.dims[0]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W2_content': tf.get_variable("W2_content", [self.dims[0], self.dims[1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2_content': tf.get_variable("b2_content", [self.dims[1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W1_link': tf.get_variable("W1_link", [self.input_dim_link, self.dims[0]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b1_link': tf.get_variable("b1_link", [self.dims[0]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W2_link': tf.get_variable("W2_link", [self.dims[0], self.dims[1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2_link': tf.get_variable("b2_link", [self.dims[1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W1_tfidf': tf.get_variable("W1_tfidf", [self.input_dim_tfidf, self.dims[0]],
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b1_tfidf': tf.get_variable("b1_tfidf", [self.dims[0]], initializer=tf.constant_initializer(0.0),
                                              dtype=tf.float32),
                'W2_tfidf': tf.get_variable("W2_tfidf", [self.dims[0], self.dims[1]],
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2_tfidf': tf.get_variable("b2_tfidf", [self.dims[1]], initializer=tf.constant_initializer(0.0),
                                              dtype=tf.float32),
                'W_z_mean_content': tf.get_variable("W_z_mean_content", [self.dims[1], self.n_z], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_mean_content': tf.get_variable("b_z_mean_content", [self.n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma_content': tf.get_variable("W_z_log_sigma_content", [self.dims[1], self.n_z], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_log_sigma_content': tf.get_variable("b_z_log_sigma_content", [self.n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_mean_link': tf.get_variable("W_z_mean_link", [self.dims[1], self.n_z], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_mean_link': tf.get_variable("b_z_mean_link", [self.n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma_link': tf.get_variable("W_z_log_sigma_link", [self.dims[1], self.n_z], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_log_sigma_link': tf.get_variable("b_z_log_sigma_link", [self.n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_mean_tfidf': tf.get_variable("W_z_mean_tfidf", [self.dims[1], self.n_z],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    dtype=tf.float32),
                'b_z_mean_tfidf': tf.get_variable("b_z_mean_tfidf", [self.n_z],
                                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma_tfidf': tf.get_variable("W_z_log_sigma_tfidf", [self.dims[1], self.n_z],
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         dtype=tf.float32),
                'b_z_log_sigma_tfidf': tf.get_variable("b_z_log_sigma_tfidf", [self.n_z],
                                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32)}

        with tf.variable_scope("generation"):
            gen = {'W2_content': tf.get_variable("W2_content", [self.n_z, self.dims[1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2_content': tf.get_variable("b2_content", [self.dims[1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W2_link': tf.get_variable("W2_link", [self.n_z, self.dims[1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2_link': tf.get_variable("b2_link", [self.dims[1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W2_tfidf': tf.get_variable("W2_tfidf", [self.n_z, self.dims[1]],
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2_tfidf': tf.get_variable("b2_tfidf", [self.dims[1]], initializer=tf.constant_initializer(0.0),
                                              dtype=tf.float32),
                'W1_content': tf.transpose(rec['W2_content']),
                'b1_content': rec['b1_content'],
                'W1_link': tf.transpose(rec['W2_link']),
                'b1_link': rec['b1_link'],
                'W1_tfidf': tf.transpose(rec['W2_tfidf']),
                'b1_tfidf': rec['b1_tfidf'],
                'W_x_content': tf.transpose(rec['W1_content']),
                'b_x_content': tf.get_variable("b_x_content", [self.input_dim_content], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_x_link': tf.transpose(rec['W1_link']),
                'b_x_link': tf.get_variable("b_x_link", [self.input_dim_link], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_x_tfidf': tf.transpose(rec['W1_tfidf']),
                'b_x_tfidf': tf.get_variable("b_x_tfidf", [self.input_dim_tfidf], initializer=tf.constant_initializer(0.0), dtype=tf.float32)}

        self.weights += [rec['W1_content'], rec['b1_content'], rec['W2_content'], rec['b2_content'],
                         rec['W1_link'], rec['b1_link'], rec['W2_link'], rec['b2_link'],
                         rec['W1_tfidf'], rec['b1_tfidf'], rec['W2_tfidf'], rec['b2_tfidf'],
                         rec['W_z_mean_content'], rec['b_z_mean_content'], rec['W_z_log_sigma_content'], rec['b_z_log_sigma_content'],
                         rec['W_z_mean_link'], rec['b_z_mean_link'], rec['W_z_log_sigma_link'], rec['b_z_log_sigma_link'],
                         rec['W_z_mean_tfidf'], rec['b_z_mean_tfidf'], rec['W_z_log_sigma_tfidf'], rec['b_z_log_sigma_tfidf']]
        self.reg_loss += tf.nn.l2_loss(rec['W1_content']) + tf.nn.l2_loss(rec['W2_content']) + tf.nn.l2_loss(rec['W1_link']) + tf.nn.l2_loss(rec['W2_link']) + tf.nn.l2_loss(rec['W1_tfidf']) + tf.nn.l2_loss(rec['W2_tfidf'])

        h1_content = self.activate(tf.matmul(content_x, rec['W1_content']) + rec['b1_content'], self.activations[0])
        h2_content = self.activate(tf.matmul(h1_content, rec['W2_content']) + rec['b2_content'], self.activations[1])
        self.z_mean_content = tf.matmul(h2_content, rec['W_z_mean_content']) + rec['b_z_mean_content']
        self.z_log_sigma_sq_content = tf.matmul(h2_content, rec['W_z_log_sigma_content']) + rec['b_z_log_sigma_content']

        h1_link = self.activate(tf.matmul(link_x, rec['W1_link']) + rec['b1_link'], self.activations[0])
        h2_link = self.activate(tf.matmul(h1_link, rec['W2_link']) + rec['b2_link'], self.activations[1])
        self.z_mean_link = tf.matmul(h2_link, rec['W_z_mean_link']) + rec['b_z_mean_link']
        self.z_log_sigma_sq_link = tf.matmul(h2_link, rec['W_z_log_sigma_link']) + rec['b_z_log_sigma_link']

        h1_tfidf = self.activate(tf.matmul(tfidf_x, rec['W1_tfidf']) + rec['b1_tfidf'], self.activations[0])
        h2_tfidf = self.activate(tf.matmul(h1_tfidf, rec['W2_tfidf']) + rec['b2_tfidf'], self.activations[1])
        self.z_mean_tfidf = tf.matmul(h2_tfidf, rec['W_z_mean_tfidf']) + rec['b_z_mean_tfidf']
        self.z_log_sigma_sq_tfidf = tf.matmul(h2_tfidf, rec['W_z_log_sigma_tfidf']) + rec['b_z_log_sigma_tfidf']

        self.z_mean = self.kongzhi_1 * self.z_mean_content + self.kongzhi_2 * self.z_mean_link + (1-self.kongzhi_1-self.kongzhi_2) * self.z_mean_tfidf
        self.z_log_sigma_sq = self.kongzhi_1 * self.z_log_sigma_sq_content + self.kongzhi_2 * self.z_log_sigma_sq_link + (1-self.kongzhi_1-self.kongzhi_2) * self.z_log_sigma_sq_tfidf

        eps = tf.random_normal((self.params.batch_size, self.n_z), 0, 1, seed=0, dtype=tf.float32)
        self.z = self.z_mean + tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)) * eps

        self.weights += [gen['W2_content'], gen['b2_content'], gen['b_x_content'],
                         gen['W2_link'], gen['b2_link'], gen['b_x_link'],
                         gen['W2_tfidf'], gen['b2_tfidf'], gen['b_x_tfidf']]
        self.reg_loss += tf.nn.l2_loss(gen['W1_content']) + tf.nn.l2_loss(gen['W_x_content']) + tf.nn.l2_loss(gen['W1_link']) + tf.nn.l2_loss(gen['W_x_link']) + tf.nn.l2_loss(gen['W1_tfidf']) + tf.nn.l2_loss(gen['W_x_tfidf'])

        h2_content = self.activate(tf.matmul(self.z, gen['W2_content']) + gen['b2_content'], self.activations[1])
        h1_content = self.activate(tf.matmul(h2_content, gen['W1_content']) + gen['b1_content'], self.activations[0])
        x_recon_content = tf.matmul(h1_content, gen['W_x_content']) + gen['b_x_content']

        h2_link = self.activate(tf.matmul(self.z, gen['W2_link']) + gen['b2_link'], self.activations[1])
        h1_link = self.activate(tf.matmul(h2_link, gen['W1_link']) + gen['b1_link'], self.activations[0])
        x_recon_link = tf.matmul(h1_link, gen['W_x_link']) + gen['b_x_link']

        h2_tfidf = self.activate(tf.matmul(self.z, gen['W2_tfidf']) + gen['b2_tfidf'], self.activations[1])
        h1_tfidf = self.activate(tf.matmul(h2_tfidf, gen['W1_tfidf']) + gen['b1_tfidf'], self.activations[0])
        x_recon_tfidf = tf.matmul(h1_tfidf, gen['W_x_tfidf']) + gen['b_x_tfidf']

        return x_recon_content, x_recon_link, x_recon_tfidf

    def cdl_estimate(self, content_x, link_x, tfidf_x, num_iter):
        for i in range(num_iter):
            b_x_content, ids_content = utils.get_batch(content_x, self.params.batch_size)
            b_x_link = link_x[ids_content, :]
            b_x_tfidf = tfidf_x[ids_content, :]
            _, l, gen_loss_content, gen_loss_link, gen_loss_tfidf, v_loss = self.sess.run((self.optimizer, self.loss, self.gen_loss_content, self.gen_loss_link, self.gen_loss_tfidf, self.v_loss),
             feed_dict={self.content_x: b_x_content, self.link_x:b_x_link, self.tfidf_x:b_x_tfidf, self.v: self.m_V[ids_content, :]})
            # Display logs per epoch step
            if i % self.print_step == 0 and self.verbose:
                print ("Iter:", '%04d' % (i+1), "loss=", "{:.5f}".format(l), "genloss_content=", "{:.5f}".format(gen_loss_content), "genloss_link=", "{:.5f}".format(gen_loss_link), "genloss_tfidf=", "{:.5f}".format(gen_loss_tfidf), "vloss=", "{:.5f}".format(v_loss))
        return gen_loss_content, gen_loss_link, gen_loss_tfidf

    def transform(self, content_data, link_data, tfidf_data):
        data_en = self.sess.run(self.z_mean, feed_dict={self.content_x:content_data, self.link_x:link_data, self.tfidf_x:tfidf_data})
        return data_en

    def pmf_estimate(self, users, items, test_users, test_items, params):
        """
        users: list of list
        """
        min_iter = 1
        a_minus_b = params.a - params.b
        converge = 1.0
        likelihood_old = 0.0
        likelihood = -math.exp(20)
        it = 0
        while ((it < params.max_iter and converge > 1e-6) or it < min_iter):
            likelihood_old = likelihood
            likelihood = 0
            # update U
            # VV^T for v_j that has at least one user liked
            ids = np.array([len(x) for x in items]) > 0
            v = self.m_V[ids]
            VVT = np.dot(v.T, v)
            XX = VVT * params.b + np.eye(self.m_num_factors) * params.lambda_u

            for i in range(self.m_num_users):
                item_ids = users[i]
                n = len(item_ids)
                if n > 0:
                    A = np.copy(XX)
                    A += np.dot(self.m_V[item_ids, :].T, self.m_V[item_ids,:])*a_minus_b
                    x = params.a * np.sum(self.m_V[item_ids, :], axis=0)
                    self.m_U[i, :] = scipy.linalg.solve(A, x)
                    
                    likelihood += -0.5 * params.lambda_u * np.sum(self.m_U[i]*self.m_U[i])

            # update V
            ids = np.array([len(x) for x in users]) > 0
            u = self.m_U[ids]
            XX = np.dot(u.T, u) * params.b
            for j in range(self.m_num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m>0 :
                    A = np.copy(XX)
                    A += np.dot(self.m_U[user_ids,:].T, self.m_U[user_ids,:])*a_minus_b
                    B = np.copy(A)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.a * np.sum(self.m_U[user_ids, :], axis=0) + params.lambda_v * self.m_theta[j,:]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)
                    
                    likelihood += -0.5 * m * params.a
                    likelihood += params.a * np.sum(np.dot(self.m_U[user_ids, :], self.m_V[j,:][:, np.newaxis]),axis=0)
                    likelihood += -0.5 * self.m_V[j,:].dot(B).dot(self.m_V[j,:][:,np.newaxis])

                    ep = self.m_V[j,:] - self.m_theta[j,:]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep) 
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.lambda_v * self.m_theta[j,:]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)
                    
                    ep = self.m_V[j,:] - self.m_theta[j,:]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep)
            # computing negative log likelihood
            #likelihood += -0.5 * params.lambda_u * np.sum(self.m_U * self.m_U)
            #likelihood += -0.5 * params.lambda_v * np.sum(self.m_V * self.m_V)
            # split R_ij into 0 and 1
            # -sum(0.5*C_ij*(R_ij - u_i^T * v_j)^2) = -sum_ij 1(R_ij=1) 0.5*C_ij +
            #  sum_ij 1(R_ij=1) C_ij*u_i^T * v_j - 0.5 * sum_j v_j^T * U C_i U^T * v_j
            
            it += 1
            converge = abs(1.0*(likelihood - likelihood_old)/likelihood_old)

            if self.verbose:
                if likelihood < likelihood_old:
                    print("likelihood is decreasing!")

                print("[iter=%04d], likelihood=%.5f, converge=%.10f" % (it, likelihood, converge))

        return likelihood

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def run(self, users, items, test_users, test_items, content_data, link_data, tfidf_data, params):
        self.m_theta[:] = self.transform(content_data, link_data, tfidf_data)
        self.m_V[:] = self.m_theta
        n = content_data.shape[0]
        for epoch in range(params.n_epochs):
            num_iter = int(n / params.batch_size)
            # gen_loss = self.cdl_estimate(data_x, params.cdl_max_iter)
            gen_loss_content, gen_loss_link, gen_loss_tfidf = self.cdl_estimate(content_data, link_data, tfidf_data, num_iter)
            self.m_theta[:] = self.transform(content_data, link_data, tfidf_data)
            likelihood = self.pmf_estimate(users, items, test_users, test_items, params)
            loss = -likelihood + 0.5 * (gen_loss_content + gen_loss_link + gen_loss_tfidf) * n * params.lambda_r
            logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss_content=%.5f, gen_loss_link=%.5f, gen_loss_tfidf=%.5f" % (epoch, loss, -likelihood, gen_loss_content, gen_loss_link, gen_loss_tfidf))

    def save_model(self, weight_path, pmf_path=None):
        self.saver.save(self.sess, weight_path)
        logging.info("Weights saved at " + weight_path)
        if pmf_path is not None:
            scipy.io.savemat(pmf_path,{"m_U": self.m_U, "m_V": self.m_V, "m_theta": self.m_theta})
            logging.info("Weights saved at " + pmf_path)

    def load_model(self, weight_path, pmf_path=None):
        logging.info("Loading weights from " + weight_path)
        self.saver.restore(self.sess, weight_path)
        if pmf_path is not None:
            logging.info("Loading pmf data from " + pmf_path)
            data = scipy.io.loadmat(pmf_path)
            self.m_U[:] = data["m_U"]
            self.m_V[:] = data["m_V"]
            self.m_theta[:] = data["m_theta"]

