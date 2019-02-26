import numpy as np
import lib.utils as utils
import tensorflow as tf
import logging

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']
def xavier_init(fan_in, fan_out, dtype=tf.float32, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class VariationalAutoEncoder:
    """A deep variational autoencoder"""
    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
        assert utils.noise_validator(self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, input_dim_content, input_dim_link, input_dim_tfidf, dims, z_dim, activations, epoch=1000, noise=None, loss='cross-entropy',
                 lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.n_z = z_dim
        self.input_dim_content = input_dim_content
        self.input_dim_link = input_dim_link
        self.input_dim_tfidf = input_dim_tfidf
        self.weights, self.biases = [], []
        self.de_weights, self.de_biases = [], []
        self.kongzhi_1 = 0.7
        self.kongzhi_2 = 0.1
        self.belta = 0.5

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), int(round(
                    frac * len(i))), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, data_content, data_link, data_tfidf, content_valid=None, link_valid=None, tfidf_valid=None):
        content_x = data_content
        link_x = data_link
        tfidf_x = data_tfidf
        for i in range(self.depth):
            logging.info('Layer {0}'.format(i + 1))
            [content_x, link_x, tfidf_x] = self.run_content_link_tfidf(data_content=content_x, data_link=link_x, data_tfidf=tfidf_x,
                          activation=self.activations[i],hidden_dim=self.dims[i],epoch=self.epoch[i],loss=self.loss,
                          batch_size=self.batch_size,lr=self.lr,print_step=self.print_step)
        # fit latent layer
        self.run_latent_content_link_tfidf(data_content=content_x, data_link=link_x, data_tfidf=tfidf_x, hidden_dim=self.n_z, batch_size=self.batch_size,
            lr=self.lr, epoch=50, print_step=self.print_step)
        self.run_all_content_link_tfidf(data_content=data_content, data_link=data_link, data_tfidf=data_tfidf, lr=self.lr, batch_size=self.batch_size,
            epoch=100, print_step=self.print_step, content_valid=content_valid, link_valid=link_valid, tfidf_valid=tfidf_valid)

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        # x = tf.constant(data, dtype=tf.float32)
        input_dim = len(data[0])
        data_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x = data_x
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return sess.run(x, feed_dict={data_x: data})
        # return x.eval(session=sess)


    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def run_all_content_link_tfidf(self, data_content, data_link, data_tfidf, lr, batch_size, epoch, print_step=100, content_valid=None, link_valid=None, tfidf_valid=None):
        tf.reset_default_graph()
        n = data_content.shape[0]
        input_dim_content = len(data_content[0])
        input_dim_link = len(data_link[0])
        input_dim_tfidf = len(data_tfidf[0])
        num_iter = int(n / batch_size)
        with tf.variable_scope("inference"):
            rec = {'W1_content': tf.get_variable("W1_content", initializer=tf.constant(self.weights[0]), dtype=tf.float32),
                'b1_content': tf.get_variable("b1_content", initializer=tf.constant(self.biases[0]), dtype=tf.float32),
                'W2_content': tf.get_variable("W2_content", initializer=tf.constant(self.weights[3]), dtype=tf.float32),
                'b2_content': tf.get_variable("b2_content", initializer=tf.constant(self.biases[3]), dtype=tf.float32),
                'W1_link': tf.get_variable("W1_link", initializer=tf.constant(self.weights[1]), dtype=tf.float32),
                'b1_link': tf.get_variable("b1_link", initializer=tf.constant(self.biases[1]), dtype=tf.float32),
                'W2_link': tf.get_variable("W2_link", initializer=tf.constant(self.weights[4]), dtype=tf.float32),
                'b2_link': tf.get_variable("b2_link", initializer=tf.constant(self.biases[4]), dtype=tf.float32),
                'W1_tfidf': tf.get_variable("W1_tfidf", initializer=tf.constant(self.weights[2]), dtype=tf.float32),
                'b1_tfidf': tf.get_variable("b1_tfidf", initializer=tf.constant(self.biases[2]), dtype=tf.float32),
                'W2_tfidf': tf.get_variable("W2_tfidf", initializer=tf.constant(self.weights[5]), dtype=tf.float32),
                'b2_tfidf': tf.get_variable("b2_tfidf", initializer=tf.constant(self.biases[5]), dtype=tf.float32),
                'W_z_mean_content': tf.get_variable("W_z_mean_content", initializer=tf.constant(self.weights[6]), dtype=tf.float32),
                'b_z_mean_content': tf.get_variable("b_z_mean_content", initializer=tf.constant(self.biases[6]), dtype=tf.float32),
                'W_z_log_sigma_content': tf.get_variable("W_z_log_sigma_content", initializer=tf.constant(self.weights[9]), dtype=tf.float32),
                'b_z_log_sigma_content': tf.get_variable("b_z_log_sigma_content", initializer=tf.constant(self.biases[9]), dtype=tf.float32),
                'W_z_mean_link': tf.get_variable("W_z_mean_link", initializer=tf.constant(self.weights[7]), dtype=tf.float32),
                'b_z_mean_link': tf.get_variable("b_z_mean_link", initializer=tf.constant(self.biases[7]), dtype=tf.float32),
                'W_z_log_sigma_link': tf.get_variable("W_z_log_sigma_link", initializer=tf.constant(self.weights[10]), dtype=tf.float32),
                'b_z_log_sigma_link': tf.get_variable("b_z_log_sigma_link", initializer=tf.constant(self.biases[10]), dtype=tf.float32),
                'W_z_mean_tfidf': tf.get_variable("W_z_mean_tfidf", initializer=tf.constant(self.weights[8]), dtype=tf.float32),
                'b_z_mean_tfidf': tf.get_variable("b_z_mean_tfidf", initializer=tf.constant(self.biases[8]), dtype=tf.float32),
                'W_z_log_sigma_tfidf': tf.get_variable("W_z_log_sigma_tfidf", initializer=tf.constant(self.weights[11]), dtype=tf.float32),
                'b_z_log_sigma_tfidf': tf.get_variable("b_z_log_sigma_tfidf", initializer=tf.constant(self.biases[11]), dtype=tf.float32)}

        with tf.variable_scope("generation"):
            gen = {'W2_content': tf.get_variable("W2_content", initializer=tf.constant(self.de_weights[6]), dtype=tf.float32),
                'b2_content': tf.get_variable("b2_content", initializer=tf.constant(self.de_biases[6]), dtype=tf.float32),
                'W2_link': tf.get_variable("W2_link", initializer=tf.constant(self.de_weights[7]), dtype=tf.float32),
                'b2_link': tf.get_variable("b2_link", initializer=tf.constant(self.de_biases[7]), dtype=tf.float32),
                'W2_tfidf': tf.get_variable("W2_tfidf", initializer=tf.constant(self.de_weights[8]), dtype=tf.float32),
                'b2_tfidf': tf.get_variable("b2_tfidf", initializer=tf.constant(self.de_biases[8]), dtype=tf.float32),
                'W1_content': tf.transpose(rec['W2_content']),
                'b1_content': rec['b1_content'],
                'W1_link': tf.transpose(rec['W2_link']),
                'b1_link': rec['b1_link'],
                'W1_tfidf': tf.transpose(rec['W2_tfidf']),
                'b1_tfidf': rec['b1_tfidf'],
                'W_x_content': tf.transpose(rec['W1_content']),
                'b_x_content': tf.get_variable("b_x_content", [input_dim_content], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_x_link': tf.transpose(rec['W1_link']),
                'b_x_link': tf.get_variable("b_x_link", [input_dim_link], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_x_tfidf': tf.transpose(rec['W1_tfidf']),
                'b_x_tfidf': tf.get_variable("b_x_tfidf", [input_dim_tfidf], initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        weights = []
        weights += [rec['W1_content'], rec['b1_content'], rec['W2_content'], rec['b2_content'],
                    rec['W1_link'], rec['b1_link'], rec['W2_link'], rec['b2_link'],
                    rec['W1_tfidf'], rec['b1_tfidf'], rec['W2_tfidf'], rec['b2_tfidf'],
                    rec['W_z_mean_content'], rec['b_z_mean_content'], rec['W_z_log_sigma_content'], rec['b_z_log_sigma_content'],
                    rec['W_z_mean_link'], rec['b_z_mean_link'], rec['W_z_log_sigma_link'], rec['b_z_log_sigma_link'],
                    rec['W_z_mean_tfidf'], rec['b_z_mean_tfidf'], rec['W_z_log_sigma_tfidf'], rec['b_z_log_sigma_tfidf']]
        weights += [gen['W2_content'], gen['b2_content'], gen['b_x_content'],
                    gen['W2_link'], gen['b2_link'], gen['b_x_link'],
                    gen['W2_tfidf'], gen['b2_tfidf'], gen['b_x_tfidf']]
        saver = tf.train.Saver(weights)

        content_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_content], name='content_x')
        link_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_link], name='link_x')
        tfidf_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_tfidf], name='tfidf_x')

        h1_content = self.activate(tf.matmul(content_x, rec['W1_content']) + rec['b1_content'], self.activations[0])
        h2_content = self.activate(tf.matmul(h1_content, rec['W2_content']) + rec['b2_content'], self.activations[1])
        z_mean_content = tf.matmul(h2_content, rec['W_z_mean_content']) + rec['b_z_mean_content']
        z_log_sigma_sq_content = tf.matmul(h2_content, rec['W_z_log_sigma_content']) + rec['b_z_log_sigma_content']
        h1_link = self.activate(tf.matmul(link_x, rec['W1_link']) + rec['b1_link'], self.activations[0])
        h2_link = self.activate(tf.matmul(h1_link, rec['W2_link']) + rec['b2_link'], self.activations[1])
        z_mean_link = tf.matmul(h2_link, rec['W_z_mean_link']) + rec['b_z_mean_link']
        z_log_sigma_sq_link = tf.matmul(h2_link, rec['W_z_log_sigma_link']) + rec['b_z_log_sigma_link']
        h1_tfidf = self.activate(tf.matmul(tfidf_x, rec['W1_tfidf']) + rec['b1_tfidf'], self.activations[0])
        h2_tfidf = self.activate(tf.matmul(h1_tfidf, rec['W2_tfidf']) + rec['b2_tfidf'], self.activations[1])
        z_mean_tfidf = tf.matmul(h2_tfidf, rec['W_z_mean_tfidf']) + rec['b_z_mean_tfidf']
        z_log_sigma_sq_tfidf = tf.matmul(h2_tfidf, rec['W_z_log_sigma_tfidf']) + rec['b_z_log_sigma_tfidf']

        z_mean = self.kongzhi_1 * z_mean_content + self.kongzhi_2 * z_mean_link + (1-self.kongzhi_1-self.kongzhi_2) * z_mean_tfidf
        z_log_sigma_sq = self.kongzhi_1 * z_log_sigma_sq_content + self.kongzhi_2 * z_log_sigma_sq_link + (1-self.kongzhi_1-self.kongzhi_2) * z_log_sigma_sq_tfidf

        eps = tf.random_normal((batch_size, self.n_z), 0, 1,dtype=tf.float32)
        z = z_mean + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), 1e-10)) * eps

        h2_content = self.activate(tf.matmul(z, gen['W2_content']) + gen['b2_content'], self.activations[1])
        h1_content = self.activate(tf.matmul(h2_content, gen['W1_content']) + gen['b1_content'], self.activations[0])
        x_recon_content = tf.matmul(h1_content, gen['W_x_content']) + gen['b_x_content']
        x_recon_content = tf.nn.sigmoid(x_recon_content, name='x_recon_content')
        gen_loss_content = -tf.reduce_mean(tf.reduce_sum(content_x * tf.log(tf.maximum(x_recon_content, 1e-10))
            + (1-content_x) * tf.log(tf.maximum(1 - x_recon_content, 1e-10)),1))
        latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mean) + tf.exp(z_log_sigma_sq)
            - z_log_sigma_sq - 1, 1))
        h2_link = self.activate(tf.matmul(z, gen['W2_link']) + gen['b2_link'], self.activations[1])
        h1_link = self.activate(tf.matmul(h2_link, gen['W1_link']) + gen['b1_link'], self.activations[0])
        x_recon_link = tf.matmul(h1_link, gen['W_x_link']) + gen['b_x_link']
        x_recon_link = tf.nn.sigmoid(x_recon_link, name='x_recon_link')
        gen_loss_link = -tf.reduce_mean(tf.reduce_sum(link_x * tf.log(tf.maximum(x_recon_link, 1e-10))
            + (1 - link_x) * tf.log(tf.maximum(1 - x_recon_link, 1e-10)), 1))
        h2_tfidf = self.activate(tf.matmul(z, gen['W2_tfidf']) + gen['b2_tfidf'], self.activations[1])
        h1_tfidf = self.activate(tf.matmul(h2_tfidf, gen['W1_tfidf']) + gen['b1_tfidf'], self.activations[0])
        x_recon_tfidf = tf.matmul(h1_tfidf, gen['W_x_tfidf']) + gen['b_x_tfidf']
        x_recon_tfidf = tf.nn.sigmoid(x_recon_tfidf, name='x_recon_tfidf')
        gen_loss_tfidf = -tf.reduce_mean(tf.reduce_sum(tfidf_x * tf.log(tf.maximum(x_recon_tfidf, 1e-10)) + (1 - tfidf_x) * tf.log(tf.maximum(1 - x_recon_tfidf, 1e-10)), 1))

        loss = gen_loss_content + self.belta*latent_loss + gen_loss_link + gen_loss_tfidf
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for it in range(num_iter):
                b_x_content, ids_content = utils.get_batch(data_content, batch_size)
                b_x_link = data_link[ids_content, :]
                b_x_tfidf = data_tfidf[ids_content, :]
                _, l, gl_c, gl_l, gl_t, ll = sess.run((train_op, loss, gen_loss_content, gen_loss_link, gen_loss_tfidf, latent_loss), feed_dict={content_x:b_x_content, link_x:b_x_link, tfidf_x:b_x_tfidf})
            if (i + 1) % print_step == 0:
                if content_valid is None or link_valid is None:
                    logging.info('epoch {0}: batch loss = {1}, gen_loss_content={2}, gen_loss_link={3}, gen_loss_tfidf={4}, latent_loss={5}'.format(i, l, gl_c, gl_l, gl_t, ll))
                else:
                    content_valid_loss, link_valid_loss, tfidf_valid_loss = self.validation(content_valid, link_valid, tfidf_valid, sess, gen_loss_content, gen_loss_link, gen_loss_tfidf, content_x, link_x, tfidf_x, batch_size)
                    logging.info('epoch {0}: batch loss = {1}, gen_loss_content={2}, gen_loss_link={3}, gen_loss_tfidf={4}, latent_loss={5}, valid_loss_content={6}, valid_loss_link={7}, valid_loss_tfidf={8}'.format(i, l, gl_c, gl_l, gl_t, ll, content_valid_loss, link_valid_loss, tfidf_valid_loss))

        weight_path = "model/pretrain"
        saver.save(sess, weight_path)
        logging.info("Weights saved at " + weight_path)

    def validation(self, data_content, data_link, data_tfidf, sess, gen_loss_content, gen_loss_link, gen_loss_tfidf, content_x, link_x, tfidf_x, batch_size):
        n_samples = data_content.shape[0]
        num_batches = int(1.0*n_samples/self.batch_size)
        n_samples = num_batches * batch_size
        content_valid_loss = 0.
        link_valid_loss = 0.
        tfidf_valid_loss = 0
        for i in range(num_batches):
            ids = range(i*batch_size, (i+1)*batch_size)
            x_b_content = data_content[ids]
            x_b_link = data_link[ids]
            x_b_tfidf = data_tfidf[ids]
            [gl_c, gl_l, gl_t] = sess.run([gen_loss_content, gen_loss_link, gen_loss_tfidf], feed_dict={content_x: x_b_content, link_x:x_b_link, tfidf_x:x_b_tfidf})
            content_valid_loss += gl_c / n_samples * batch_size
            link_valid_loss += gl_l / n_samples * batch_size
            tfidf_valid_loss += gl_t / n_samples * batch_size
        return content_valid_loss, link_valid_loss, tfidf_valid_loss

    def run_latent_content_link_tfidf(self, data_content, data_link, data_tfidf, hidden_dim, batch_size, lr, epoch, print_step=100):
        tf.reset_default_graph()
        n = data_content.shape[0]
        input_dim_content = len(data_content[0])
        input_dim_link = len(data_link[0])
        input_dim_tfidf = len(data_tfidf[0])
        num_iter = int(n / batch_size)
        sess = tf.Session()
        rec = { 'W_z_mean_content': tf.get_variable("W_z_mean_content", [self.dims[1], self.n_z],
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_mean_content': tf.get_variable("b_z_mean_content", [self.n_z],
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma_content': tf.get_variable("W_z_log_sigma_content", [self.dims[1], self.n_z],
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_log_sigma_content': tf.get_variable("b_z_log_sigma_content", [self.n_z],
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_mean_link': tf.get_variable("W_z_mean_link", [self.dims[1], self.n_z],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    dtype=tf.float32),
                'b_z_mean_link': tf.get_variable("b_z_mean_link", [self.n_z],
                                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma_link': tf.get_variable("W_z_log_sigma_link", [self.dims[1], self.n_z],
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         dtype=tf.float32),
                'b_z_log_sigma_link': tf.get_variable("b_z_log_sigma_link", [self.n_z],
                                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_mean_tfidf': tf.get_variable("W_z_mean_tfidf", [self.dims[1], self.n_z],
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 dtype=tf.float32),
                'b_z_mean_tfidf': tf.get_variable("b_z_mean_tfidf", [self.n_z],
                                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma_tfidf': tf.get_variable("W_z_log_sigma_tfidf", [self.dims[1], self.n_z],
                                                      initializer=tf.contrib.layers.xavier_initializer(),
                                                      dtype=tf.float32),
                'b_z_log_sigma_tfidf': tf.get_variable("b_z_log_sigma_tfidf", [self.n_z],
                                                      initializer=tf.constant_initializer(0.0), dtype=tf.float32)
                }
        gen = {'W2_content': tf.get_variable("W2_content", [self.n_z, self.dims[1]],
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2_content': tf.get_variable("b2_content", [self.dims[1]],
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
               'W2_link': tf.get_variable("W2_link", [self.n_z, self.dims[1]],
                                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
               'b2_link': tf.get_variable("b2_link", [self.dims[1]],
                                             initializer=tf.constant_initializer(0.0), dtype=tf.float32),
               'W2_tfidf': tf.get_variable("W2_tfidf", [self.n_z, self.dims[1]],
                                          initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
               'b2_tfidf': tf.get_variable("b2_tfidf", [self.dims[1]],
                                          initializer=tf.constant_initializer(0.0), dtype=tf.float32)
               }
        content_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_content], name='content_x')
        link_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_link], name='link_x')
        tfidf_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_tfidf], name='tfidf_x')
        z_mean_content = tf.matmul(content_x, rec['W_z_mean_content']) + rec['b_z_mean_content']
        z_mean_link = tf.matmul(link_x, rec['W_z_mean_link']) + rec['b_z_mean_link']
        z_mean_tfidf = tf.matmul(tfidf_x, rec['W_z_mean_tfidf']) + rec['b_z_mean_tfidf']
        z_log_sigma_sq_content = tf.matmul(content_x, rec['W_z_log_sigma_content']) + rec['b_z_log_sigma_content']
        z_log_sigma_sq_link = tf.matmul(link_x, rec['W_z_log_sigma_link']) + rec['b_z_log_sigma_link']
        z_log_sigma_sq_tfidf = tf.matmul(tfidf_x, rec['W_z_log_sigma_tfidf']) + rec['b_z_log_sigma_tfidf']

        z_mean = self.kongzhi_1 * z_mean_content + self.kongzhi_2 * z_mean_link + (1-self.kongzhi_1-self.kongzhi_2) * z_mean_tfidf
        z_log_sigma_sq = self.kongzhi_1 * z_log_sigma_sq_content + self.kongzhi_2 * z_log_sigma_sq_link + (1-self.kongzhi_1-self.kongzhi_2) * z_log_sigma_sq_tfidf

        eps = tf.random_normal((batch_size, hidden_dim), 0, 1,dtype=tf.float32)
        z = z_mean + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), 1e-10)) * eps
        x_recon_content = tf.matmul(z, gen['W2_content']) + gen['b2_content']
        x_recon_content = tf.nn.sigmoid(x_recon_content, name='x_recon_content')
        x_recon_link = tf.matmul(z, gen['W2_link']) + gen['b2_link']
        x_recon_link = tf.nn.sigmoid(x_recon_link, name='x_recon_link')
        x_recon_tfidf = tf.matmul(z, gen['W2_tfidf']) + gen['b2_tfidf']
        x_recon_tfidf = tf.nn.sigmoid(x_recon_tfidf, name='x_recon_tfidf')
        gen_loss_content = -tf.reduce_mean(tf.reduce_sum(content_x * tf.log(tf.maximum(x_recon_content, 1e-10))
            + (1-content_x) * tf.log(tf.maximum(1 - x_recon_content, 1e-10)),1))
        latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mean) + tf.exp(z_log_sigma_sq)
            - z_log_sigma_sq - 1, 1))
        gen_loss_link = -tf.reduce_mean(tf.reduce_sum(link_x * tf.log(tf.maximum(x_recon_link, 1e-10))
             + (1 - link_x) * tf.log(tf.maximum(1 - x_recon_link, 1e-10)), 1))
        gen_loss_tfidf = -tf.reduce_mean(tf.reduce_sum(tfidf_x * tf.log(tf.maximum(x_recon_tfidf, 1e-10))
                                                      + (1 - tfidf_x) * tf.log(tf.maximum(1 - x_recon_tfidf, 1e-10)), 1))
        loss = gen_loss_content + self.belta * latent_loss + gen_loss_link + gen_loss_tfidf
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for it in range(num_iter):
                b_x_content, ids_content = utils.get_batch(data_content, batch_size)
                b_x_link = data_link[ids_content, :]
                b_x_tfidf = data_tfidf[ids_content, :]
                _, l, gl_c, gl_l, gl_t, ll = sess.run((train_op, loss, gen_loss_content, gen_loss_link, gen_loss_tfidf, latent_loss), feed_dict={content_x: b_x_content, link_x:b_x_link, tfidf_x:b_x_tfidf})
            if (i + 1) % print_step == 0:
                logging.info('epoch {0}: batch loss = {1}, gen_loss_content={2}, gen_loss_link={3}, gen_loss_tfidf={4}, latent_loss={5}'.format(i, l, gl_c, gl_l, gl_t, ll))

        self.weights.append(sess.run(rec['W_z_mean_content']))
        self.weights.append(sess.run(rec['W_z_mean_link']))
        self.weights.append(sess.run(rec['W_z_mean_content']))
        self.weights.append(sess.run(rec['W_z_log_sigma_content']))
        self.weights.append(sess.run(rec['W_z_log_sigma_link']))
        self.weights.append(sess.run(rec['W_z_log_sigma_tfidf']))
        self.biases.append(sess.run(rec['b_z_mean_content']))
        self.biases.append(sess.run(rec['b_z_mean_link']))
        self.biases.append(sess.run(rec['b_z_mean_tfidf']))
        self.biases.append(sess.run(rec['b_z_log_sigma_content']))
        self.biases.append(sess.run(rec['b_z_log_sigma_link']))
        self.biases.append(sess.run(rec['b_z_log_sigma_tfidf']))
        self.de_weights.append(sess.run(gen['W2_content']))
        self.de_weights.append(sess.run(gen['W2_link']))
        self.de_weights.append(sess.run(gen['W2_tfidf']))
        self.de_biases.append(sess.run(gen['b2_content']))
        self.de_biases.append(sess.run(gen['b2_link']))
        self.de_biases.append(sess.run(gen['b2_tfidf']))

    # 内容和链接和TFIDF关系
    def run_content_link_tfidf(self, data_content, data_link, data_tfidf, hidden_dim, activation, loss, lr, print_step, epoch, batch_size=100):
        tf.reset_default_graph()
        input_dim_content = len(data_content[0])
        input_dim_link = len(data_link[0])
        input_dim_tfidf = len(data_tfidf[0])
        # print(input_dim_content, input_dim_link)
        n = data_content.shape[0]
        num_iter = int(n / batch_size)
        sess = tf.Session()
        content_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_content], name='content_x')
        link_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_link], name='link_x')
        tfidf_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_tfidf], name='tfidf_x')
        content_x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_content], name='content_x_')
        link_x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_link], name='link_x_')
        tfidf_x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_tfidf], name='tfidf_x_')

        content_encode = {
            'content_weights': tf.Variable(xavier_init(input_dim_content, hidden_dim, dtype=tf.float32)),
             'content_biases': tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32))}
        content_decode = {'content_biases': tf.Variable(tf.zeros([input_dim_content], dtype=tf.float32)),
                            'content_weights': tf.transpose(content_encode['content_weights'])}
        content_encoded = self.activate(
            tf.matmul(content_x, content_encode['content_weights']) + content_encode['content_biases'], activation)
        content_decoded = tf.matmul(content_encoded, content_decode['content_weights']) + content_decode[
            'content_biases']

        link_encode = {'link_weights': tf.Variable(xavier_init(input_dim_link, hidden_dim, dtype=tf.float32)),
                        'link_biases': tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32))}
        link_decode = {'link_biases': tf.Variable(tf.zeros([input_dim_link], dtype=tf.float32)),
                        'link_weights': tf.transpose(link_encode['link_weights'])}
        link_encoded = self.activate(tf.matmul(link_x, link_encode['link_weights']) + link_encode['link_biases'], activation)
        link_decoded = tf.matmul(link_encoded, link_decode['link_weights']) + link_decode['link_biases']

        tfidf_encode = {'tfidf_weights': tf.Variable(xavier_init(input_dim_tfidf, hidden_dim, dtype=tf.float32)),
                       'tfidf_biases': tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32))}
        tfidf_decode = {'tfidf_biases': tf.Variable(tf.zeros([input_dim_tfidf], dtype=tf.float32)),
                       'tfidf_weights': tf.transpose(tfidf_encode['tfidf_weights'])}
        tfidf_encoded = self.activate(tf.matmul(tfidf_x, tfidf_encode['tfidf_weights']) + tfidf_encode['tfidf_biases'], activation)
        tfidf_decoded = tf.matmul(tfidf_encoded, tfidf_decode['tfidf_weights']) + tfidf_decode['tfidf_biases']

        # reconstruction loss
        if loss == 'rmse':
            # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
            content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(content_x_ - content_decoded), 1))
            link_loss = tf.reduce_mean(tf.reduce_sum(tf.square(link_x_ - link_decoded), 1))
            loss = content_loss + link_loss
        elif loss == 'cross-entropy':
            content_decoded = tf.nn.sigmoid(content_decoded, name='content_decoded')
            content_loss = -tf.reduce_mean(tf.reduce_sum(
                content_x_ * tf.log(tf.maximum(content_decoded, 1e-16)) + (1 - content_x_) * tf.log(
                    tf.maximum(1 - content_decoded, 1e-16)), 1))
            link_decoded = tf.nn.sigmoid(link_decoded, name='link_decoded')
            link_loss = -tf.reduce_mean(tf.reduce_sum(
                link_x_ * tf.log(tf.maximum(link_decoded, 1e-16)) + (1 - link_x_) * tf.log(
                    tf.maximum(1 - link_decoded, 1e-16)), 1))
            tfidf_decoded = tf.nn.sigmoid(tfidf_decoded, name='tfidf_decoded')
            tfidf_loss = -tf.reduce_mean(tf.reduce_sum(
                tfidf_x_ * tf.log(tf.maximum(tfidf_decoded, 1e-16)) + (1 - tfidf_x_) * tf.log(
                    tf.maximum(1 - tfidf_decoded, 1e-16)), 1))
            loss = content_loss + link_loss + tfidf_loss
            # loss = content_loss
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for it in range(num_iter):
                b_content_, ids_content = utils.get_batch(data_content, batch_size)
                b_content = self.add_noise(b_content_)
                b_link_ = data_link[ids_content,]
                # print('data_link shape:',np.shape(data_link), np.shape(b_link_))
                b_link = self.add_noise(b_link_)
                b_tfidf_ = data_tfidf[ids_content,]
                b_tfidf = self.add_noise(b_tfidf_)
                _, l = sess.run((train_op, loss),
                                feed_dict={content_x: b_content, link_x: b_link, tfidf_x: b_tfidf,
                                           content_x_: b_content_, link_x_: b_link_, tfidf_x_: b_tfidf_})
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={content_x: b_content, link_x: b_link, tfidf_x: b_tfidf,
                                              content_x_: b_content_, link_x_: b_link_, tfidf_x_:b_tfidf_})
                logging.info('epoch {0}: batch loss = {1}'.format(i, l))
        # debug
        self.weights.append(sess.run(content_encode['content_weights']))
        self.weights.append(sess.run(link_encode['link_weights']))
        self.weights.append(sess.run(tfidf_encode['tfidf_weights']))
        self.biases.append(sess.run(content_encode['content_biases']))
        self.biases.append(sess.run(link_encode['link_biases']))
        self.biases.append(sess.run(tfidf_encode['tfidf_biases']))
        self.de_weights.append(sess.run(content_decode['content_weights']))
        self.de_weights.append(sess.run(link_decode['link_weights']))
        self.de_weights.append(sess.run(tfidf_decode['tfidf_weights']))
        self.de_biases.append(sess.run(content_decode['content_biases']))
        self.de_biases.append(sess.run(link_decode['link_biases']))
        self.de_biases.append(sess.run(tfidf_decode['tfidf_biases']))
        return sess.run([content_encoded, link_encoded, tfidf_encoded], feed_dict={content_x: data_content, link_x: data_link,
                                                                                   tfidf_x: data_tfidf})

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
