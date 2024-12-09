# models/gan.py

import tensorflow as tf
import numpy as np
import time
from math import ceil
from utils.helpers import (
    extract_time,
    rnn_cell,
    random_generator,
    batch_generator,
    TokenAndPositionEmbedding,
    EncoderLayer,
    DecoderLayer,
    create_look_ahead_mask,
)
from utils.data_processing import train_test_divide


class CryptoGAN:
    def __init__(self, parameters):
        """
        Initialize the CryptoGAN with the given parameters.

        Args:
            parameters (dict): Dictionary containing model and training parameters.
        """
        self.parameters = parameters
        self.build_model()

    def build_model(self):
        """
        Build the GAN and transformer models, define loss functions and optimizers.
        """
        parameters = self.parameters

        # Reset the default graph
        tf.reset_default_graph()

        # Basic Parameters
        self.hidden_dim = parameters['hidden_dim']
        self.num_layers = parameters['num_layer']
        self.iterations = parameters['iterations']
        self.batch_size = parameters['batch_size']
        self.module_name = parameters['module']
        self.d_model = parameters['d_model']
        self.num_heads = parameters['num_heads']
        self.dff = parameters['dff']
        self.z_dim = parameters['z_dim']
        self.gamma = parameters.get('gamma', 1)

        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, None, parameters.get('dim', 1)], name="input_x")
        self.Z = tf.placeholder(tf.float32, [None, None, self.z_dim], name="input_z")
        self.T = tf.placeholder(tf.int32, [None], name="input_t")
        self.training = tf.placeholder(tf.bool, shape=(), name="training")

        # Extract time and max sequence length
        self.time, self.max_seq_len = extract_time(self.X)

        # Create look-ahead mask
        self.mask = create_look_ahead_mask(self.max_seq_len)

        # Build Models
        self.H = self.embedder(self.X, self.T)
        self.X_hat = self.supervisor(self.generator(self.Z, self.T), self.T)
        self.E_hat = self.embedder(self.X_hat, self.T)
        self.X_tilde = self.recovery(self.X, self.H)
        self.X_hat_e = self.recovery(self.X_hat, self.E_hat)

        # Discriminator
        self.Y_real = self.discriminator(self.X, self.T)
        self.Y_fake = self.discriminator(self.X_hat, self.T)

        # Define Variables
        self.e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
        self.r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
        self.g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
        self.s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
        self.d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

        # Define Losses
        self.define_losses()

        # Define Optimizers
        self.define_optimizers()

        # Initialize Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def embedder(self, X, T):
        """
        Embedding network to encode input time-series data.

        Args:
            X (tf.Tensor): Input time-series data.
            T (tf.Tensor): Time information.

        Returns:
            tf.Tensor: Embedded representations.
        """
        with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
            embedding_layer = TokenAndPositionEmbedding(self.max_seq_len, self.d_model, self.dff, True)
            x = embedding_layer(X)
            for _ in range(3):
                encoder_block = EncoderLayer(self.d_model, self.num_heads, self.dff)
                x = encoder_block(x, self.training, None)
            H = x
        return H

    def recovery(self, X1, H1):
        """
        Recovery network to decode embedded representations back to original space.

        Args:
            X1 (tf.Tensor): Input data.
            H1 (tf.Tensor): Embedded representations.

        Returns:
            tf.Tensor: Recovered data.
        """
        with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
            embedding_layer = TokenAndPositionEmbedding(self.max_seq_len, self.d_model, self.dff, True)
            x = tf.concat([tf.zeros_like(X1[:, :1, :]), X1[:, :-1, :]], axis=1)
            x = embedding_layer(x)
            for _ in range(3):
                decoder_block = DecoderLayer(self.d_model, self.num_heads, self.dff)
                x = decoder_block(x, H1, self.training, self.mask, None)
            X_tilde1 = tf.layers.Dense(X1.get_shape().as_list()[-1])(x)
        return X_tilde1

    def generator(self, Z, T):
        """
        Generator network to produce synthetic time-series data.

        Args:
            Z (tf.Tensor): Random noise vector.
            T (tf.Tensor): Time information.

        Returns:
            tf.Tensor: Generated data.
        """
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(self.module_name, self.hidden_dim) for _ in range(2)])
            e_outputs, _ = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
            embedding_layer = TokenAndPositionEmbedding(self.max_seq_len, self.d_model, self.dff, False)
            x = embedding_layer(e_outputs)
            encoder_block = EncoderLayer(self.d_model, self.num_heads, self.dff)
            x = encoder_block(x, self.training, None)
        return x

    def supervisor(self, X, T):
        """
        Supervisor function to refine generator output.

        Args:
            X (tf.Tensor): Pre-generated data from the generator.
            T (tf.Tensor): Time information.

        Returns:
            tf.Tensor: Supervised generated data.
        """
        with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
            e_cell2 = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(self.module_name, self.hidden_dim) for _ in range(2)])
            e_outputs2, _ = tf.nn.dynamic_rnn(e_cell2, X, dtype=tf.float32, sequence_length=T)
            F = e_outputs2 + X
            E = tf.layers.dense(F, self.X.get_shape().as_list()[-1], activation=None)
        return E

    def discriminator(self, X, T):
        """
        Discriminator network to distinguish between real and synthetic data.

        Args:
            X (tf.Tensor): Input time-series data.
            T (tf.Tensor): Time information.

        Returns:
            tf.Tensor: Discriminator output logits.
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(self.module_name, self.hidden_dim) for _ in range(3)])
            d_outputs, _ = tf.nn.static_rnn(d_cell, [X[:, i, :] for i in range(self.max_seq_len)], dtype=tf.float32, sequence_length=T)
            Y_hat = tf.layers.dense(d_outputs, 1, activation=None)
        return Y_hat

    def define_losses(self):
        """
        Define the loss functions for the generator and discriminator.
        """
        # Discriminator WGAN-GP loss
        self.D_loss_real = tf.reduce_mean(self.Y_real)
        self.D_loss_fake = tf.reduce_mean(self.Y_fake)

        alpha = tf.random_uniform(
            shape=[self.batch_size, 1, 1],
            minval=0.0,
            maxval=1.0
        )
        real_data = self.X
        fake_data = self.X_hat
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        gradients = tf.gradients(tf.reduce_mean(self.discriminator(interpolates, self.T), axis=1), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)

        self.D_loss = -self.D_loss_real + self.D_loss_fake + 10 * self.gradient_penalty

        # Generator loss
        self.G_loss_U = -tf.reduce_mean(self.Y_fake)
        self.G_loss_S = tf.losses.mean_squared_error(self.X_hat, self.X_hat_e)
        G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(self.X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(self.X, [0])[1] + 1e-6)))
        G_loss_V2 = tf.reduce_mean(tf.abs(tf.nn.moments(self.X_hat, [0])[0] - tf.nn.moments(self.X, [0])[0]))
        self.G_loss_V = G_loss_V1 + G_loss_V2
        self.G_loss = self.G_loss_U + 100 * tf.sqrt(self.G_loss_S) + 100 * self.G_loss_V

        # Transformer network loss for pretraining
        self.X_tilde = self.recovery(self.X, self.H)
        self.E_loss_T0 = tf.losses.mean_squared_error(self.X, self.X_tilde)
        self.E_loss0 = 10 * self.E_loss_T0

    def define_optimizers(self):
        """
        Define the optimizers for the transformer, discriminator, and generator.
        """
        self.E0_solver = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.0,
            beta2=0.9
        ).minimize(self.E_loss0, var_list=self.e_vars + self.r_vars)

        self.D_solver = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.0,
            beta2=0.9
        ).minimize(self.D_loss, var_list=self.d_vars)

        self.G_solver = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.0,
            beta2=0.9
        ).minimize(self.G_loss, var_list=self.g_vars + self.s_vars)

    def train(self, ori_data):
        """
        Train the GAN model using the original data.

        Args:
            ori_data (np.ndarray): Original time-series data.
        """
        start = time.time()
        no, _, _ = ori_data.shape

        # Preprocess data
        ori_time, _ = extract_time(ori_data)
        norm_data, min_val, max_val = self.min_max_scaler(ori_data)
        print(norm_data.shape)

        # Pretraining the transformer
        print('Start Embedding Network Training')
        for itt in range(self.iterations):
            X_mb, T_mb = batch_generator(norm_data, ori_time, self.batch_size)
            _, step_e_loss = self.sess.run([self.E0_solver, self.E_loss_T0],
                                           feed_dict={self.X: X_mb, self.T: T_mb, self.training: True})
            if itt % 1000 == 0:
                print(f'step: {itt}/{self.iterations}, e_loss: {np.round(step_e_loss, 4)}')
        print('Finish Embedding Network Training')

        # GAN Training
        print('Start Joint Training')
        for itt in range(self.iterations):
            if itt % 100 == 0:
                print(f'step: {itt}/{self.iterations}')

            # Train generator twice
            for _ in range(2):
                X_mb, T_mb = batch_generator(norm_data, ori_time, self.batch_size)
                Z_mb = random_generator(self.batch_size, self.z_dim, T_mb, self.max_seq_len)
                _, g_loss_u, g_loss_s, g_loss_v = self.sess.run(
                    [self.G_solver, self.G_loss_U, self.G_loss_S, self.G_loss_V],
                    feed_dict={self.Z: Z_mb, self.X: X_mb, self.T: T_mb, self.training: True}
                )
                if itt % 100 == 0:
                    print(f', g_loss_u: {np.round(g_loss_u, 4)}, g_loss_s: {np.round(g_loss_s, 4)}, g_loss_v: {np.round(g_loss_v, 4)}')

            # Train discriminator once
            for _ in range(1):
                X_mb, T_mb = batch_generator(norm_data, ori_time, self.batch_size)
                Z_mb = random_generator(self.batch_size, self.z_dim, T_mb, self.max_seq_len)
                _, d_loss, d_loss_real, d_loss_fake, gp = self.sess.run(
                    [self.D_solver, self.D_loss, self.D_loss_real, self.D_loss_fake, self.gradient_penalty],
                    feed_dict={self.X: X_mb, self.T: T_mb, self.Z: Z_mb, self.training: True}
                )
                if itt % 100 == 0:
                    print(f', d_loss: {np.round(d_loss, 4)}, d_loss_real: {np.round(d_loss_real, 4)}, '
                          f'd_loss_fake: {np.round(d_loss_fake, 4)}, gp: {np.round(gp, 4)}')
        print('Finish Joint Training')
        print('Overall training time:', time.time() - start)

    def generate(self, ori_data):
        """
        Generate synthetic time-series data using the trained GAN model.

        Args:
            ori_data (np.ndarray): Original time-series data.

        Returns:
            list: Generated synthetic time-series data.
        """
        no, _, _ = ori_data.shape
        num_batches = int(ceil(no / self.batch_size))
        generated_data_curr = None

        for i in range(num_batches):
            current_batch_size = self.batch_size if i != num_batches - 1 else no - i * self.batch_size
            Z_mb = random_generator(current_batch_size, self.z_dim, self.time[i * self.batch_size:(i + 1) * self.batch_size], self.max_seq_len)
            X_batch = ori_data[i * self.batch_size:(i + 1) * self.batch_size]
            T_batch = self.time[i * self.batch_size:(i + 1) * self.batch_size]
            generated_batch = self.sess.run(self.X_hat, feed_dict={self.Z: Z_mb, self.X: X_batch, self.T: T_batch, self.training: False})
            if generated_data_curr is None:
                generated_data_curr = generated_batch
            else:
                generated_data_curr = np.concatenate((generated_data_curr, generated_batch), axis=0)

        generated_data = []
        for i in range(no):
            temp = generated_data_curr[i, :self.time[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data = np.array(generated_data) * self.max_val
        generated_data = generated_data + self.min_val

        return generated_data

    @staticmethod
    def min_max_scaler(data):
        """
        Apply Min-Max scaling to the data.

        Args:
            data (np.ndarray): Original data.

        Returns:
            tuple: Normalized data, min values, max values.
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        norm_data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = norm_data / (max_val + 1e-7)
        return norm_data, min_val, max_val
