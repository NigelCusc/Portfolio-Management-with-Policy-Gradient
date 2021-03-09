'''
Citation:
    Heavily inspired by this work:
    @Author: Louis Liang
    https://github.com/liangzp
    Several of his papers are discussed in my report
'''

import tensorflow as tf
import tflearn
import numpy as np
import os

class PG:
    def __init__(self, M, L, N, name, load_weights, trainable, number, lr=10e-2):
        # Initial buffer
        self.buffer = list()
        self.name = name
        self.learning_rate = lr
        self.number = str(number)
        # Build up models
        self.session = tf.Session()

        # Initial input shape
        self.M = M  # number of assets
        self.L = L  # window length
        self.N = N  # number of features
        self.global_step = tf.Variable(0, trainable=False)

        self.state, self.w_previous, self.out = self.build_net()
        self.future_price = tf.placeholder(tf.float32, [None] + [self.M])
        self.pv_vector = tf.reduce_sum(self.out * self.future_price, reduction_indices=[1]) * self.pc()
        self.profit = tf.reduce_prod(self.pv_vector)
        self.loss = -tf.reduce_mean(tf.log(self.pv_vector))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Initial saver
        self.saver = tf.train.Saver(max_to_keep=10)

        if load_weights == 'True':
            print("Loading Model for attempt: " + self.number)
            try:
                checkpoint = tf.train.get_checkpoint_state('./result/ai/PG/'+self.number+'/'+'saved_network/')
                print('./saved_network/PG/')
                if checkpoint and checkpoint.model_checkpoint_path:
                    tf.compat.v1.reset_default_graph()
                    self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                    print("Successfully loaded:", checkpoint.model_checkpoint_path)
                else:
                    print("Could not find old network weights")
                    self.session.run(tf.compat.v1.global_variables_initializer())

            except:
                print("Could not find old network weights")
                self.session.run(tf.compat.v1.global_variables_initializer())
        else:
            self.session.run(tf.compat.v1.global_variables_initializer())

        if trainable == 'True':
            # Initial summary
            self.summary_writer = tf.summary.FileWriter('./result/ai/PG/'+self.number+'/'+'summary/', self.session.graph)
            self.summary_ops, self.summary_vars = self.build_summaries()

    # Setting up Neural Network
    def build_net(self):
        state = tf.placeholder(tf.float32, shape=[None]+[self.M]+[self.L]+[self.N], name='market_situation')
        network = tflearn.layers.conv_2d(state, 2,
                                         [1, 2],
                                         [1, 1, 1, 1],
                                         'valid',
                                         'relu')
        width = network.get_shape()[2]
        network = tflearn.layers.conv_2d(network, 48,
                                         [1, width],
                                         [1, 1],
                                         "valid",
                                         'relu',
                                         regularizer="L2",
                                         weight_decay=5e-9)
        w_previous = tf.placeholder(tf.float32, shape=[None, self.M])
        network = tf.concat([network, tf.reshape(w_previous, [-1, self.M, 1, 1])], axis=3)
        network = tflearn.layers.conv_2d(network, 1,
                                         [1, network.get_shape()[2]],
                                         [1, 1],
                                         "valid",
                                         'relu',
                                         regularizer="L2",
                                         weight_decay=5e-9)
        network = tf.layers.flatten(network)
        w_init = tf.random_uniform_initializer(-0.005, 0.005)
        out = tf.layers.dense(network, self.M, activation=tf.nn.softmax, kernel_initializer=w_init)

        return state, w_previous, out

    def pc(self):
        return 1-tf.reduce_sum(tf.abs(self.out[:, 1:]-self.w_previous[:, 1:]), axis=1)*0.0005

    def predict(self, s, a_previous):
        return self.session.run(self.out, feed_dict={self.state: s, self.w_previous: a_previous})

    def save_transition(self, s, p, action, action_previous):
        self.buffer.append((s, p, action, action_previous))

    def train(self):
        s, p, a, a_previous = self.get_buffer()
        profit, _ = self.session.run([self.profit, self.optimize],
                                    feed_dict={self.state: s,
                                               self.out: np.reshape(a, (-1, self.M)),
                                               self.future_price: np.reshape(p, (-1, self.M)),
                                               self.w_previous: np.reshape(a_previous, (-1, self.M))
                                               })

    def get_buffer(self):
        s = [data[0][0] for data in self.buffer]
        p = [data[1] for data in self.buffer]
        a = [data[2] for data in self.buffer]
        a_previous = [data[3] for data in self.buffer]
        return s, p, a, a_previous

    def reset_buffer(self):
        self.buffer = list()

    def save_model(self):
        path = './result/ai/PG/'+self.number+'/'+'saved_network/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.session, path+self.name, global_step=self.global_step)

    def write_summary(self, reward):
        summary_str = self.session.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: reward,
        })
        self.summary_writer.add_summary(summary_str, self.session.run(self.global_step))

    def close(self):
        self.session.close()

    def build_summaries(self):
        self.reward = tf.Variable(0.)
        tf.summary.scalar('Reward', self.reward)
        summary_vars = [self.reward]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars
