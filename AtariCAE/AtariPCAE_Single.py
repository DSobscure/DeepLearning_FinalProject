import gym
import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

GAMMA = 0.99

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORE_STPES = 200000

# replay memory
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 100000

BATCH_SIZE = 32
CODE_SIZE = 8
CODE_LEVEL = 2

def plot_n_reconstruct(origin_img, reconstruct_img, n = 10):

    plt.figure(figsize=(2 * 10, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(origin_img[i].reshape(84, 84))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruct_img[i].reshape(84, 84))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state).reshape([84,84,1])

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=True)

class CAE():
    def __init__(self):
        self.encode_level = CODE_LEVEL
        self.code_size = CODE_SIZE
        self.state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='state')
        self.original_states = [self.state]
        self.rec_states = []
        self.codes = []
        for i in range(self.encode_level):
            rec_state, code = self.build_network(self.original_states[i], trainable=True)  
            self.rec_states.append(rec_state)
            self.codes.append(code)
            self.original_states.append(self.original_states[i] - rec_state)
        self.rec_loss = []
        self.code_loss = []
        for i in range(self.encode_level):
            self.rec_loss.append(tf.reduce_mean(tf.pow(self.rec_states[i] - self.original_states[i], 2)))
            self.code_loss.append(tf.reduce_mean(tf.pow(self.codes[i] - tf.random_uniform(shape = [self.code_size],minval=-1,maxval=1), 2)))
            print('build loss', i)
        self.optimize = []
        for i in range(self.encode_level):
            self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(self.rec_loss[i] + self.code_loss[i]))
            print('build optimize', i)
        #self.rec_states = []
        #self.codes = []
        #for i in range(self.encode_level):
        #    rec_state, code = self.build_network(self.state, trainable=True)  
        #    self.rec_states.append(rec_state)
        #    self.codes.append(code)
        self.rec_state = sum(self.rec_states)
        #print(self.rec_state.get_shape())
        #self.rec_loss = tf.reduce_mean(tf.pow(self.rec_state - self.state, 2))
        #self.code_loss = []
        #for i in range(self.encode_level):
        #    self.code_loss.append(-tf.reduce_mean(tf.pow(self.codes[i], 2)))
        #self.code_loss_sum = sum(self.code_loss)
        #self.optimize = [tf.train.RMSPropOptimizer(0.00025).minimize(self.rec_loss + self.code_loss_sum)]
        
        ##for i in range(self.encode_level):
        ##    self.optimize.append(tf.train.RMSPropOptimizer(0.001).minimize(self.code_loss[i]))
        ##    print('build code optimize', i)
        

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias      
        conv1_hidden_bn = batch_norm(conv1_hidden_sum)        
        conv1_hidden = tf.nn.elu(conv1_hidden_bn)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias
        conv2_hidden_bn = batch_norm(conv2_hidden_sum)
        conv2_hidden = tf.nn.elu(conv2_hidden_bn)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden_bn = batch_norm(conv3_hidden_sum)
        conv3_hidden = tf.nn.elu(conv3_hidden_bn)

        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*32])
        fc1_weight = tf.Variable(tf.truncated_normal([11*11*32, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)      
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden_bn = batch_norm(fc1_hidden_sum)
        fc1_hidden = tf.nn.elu(fc1_hidden_bn)

        fc2_weight = tf.Variable(tf.truncated_normal([512, self.code_size], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)      
        fc2_hidden_sum = tf.matmul(fc1_hidden, fc2_weight) + fc2_bias
        fc2_hidden_bn = batch_norm(fc2_hidden_sum)
        fc2_hidden = tf.nn.tanh(fc2_hidden_bn)

        code_layer = fc2_hidden
        print("code layer shape : %s" % code_layer.get_shape())

        dfc1_weight = tf.Variable(tf.truncated_normal([self.code_size, 512], stddev = 0.02))
        dfc1_bias = tf.Variable(tf.constant(0.02, shape = [512]))
        dfc1_hidden_sum = tf.matmul(fc2_hidden, dfc1_weight) + dfc1_bias
        dfc1_hidden_bn = batch_norm(dfc1_hidden_sum)
        dfc1_hidden = tf.nn.elu(dfc1_hidden_bn)

        dfc2_weight = tf.Variable(tf.truncated_normal([512, 11*11*32], stddev = 0.02))
        dfc2_bias = tf.Variable(tf.constant(0.02, shape = [11*11*32]))
        dfc2_hidden_sum = tf.matmul(dfc1_hidden, dfc2_weight) + dfc2_bias
        dfc2_hidden_bn = batch_norm(dfc2_hidden_sum)
        dfc2_hidden = tf.nn.elu(dfc2_hidden_bn)
        dfc2_hidden_conv = tf.reshape(dfc2_hidden, [-1, 11, 11, 32])

        dconv1_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        dconv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        dconv1_output_shape = tf.stack([tf.shape(x)[0], 11, 11, 32])
        dconv1_hidden_sum = tf.nn.conv2d_transpose(dfc2_hidden_conv, dconv1_weight, dconv1_output_shape, strides = [1,1,1,1], padding='SAME') + dconv1_bias
        dconv1_hidden_bn = batch_norm(dconv1_hidden_sum)
        dconv1_hidden = tf.nn.elu(dconv1_hidden_bn)

        dconv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        dconv2_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)
        dconv2_output_shape = tf.stack([tf.shape(x)[0], 21, 21, 16])
        dconv2_hidden_sum = tf.nn.conv2d_transpose(dconv1_hidden, dconv2_weight, dconv2_output_shape, strides = [1,2,2,1], padding='SAME') + dconv2_bias
        dconv2_hidden_bn = batch_norm(dconv2_hidden_sum)
        dconv2_hidden = tf.nn.elu(dconv2_hidden_bn)

        dconv3_weight = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev = 0.02), trainable = trainable)
        dconv3_bias = tf.Variable(tf.constant(0.02, shape = [1]), trainable = trainable)
        dconv3_output_shape = tf.stack([tf.shape(x)[0], 84, 84, 1])
        dconv3_hidden_sum = tf.nn.conv2d_transpose(dconv2_hidden, dconv3_weight, dconv3_output_shape, strides = [1,4,4,1], padding='SAME') + dconv3_bias

        return dconv3_hidden_sum, code_layer

    def update_code(self, sess, state):
        sess.run(self.optimize, feed_dict={self.state : state})

    def get_code(self, state):
        codes = []
        for i in range(self.encode_level):
            codes.append(self.codes[i].eval(feed_dict={self.state: state}))
        result = []
        
        for i in range(len(state)):
            number = 0
            for c in range(self.encode_level):
                for j in range(CODE_SIZE):
                    number *= 2
                    if codes[c][i][j] > 0:
                        number += 1
            result.append(number)
        return result

    def get_rec_state(self, state):
        return self.rec_state.eval(feed_dict={self.state: state})

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")

    # The replay memory
    replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    cae = CAE()
    code_set = set()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = observation
    initial_state = state

    episode_reward = 0    
    epsilon = INITIAL_EPSILON

    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = random.randrange(4)

        next_observation, reward, done, _ = env.step(action)
        next_observation = process_state(next_observation)
        next_state = next_observation
        replay_memory.append(next_state - state)

        episode_reward += reward

        # Current game episode is over
        if done:
            observation = env.reset()                 # retrive first env image
            observation = process_state(observation)
            state = observation

            print ("Episode reward: ", episode_reward, "Buffer: ", len(replay_memory))
            episode_reward = 0
        # Not over yet
        else:
            state = next_state

    for i in range(500000):
        samples = random.sample(replay_memory, BATCH_SIZE)
        cae.update_code(sess, samples)
        if i % 1000 == 0:
            print("generate code...", i)
            print(cae.get_code(samples))
            print(cae.get_code([initial_state]))
            for c in range(CODE_LEVEL):
                print(cae.rec_loss[c].eval(feed_dict={cae.state: samples}))
                print(cae.code_loss[c].eval(feed_dict={cae.state: samples}))
                
        if (i) % 10000 == 0:
            #for c in range(CODE_LEVEL):
            #    plot_n_reconstruct(samples, cae.rec_states[c].eval(feed_dict={cae.state: samples}))
            plot_n_reconstruct(samples, cae.get_rec_state(samples))

if __name__ == '__main__':
    tf.app.run()