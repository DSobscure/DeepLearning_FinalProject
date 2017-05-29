import gym
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

EXPLORE_STPES = 500000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 1000000

BATCH_SIZE = 32

def plot_n_reconstruct(origin_img, reconstruct_img, n = 10):

    plt.figure(figsize=(2 * 10, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(origin_img[i].reshape(84, 84, 4))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruct_img[i].reshape(84, 84, 4))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def code_converter(codeBatch):
    result = []
    for i in range(BATCH_SIZE):
        number = 0
        for j in range(64):
            number *= 2
            if codeBatch[i][j] > 0:
                number += 1
        result.append(number)
    return result

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state)

class CAE():
    def __init__(self):
        self.num_actions = 4
        with tf.variable_scope('inputs'):
            self.first_observation = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='first_observation')
        with tf.variable_scope('target_model'):
            self.target_output, self.code = self.build_network(self.first_observation, trainable=True)
        self.loss = tf.reduce_mean(tf.pow(self.target_output - self.first_observation, 2))
        self.train_step = tf.train.RMSPropOptimizer(0.000025, momentum=0.95, epsilon=0.01).minimize(self.loss)    

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv1_hidden = tf.nn.relu(tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv2_hidden = tf.nn.relu(tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv3_hidden = tf.nn.relu(tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias)

        #fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, 512], stddev = 0.02), trainable = trainable)
        #fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        #conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])
        #fc1_hidden = tf.nn.relu(tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias)

        #fc2_weight = tf.Variable(tf.truncated_normal([512, 64], stddev = 0.02), trainable = trainable)
        #fc2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        #fc2_hidden = tf.nn.relu(tf.matmul(fc1_hidden, fc2_weight) + fc2_bias)

        #print("code layer shape : %s" % fc2_hidden.get_shape())

        #dfc1_weight = tf.Variable(tf.truncated_normal([64, 512], stddev = 0.02), trainable = trainable)
        #dfc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        #dfc1_hidden = tf.nn.relu(tf.matmul(fc2_hidden, dfc1_weight) + dfc1_bias)

        #dfc2_weight = tf.Variable(tf.truncated_normal([512, 11*11*64], stddev = 0.02), trainable = trainable)
        #dfc2_bias = tf.Variable(tf.constant(0.02, shape = [11*11*64]), trainable = trainable)
        #dfc2_hidden = tf.nn.relu(tf.matmul(dfc1_hidden, dfc2_weight) + dfc2_bias)

        #dfc2_hidden_conv = tf.reshape(dfc2_hidden, [-1, 11, 11, 64])
        dfc2_hidden_conv = conv3_hidden

        dconv1_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        dconv1_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        dconv1_output_shape = tf.stack([tf.shape(x)[0], 11, 11, 64])
        dconv1_hidden = tf.nn.relu(tf.nn.conv2d_transpose(dfc2_hidden_conv, dconv1_weight, dconv1_output_shape, strides = [1,1,1,1], padding='SAME') + dconv1_bias)

        dconv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        dconv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        dconv2_output_shape = tf.stack([tf.shape(x)[0], 21, 21, 32])
        dconv2_hidden = tf.nn.relu(tf.nn.conv2d_transpose(dconv1_hidden, dconv2_weight, dconv2_output_shape, strides = [1,2,2,1], padding='SAME') + dconv2_bias)

        dconv3_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        dconv3_bias = tf.Variable(tf.constant(0.02, shape = [4]), trainable = trainable)
        dconv3_output_shape = tf.stack([tf.shape(x)[0], 84, 84, 4])
        dconv3_hidden = tf.nn.relu(tf.nn.conv2d_transpose(dconv2_hidden, dconv3_weight, dconv3_output_shape, strides = [1,4,4,1], padding='SAME') + dconv3_bias)

        return dconv3_hidden, conv3_hidden

    def update(self, sess, ob0):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.first_observation : ob0})

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")

    # The replay memory
    replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    cae = CAE()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1000)

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times

    episode_reward = 0    
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = random.randrange(4)

        next_observation, reward, done, _ = env.step(action)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((state))

        episode_reward += reward

        # Current game episode is over
        if done:
            observation = env.reset()
            observation = process_state(observation)
            state = np.stack([observation] * 4, axis=2)

            print ("Episode reward: ", episode_reward, "Buffer: ", len(replay_memory))
            episode_reward = 0
        # Not over yet
        else:
            state = next_state

    # total steps
    total_t = 0

    for episode in range(5000):

        # Reset the environment
        observation = env.reset()
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward
        '''
        How to update episode reward:
        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        episode_reward += reward
        '''

        for t in itertools.count():

            # choose a action
            action = random.randrange(4)
            # execute the action
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # save the transition to replay buffer
            replay_memory.append((state))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                if total_t % 1000 == 0:
                    print("step %d, loss %g"%(total_t, cae.loss.eval(feed_dict={cae.first_observation: samples})))
                    if total_t % 10000 == 0:
                        test_size = 10
                        test_reconstruct_img = np.reshape(cae.target_output.eval(feed_dict = {cae.first_observation: samples}), [-1, 84, 84, 4])
                        plot_n_reconstruct(samples, test_reconstruct_img)

                    #print(code_converter(cae.code.eval(feed_dict={cae.first_observation: samples})))
			    # Update network
                cae.update(sess, samples)

            if done:
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t)
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()