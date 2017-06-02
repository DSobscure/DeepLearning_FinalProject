import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game

EXPLORE_STPES = 500000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 1000
REPLAY_MEMORY_SIZE = 1000000

BATCH_SIZE = 32
CODE_SIZE = 64

def plot_n_reconstruct(origin_img, reconstruct_img, n = 10):

    plt.figure(figsize=(2 * 10, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(origin_img[i].reshape(80, 80, 4))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruct_img[i].reshape(80, 80, 4))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def code_converter(codeBatch):
    result = []
    for i in range(len(codeBatch)):
        number = 0
        for j in range(CODE_SIZE):
            number *= 2
            if codeBatch[i][j] > 0.5:
                number += 1
        result.append(number)
    return result

def process_state(state):
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return np.array(state)

class CAE():
    def __init__(self):
        with tf.variable_scope('inputs'):
            self.first_observation = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name='first_observation')
        with tf.variable_scope('target_model'):
            self.target_output, self.code = self.build_network(self.first_observation, trainable=True)
        self.loss = tf.reduce_mean(tf.pow(self.target_output - self.first_observation, 2))
        self.train_step = tf.train.RMSPropOptimizer(0.00025).minimize(self.loss)    

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 16], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)        
        conv1_hidden = tf.nn.elu(tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv2_hidden = tf.nn.elu(tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)        
        conv3_hidden = tf.nn.elu(tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias)

        fc1_weight = tf.Variable(tf.truncated_normal([10*10*32, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 10*10*32])
        fc1_hidden = tf.nn.elu(tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias)

        fc2_weight = tf.Variable(tf.truncated_normal([512, CODE_SIZE], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [CODE_SIZE]), trainable = trainable)
        fc2_hidden = tf.nn.elu(tf.matmul(fc1_hidden, fc2_weight) + fc2_bias)

        print("code layer shape : %s" % fc2_hidden.get_shape())

        dfc1_weight = tf.Variable(tf.truncated_normal([CODE_SIZE, 512], stddev = 0.02), trainable = trainable)
        dfc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        dfc1_hidden = tf.nn.elu(tf.matmul(fc2_hidden, dfc1_weight) + dfc1_bias)

        dfc2_weight = tf.Variable(tf.truncated_normal([512, 10*10*32], stddev = 0.02), trainable = trainable)
        dfc2_bias = tf.Variable(tf.constant(0.02, shape = [10*10*32]), trainable = trainable)
        dfc2_hidden = tf.nn.sigmoid(tf.matmul(dfc1_hidden, dfc2_weight) + dfc2_bias)

        dfc2_hidden_conv = tf.reshape(dfc2_hidden, [-1, 10, 10, 32])
        #dfc2_hidden_conv = conv3_hidden

        dconv1_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        dconv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        dconv1_output_shape = tf.stack([tf.shape(x)[0], 10, 10, 32])        
        dconv1_hidden = tf.nn.elu(tf.nn.conv2d_transpose(dfc2_hidden_conv, dconv1_weight, dconv1_output_shape, strides = [1,1,1,1], padding='SAME') + dconv1_bias)

        dconv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        dconv2_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)
        dconv2_output_shape = tf.stack([tf.shape(x)[0], 20, 20, 16])        
        dconv2_hidden = tf.nn.elu(tf.nn.conv2d_transpose(dconv1_hidden, dconv2_weight, dconv2_output_shape, strides = [1,2,2,1], padding='SAME') + dconv2_bias)

        dconv3_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 16], stddev = 0.02), trainable = trainable)
        dconv3_bias = tf.Variable(tf.constant(0.02, shape = [4]), trainable = trainable)
        dconv3_output_shape = tf.stack([tf.shape(x)[0], 80, 80, 4])       
        dconv3_hidden = tf.nn.elu(tf.nn.conv2d_transpose(dconv2_hidden, dconv3_weight, dconv3_output_shape, strides = [1,4,4,1], padding='SAME') + dconv3_bias)

        return dconv3_hidden, fc2_hidden

    def update(self, sess, ob0):
        sess.run(self.train_step, feed_dict={self.first_observation : ob0})

def main(_):
    # make game eviornment
    env = Game.GameState()

    # The replay memory
    replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    cae = CAE()
    code_set = set()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1000)

    # Populate the replay buffer
    observation = Game.GameState()
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing)                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)

    episode_reward = 0    
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= 0.1:
            action = 1
            actions[action] = 1
        else:
            action = 0
            actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((next_state))

        episode_reward += reward

        # Current game episode is over
        if done:
            observation = Game.GameState()
            do_nothing = np.zeros(2)
            do_nothing[0] = 1
            observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
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
        observation = Game.GameState()
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
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
            actions = np.zeros([2])
            if random.random() <= 0.1:
                action = 1
                actions[action] = 1
            else:
                action = 0
                actions[action] = 1
            # execute the action
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            code_set.add(code_converter(cae.code.eval(feed_dict={cae.first_observation: [next_state]}))[0]) 
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # save the transition to replay buffer
            replay_memory.append((next_state))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                state_batch = [sample for sample in samples]
                if total_t % 1000 == 0:
                    print("step %d, loss %g"%(total_t, cae.loss.eval(feed_dict={cae.first_observation: state_batch})))
                    print(code_converter(cae.code.eval(feed_dict={cae.first_observation: state_batch})))
                    print("Code Set: ", len(code_set))
                    #if total_t % 10000 == 0:
                    #    test_size = 10
                    #    test_reconstruct_img = np.reshape(cae.target_output.eval(feed_dict = {cae.first_observation: state_batch}), [-1, 80, 80, 4])
                    #    plot_n_reconstruct(state_batch, test_reconstruct_img)

			    # Update network
                cae.update(sess, state_batch)

            if done:
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t)
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()