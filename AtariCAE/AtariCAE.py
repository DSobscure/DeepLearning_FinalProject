import gym
import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

EXPLORE_STPES = 500000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 1000000

BATCH_SIZE = 32
CODE_SIZE = 20

def plot_n_reconstruct(origin_img, reconstruct_img, n = 8):
    plt.close()
    plt.figure(figsize=(2 * n, 4))

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
    plt.show(block=False)

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
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state)

def batch_norm(x, scope, is_training = True, epsilon=0.001, decay=0.99):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=True)

class CAE():
    def __init__(self):
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.state_output, self.code_output = self.build_network(self.state, trainable=True)            
        self.state_loss = tf.reduce_mean(tf.pow(self.state_output - self.state, 2))        
        self.optimize_state = tf.train.RMSPropOptimizer(0.00025).minimize(self.state_loss)   

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias
        conv1_hidden_bn = batch_norm(conv1_hidden_sum, 'conv1_hidden_bn')
        conv1_hidden = tf.nn.elu(conv1_hidden_bn)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias
        conv2_hidden_bn = batch_norm(conv2_hidden_sum, 'conv2_hidden_bn')
        conv2_hidden = tf.nn.elu(conv2_hidden_bn)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden_bn = batch_norm(conv3_hidden_sum, 'conv3_hidden_bn')
        conv3_hidden = tf.nn.elu(conv3_hidden_bn)

        fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden_bn = batch_norm(fc1_hidden_sum, 'fc1_hidden_bn')
        fc1_hidden = tf.nn.sigmoid(fc1_hidden_bn)

        fc2_weight = tf.Variable(tf.truncated_normal([512, CODE_SIZE], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.truncated_normal([CODE_SIZE], stddev = 0.02), trainable = trainable)
        fc2_hidden_sum = tf.matmul(fc1_hidden, fc2_weight) + fc2_bias
        fc2_hidden_bn = batch_norm(fc2_hidden_sum, 'fc2_hidden_bn')
        fc2_hidden = tf.nn.sigmoid(fc2_hidden_bn)

        print("code layer shape : %s" % fc2_hidden.get_shape())

        dfc1_weight = tf.Variable(tf.truncated_normal([CODE_SIZE, 512], stddev = 0.02), trainable = trainable)
        dfc1_bias = tf.Variable(tf.truncated_normal([512], stddev = 0.02), trainable = trainable)
        dfc1_hidden_sum = tf.matmul(fc2_hidden, dfc1_weight) + dfc1_bias
        dfc1_hidden_bn = batch_norm(dfc1_hidden_sum, 'dfc1_hidden_bn')
        dfc1_hidden = tf.nn.elu(dfc1_hidden_bn)

        dfc2_weight = tf.Variable(tf.truncated_normal([512, 11*11*64], stddev = 0.02), trainable = trainable)
        dfc2_bias = tf.Variable(tf.constant(0.02, shape = [11*11*64]), trainable = trainable)
        dfc2_hidden_sum = tf.matmul(dfc1_hidden, dfc2_weight) + dfc2_bias
        dfc2_hidden_bn = batch_norm(dfc2_hidden_sum, 'dfc2_hidden_bn')
        dfc2_hidden = tf.nn.elu(dfc2_hidden_bn)

        dfc2_hidden_conv = tf.reshape(dfc2_hidden, [-1, 11, 11, 64])
        #dfc2_hidden_conv = conv3_hidden

        dconv1_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        dconv1_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        dconv1_output_shape = tf.stack([tf.shape(x)[0], 11, 11, 64])      
        dconv1_hidden_sum = tf.nn.conv2d_transpose(dfc2_hidden_conv, dconv1_weight, dconv1_output_shape, strides = [1,1,1,1], padding='SAME') + dconv1_bias
        dconv1_hidden_bn = batch_norm(dconv1_hidden_sum, 'dconv1_hidden_bn')
        dconv1_hidden = tf.nn.elu(dconv1_hidden_bn)

        dconv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        dconv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        dconv2_output_shape = tf.stack([tf.shape(x)[0], 21, 21, 32])        
        dconv2_hidden_sum = tf.nn.conv2d_transpose(dconv1_hidden, dconv2_weight, dconv2_output_shape, strides = [1,2,2,1], padding='SAME') + dconv2_bias
        dconv2_hidden_bn = batch_norm(dconv2_hidden_sum, 'dconv2_hidden_bn')
        dconv2_hidden = tf.nn.elu(dconv2_hidden_bn)


        dconv3_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        dconv3_bias = tf.Variable(tf.constant(0.02, shape = [4]), trainable = trainable)
        dconv3_output_shape = tf.stack([tf.shape(x)[0], 84, 84, 4])       
        dconv3_hidden_sum = tf.nn.conv2d_transpose(dconv2_hidden, dconv3_weight, dconv3_output_shape, strides = [1,4,4,1], padding='SAME') + dconv3_bias
        dconv3_hidden_bn = batch_norm(dconv3_hidden_sum, 'dconv3_hidden_bn')
        dconv3_hidden = tf.nn.elu(dconv3_hidden_bn)

        return dconv3_hidden, fc2_hidden

    def update_state(self, sess, state):
        sess.run(self.optimize_state, feed_dict={self.state : state})

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

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times
    initial_state = state

    episode_reward = 0    
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = random.randrange(4)

        next_observation, reward, done, _ = env.step(action)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((state, 0))

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

    for episode in range(1000000):

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
            replay_memory.append((state, 0))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                state_batch = [sample[0] for sample in samples]
                if total_t % 1000 == 0:
                    print("step %d, state loss %g"%(total_t, cae.state_loss.eval(feed_dict={cae.state: state_batch})))
                    if total_t % 10000 == 0:
                        test_reconstruct_img = np.reshape(cae.state_output.eval(feed_dict = {cae.state: state_batch}), [-1, 84, 84, 4])
                        plot_n_reconstruct(state_batch, test_reconstruct_img)

                    print(code_converter(cae.code_output.eval(feed_dict={cae.state: state_batch})))
                    print(code_converter(cae.code_output.eval(feed_dict={cae.state: [initial_state]})))
			    # Update network
                cae.update_state(sess, state_batch)

            if done:
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t)
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()