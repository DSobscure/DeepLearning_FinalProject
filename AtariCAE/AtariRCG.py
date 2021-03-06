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
FINAL_EPSILON = 0.01
EXPLORE_STPES = 50000

# replay memory
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 1000000

BATCH_SIZE = 32

CODE_SIZE = 6

def plot_n_reconstruct(origin_img, n = 10):

    plt.figure(figsize=(10, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(origin_img[i].reshape(84, 84, 4))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def random_code():
    result = np.zeros(CODE_SIZE)
    for i in range(CODE_SIZE):
        result[i] = np.random.randint(2)
    return result

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

class CG():
    def __init__(self):
        with tf.variable_scope('inputs'):
            self.observation1 = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='first_observation')
            self.state_code = tf.placeholder(shape=[None, CODE_SIZE], dtype=tf.float32, name='state_code')
        with tf.variable_scope('target_model'):
            self.cg_code = self.build_network(self.observation1, trainable=True)
        self.loss = tf.reduce_sum(tf.pow(self.state_code - self.cg_code, 2))

        self.optimize_op = tf.train.RMSPropOptimizer(0.000025, momentum=0.95, epsilon=0.01).minimize(self.loss)   

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

        fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])
        fc1_hidden = tf.nn.relu(tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias)

        fc2_weight = tf.Variable(tf.truncated_normal([512, CODE_SIZE], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [CODE_SIZE]), trainable = trainable)
        fc2_hidden = tf.nn.sigmoid(tf.matmul(fc1_hidden, fc2_weight) + fc2_bias)

        print("code layer shape : %s" % fc2_hidden.get_shape())

        return fc2_hidden

    def update(self, sess, state, state_code):
        sess.run(self.optimize_op, feed_dict={self.observation1 : state, self.state_code: state_code})

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")
    qValue = np.array([np.zeros(2 ** CODE_SIZE), np.zeros(2 ** CODE_SIZE), np.zeros(2 ** CODE_SIZE), np.zeros(2 ** CODE_SIZE)])

    # The replay memory
    replay_memory = deque()

    # Behavior Network & Target Network
    cg = CG()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times
    initial_state = state

    episode_reward = 0    
    epsilon = INITIAL_EPSILON
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = random.randrange(4)

        next_observation, reward, done, _ = env.step(action)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((state, random_code(), action, reward, next_state, done))

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

        for t in itertools.count():
            # choose a action
            action = None
            if random.random() <= epsilon:
                action = random.randrange(4)
            else:
                code = code_converter(cg.cg_code.eval(feed_dict={cg.observation1: [state]}))[0]
                action = np.argmax([value[code] for value in qValue])
            if epsilon > FINAL_EPSILON:
                epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            if total_t > 50000:
                env.render()
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # save the transition to replay buffer
            replay_memory.append((state, random_code(), action, reward, next_state, done))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                if total_t < 10000:
                    state_batch = [sample[0] for sample in samples]
                    state_code_batch = [sample[1] for sample in samples]
                    if total_t % 1000 == 0:
                        print("step %d, loss %g"%(total_t, cg.loss.eval(feed_dict={cg.observation1: state_batch, cg.state_code: state_code_batch})))
                        print(code_converter(cg.cg_code.eval(feed_dict={cg.observation1: [initial_state]})))
                        print(code_converter(cg.cg_code.eval(feed_dict={cg.observation1: state_batch})))
			        # Update network
                    cg.update(sess, state_batch, state_code_batch)
                else:
                    state_batch = [sample[0] for sample in samples]
                    action_batch = [sample[2] for sample in samples]
                    reward_batch = [sample[3] for sample in samples]
                    next_state_batch = [sample[4] for sample in samples]
                    done_batch = [sample[5] for sample in samples]   
                    
                    state_code_batch = code_converter(cg.cg_code.eval(feed_dict={cg.observation1: state_batch}))
                    next_state_code_batch = code_converter(cg.cg_code.eval(feed_dict={cg.observation1: next_state_batch}))

                    for i in range(BATCH_SIZE):
                        if done_batch[i]:
                            qValue[action_batch[i]][state_code_batch[i]] += 0.0025 * (reward_batch[i] - qValue[action_batch[i]][state_code_batch[i]])
                        else:
                            next_max = np.max([value[next_state_code_batch[i]] for value in qValue])
                            qValue[action_batch[i]][state_code_batch[i]] += 0.0025 * (reward_batch[i] + next_max - qValue[action_batch[i]][state_code_batch[i]])

            if done:
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t)
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()