import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game

GAMMA = 0.99

INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EXPLORE_STPES = 100000

BATCH_SIZE = 32

CODE_SIZE = 20

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
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return np.array(state)

class CG():
    def __init__(self):
        with tf.variable_scope('inputs'):
            self.observation1 = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name='first_observation')
            self.state_code = tf.placeholder(shape=[None, CODE_SIZE], dtype=tf.float32, name='state_code')
        with tf.variable_scope('target_model'):
            self.cg_code = self.build_network(self.observation1, trainable=True)
        self.loss = tf.reduce_sum(tf.pow(self.state_code - self.cg_code, 2))

        self.optimize_op = tf.train.RMSPropOptimizer(1e-6).minimize(self.loss)   

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.01), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.01, shape = [32]), trainable = trainable)
        conv1_hidden = tf.nn.relu(tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias)	
        conv1_hidden_pool = tf.nn.max_pool(conv1_hidden, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.01), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.01, shape = [64]), trainable = trainable)
        conv2_hidden = tf.nn.relu(tf.nn.conv2d(conv1_hidden_pool, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.01, shape = [64]), trainable = trainable)        
        conv3_hidden = tf.nn.relu(tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias)

        fc1_weight = tf.Variable(tf.truncated_normal([1600, 512], stddev = 0.01), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.01, shape = [512]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 1600])
        fc1_hidden = tf.nn.relu(tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias)

        fc2_weight = tf.Variable(tf.truncated_normal([512, CODE_SIZE], stddev = 0.01), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.01, shape = [CODE_SIZE]), trainable = trainable)
        fc2_hidden = tf.nn.sigmoid(tf.matmul(fc1_hidden, fc2_weight) + fc2_bias)

        print("code layer shape : %s" % fc2_hidden.get_shape())

        return fc2_hidden

    def update(self, sess, state, state_code):
        sess.run(self.optimize_op, feed_dict={self.observation1 : state, self.state_code: state_code})

    def train(self, sess, env, coder_replay_memory):
        for i in range(10000):
            samples = random.sample(coder_replay_memory, BATCH_SIZE)
            state_batch = [sample[0] for sample in samples]
            state_code_batch = [sample[1] for sample in samples]
            if i % 1000 == 0:
                print(code_converter(self.cg_code.eval(feed_dict={self.observation1: state_batch})))
		    # Update network
            self.update(sess, state_batch, state_code_batch)

def main(_):
    # make game eviornment
    env = Game.GameState()
    qValue = np.array([np.zeros(2 ** CODE_SIZE), np.zeros(2 ** CODE_SIZE)])

    # The replay memory
    coder_replay_memory = deque()

    # Behavior Network & Target Network
    cg = CG()
    code_set = set()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times

    episode_reward = 0    
    epsilon = INITIAL_EPSILON
    while len(coder_replay_memory) < 5000:
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
        coder_replay_memory.append((state, random_code()))

        episode_reward += reward

        # Current game episode is over
        if done:
            observation = Game.GameState()
            do_nothing = np.zeros(2)
            do_nothing[0] = 1
            observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
            observation = process_state(observation)
            state = np.stack([observation] * 4, axis=2)

            print ("Episode reward: ", episode_reward, "Buffer: ", len(coder_replay_memory))
            episode_reward = 0
        # Not over yet
        else:
            state = next_state
    cg.train(sess, env, coder_replay_memory)
    coder_replay_memory.clear();
    
    # total steps
    total_t = 0

    for episode in range(100000):

        # Reset the environment
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        state_code = code_converter(cg.cg_code.eval(feed_dict={cg.observation1: [state]}))[0]   
        episode_reward = 0                              # store the episode reward

        for t in itertools.count():
            # choose a action
            actions = np.zeros([2])    
            print([value[state_code] for value in qValue]) 
            if t < 10000:
                if random.random() <= 0.1:
                    action = 1
                    actions[action] = 1
                else:
                    action = 0
                    actions[action] = 1
            else:
                if random.random() <= epsilon:
                    if random.random() <= 0.1:
                        action = 1
                        actions[action] = 1
                    else:
                        action = 0
                        actions[action] = 1
                else:    
                    action = np.argmax([value[state_code] for value in qValue])
                    actions[action] = 1
                if epsilon > FINAL_EPSILON:
                    epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            episode_reward += reward

            coder_replay_memory.append((next_state, random_code()))
            next_state_code = code_converter(cg.cg_code.eval(feed_dict={cg.observation1: [next_state]}))[0]   
            code_set.add(next_state_code)    

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)

            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 1 == 0:                
                if total_t % 1000 == 0:
                    print("Code Set: ", len(code_set))

                if done:
                    qValue[action][state_code] += 0.25 * (reward + 0 - qValue[action][state_code])
                else:
                    next_max = GAMMA * np.max([value[next_state_code] for value in qValue])
                    qValue[action][state_code] += 0.25 * (reward + next_max - qValue[action][state_code])

            if done:
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t)
                break

            if len(coder_replay_memory) > 50000:
                cg.train(sess, env, coder_replay_memory)
                coder_replay_memory.clear()
                code_set.clear()
                qValue = np.array([np.zeros(2 ** CODE_SIZE), np.zeros(2 ** CODE_SIZE)])

            state = next_state
            state_code = next_state_code
            total_t += 1


if __name__ == '__main__':
    tf.app.run()