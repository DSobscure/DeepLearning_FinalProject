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

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORE_STPES = 200000

# replay memory
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 1000000

BATCH_SIZE = 32

CODE_SIZE = 16

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

def main(_):
    # make game eviornment
    env = Game.GameState()
    qValue = np.array([np.zeros(2 ** CODE_SIZE), np.zeros(2 ** CODE_SIZE)])

    # The replay memory
    replay_memory = deque()

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
    initial_state = state

    episode_reward = 0    
    epsilon = INITIAL_EPSILON
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        action = random.randrange(2)
        actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((state, random_code(), action, reward, next_state, done))

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

    for episode in range(10000):

        # Reset the environment
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward

        for t in itertools.count():
            # choose a action
            actions = np.zeros([2])         
            if random.random() <= epsilon:
                action = random.randrange(2)
                actions[action] = 1
            else:    
                code = code_converter(cg.cg_code.eval(feed_dict={cg.observation1: [state]}))[0]   
                code_set.add(code)         
                action = np.argmax([value[code] for value in qValue])
                actions[action] = 1
            if epsilon > FINAL_EPSILON:
                epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # save the transition to replay buffer
            replay_memory.append((state, random_code(), action, reward, next_state, done))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                if total_t < 20000:
                    state_batch = [sample[0] for sample in samples]
                    state_code_batch = [sample[1] for sample in samples]
                    if total_t % 1000 == 0:
                        print("step %d, loss %g"%(total_t, cg.loss.eval(feed_dict={cg.observation1: state_batch, cg.state_code: state_code_batch})))
                        print(code_converter(cg.cg_code.eval(feed_dict={cg.observation1: [initial_state]})))
                        print(code_converter(cg.cg_code.eval(feed_dict={cg.observation1: state_batch})))
			        # Update network
                    cg.update(sess, state_batch, state_code_batch)
                else:
                    if total_t % 1000 == 0:
                        print("Code Set: ", len(code_set))
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