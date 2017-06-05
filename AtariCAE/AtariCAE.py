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
EXPLORE_STPES = 500000

# replay memory
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 10000

BATCH_SIZE = 32
CODE_SIZE = 16

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

def random_code():
    result = np.zeros(CODE_SIZE)
    for i in range(CODE_SIZE):
        result[i] = np.random.randint(2)
    return result

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state)

def batch_norm(x, scope, is_training = True, epsilon=0.001, decay=0.99):
    return x

class CAE():
    def __init__(self):
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.state2 = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state2')
        self.state_code = tf.placeholder(shape=[None, CODE_SIZE], dtype=tf.float32, name='state_code')
        self.isTranning = tf.placeholder(tf.bool)
        self.code_output, self.code_output2 = self.build_network(self.state, self.state2, trainable=True)            

        self.code_loss = tf.reduce_mean(tf.pow(self.code_output - self.state_code, 2))
        self.optimize_code = tf.train.RMSPropOptimizer(0.00025).minimize(self.code_loss)   

        self.code_loss2 = tf.reduce_mean(tf.pow(self.code_output - self.code_output2, 2))
        self.optimize_code2 = tf.train.RMSPropOptimizer(0.00025).minimize(self.code_loss2)   

    def build_network(self, x, x2, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias      
        conv1_hidden_bn = batch_norm(conv1_hidden_sum, 'conv1_hidden_bn', self.isTranning)        
        conv1_hidden = tf.nn.elu(conv1_hidden_bn)

        conv1_hidden_sum2 = tf.nn.conv2d(x2, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias
        conv1_hidden_bn2 = batch_norm(conv1_hidden_sum2, 'conv1_hidden_bn2', self.isTranning)
        conv1_hidden2 = tf.nn.elu(conv1_hidden_bn2)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias
        conv2_hidden_bn = batch_norm(conv2_hidden_sum, 'conv2_hidden_bn', self.isTranning)
        conv2_hidden = tf.nn.elu(conv2_hidden_bn)

        conv2_hidden_sum2 = tf.nn.conv2d(conv1_hidden2, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias
        conv2_hidden_bn2 = batch_norm(conv2_hidden_sum2, 'conv2_hidden_bn2', self.isTranning)
        conv2_hidden2 = tf.nn.elu(conv2_hidden_bn2)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden_bn = batch_norm(conv3_hidden_sum, 'conv3_hidden_bn', self.isTranning)
        conv3_hidden = tf.nn.elu(conv3_hidden_bn)

        conv3_hidden_sum2 = tf.nn.conv2d(conv2_hidden2, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden_bn2 = batch_norm(conv3_hidden_sum2, 'conv3_hidden_bn2', self.isTranning)
        conv3_hidden2 = tf.nn.elu(conv3_hidden_bn2)

        fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, CODE_SIZE], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [CODE_SIZE]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden_bn = batch_norm(fc1_hidden_sum, 'fc1_hidden_bn', self.isTranning)
        fc1_hidden = tf.nn.sigmoid(fc1_hidden_bn)

        conv3_hidden_flat2 = tf.reshape(conv3_hidden2, [-1, 11*11*64])
        fc1_hidden_sum2 = tf.matmul(conv3_hidden_flat2, fc1_weight) + fc1_bias
        fc1_hidden_bn2 = batch_norm(fc1_hidden_sum2, 'fc1_hidden_bn2', self.isTranning)
        fc1_hidden2 = tf.nn.sigmoid(fc1_hidden_bn2)

        print("code layer shape : %s" % fc1_hidden.get_shape())

        return fc1_hidden, fc1_hidden2

    def update_code(self, sess, state, code, next_state):
        sess.run([self.optimize_code, self.optimize_code2], feed_dict={self.state : state, self.state_code: code, self.state2: next_state, self.isTranning: True})

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")
    qValue = np.array([np.random.uniform(size=2 ** CODE_SIZE), np.random.uniform(size=2 ** CODE_SIZE), np.random.uniform(size=2 ** CODE_SIZE), np.random.uniform(size=2 ** CODE_SIZE)])

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
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)
    random_state_code = random_code()
    initial_state = state

    episode_reward = 0    
    epsilon = INITIAL_EPSILON

    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = random.randrange(4)

        next_observation, reward, done, _ = env.step(action)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        random_next_state_code = random_code()
        replay_memory.append((state, random_state_code, next_state, random_next_state_code))

        episode_reward += reward

        # Current game episode is over
        if done:
            observation = env.reset()                 # retrive first env image
            observation = process_state(observation)
            state = np.stack([observation] * 4, axis=2)
            random_state_code = random_code()

            print ("Episode reward: ", episode_reward, "Buffer: ", len(replay_memory))
            episode_reward = 0
        # Not over yet
        else:
            state = next_state
            random_state_code = random_next_state_code

    # total steps
    total_t = 0

    for episode in range(1000000):

        # Reset the environment
        observation = env.reset()                 # retrive first env image
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        random_state_code = random_code()
        state_code = code_converter(cae.code_output.eval(feed_dict={cae.state: [state], cae.isTranning: False}))[0]   
        episode_reward = 0                              # store the episode reward
        '''
        How to update episode reward:
        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        episode_reward += reward
        '''
        episode_replay = []

        for t in itertools.count():

            #choose a action
            #if total_t < 20000:
            #    print(state_code)
            #else:
            #    print([value[state_code] for value in qValue]) 
            if random.random() <= epsilon:
                action = random.randrange(4)
            else:    
                action = np.argmax([value[state_code] for value in qValue])
            if epsilon > FINAL_EPSILON:
                epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            random_next_state_code = random_code()
            next_state_code = code_converter(cae.code_output.eval(feed_dict={cae.state: [next_state], cae.isTranning: False}))[0]      
            episode_reward += reward

            episode_replay.append((state_code, action, reward, done, next_state_code));

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # save the transition to replay buffer
            replay_memory.append((state, random_state_code, next_state, random_next_state_code))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 1 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                state_batch = [sample[0] for sample in samples]
                state_code_batch = [sample[1] for sample in samples]
                next_state_batch = [sample[2] for sample in samples]
                if total_t % 1000 == 0:
                    print("Code Set: ", len(code_set))

                    print(code_converter(cae.code_output.eval(feed_dict={cae.state: state_batch, cae.isTranning: False})))
                    print(code_converter(cae.code_output.eval(feed_dict={cae.state: [initial_state], cae.isTranning: False})))
			    # Update network
                if total_t > 20000:
                    code_set.add(state_code)    
                    env.render()
                if total_t > 20000 and done:
                    episode_replay.reverse()
                    for replay in episode_replay:
                        replay_state_code = replay[0]
                        replay_action = replay[1]
                        replay_reward = replay[2]
                        replay_done = replay[3]
                        replay_next_state_code = replay[4]
                        if replay_done:
                            qValue[replay_action][replay_state_code] += 0.9 * (replay_reward + 0 - qValue[replay_action][replay_state_code])
                        else:
                            next_max = GAMMA * np.max([value[replay_next_state_code] for value in qValue])
                            qValue[replay_action][replay_state_code] += 0.9 * (replay_reward + next_max - qValue[replay_action][replay_state_code])
                #cae.update_state(sess, state_batch)
                if total_t < 20000:
                    cae.update_code(sess, state_batch, state_code_batch, next_state_batch)

            if done:
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t)
                break

            state = next_state
            state_code = next_state_code
            random_state_code = random_next_state_code
            total_t += 1


if __name__ == '__main__':
    tf.app.run()