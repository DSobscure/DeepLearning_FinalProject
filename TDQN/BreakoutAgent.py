import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
from TDQN import TDQN
from TupleNetwork import TupleNetwork

# Hyper Parameters:
GAMMA = 0.99

# Epsilon
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORE_STPES = 500000

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 500000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

MONITOR_PATH = 'breakout_videos/'

# Valid actions for breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
VALID_ACTIONS = [0, 1, 2, 3]

SCORE_LOG_SIZE = 100

CODE_SIZE = 64
Q_LEARNING_RATE = 0.01

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state)

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")
    qValue = np.array([TupleNetwork(), TupleNetwork(), TupleNetwork(), TupleNetwork()])

    # The replay memory
    replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    tdqn = TDQN(len(VALID_ACTIONS), CODE_SIZE, GAMMA)

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times

    epsilon = INITIAL_EPSILON
    episode_reward = 0    
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = None
        if random.random() <= epsilon:
            action = random.randrange(4)
        else:
            action = tdqn.select_action(sess, state)

        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((state, action, reward, next_state, done))

        episode_reward += reward

        # Current game episode is over
        if done:
            observation = env.reset()
            observation = process_state(observation)
            state = np.stack([observation] * 4, axis=2)
            log.append(episode_reward);
            if len(log) > SCORE_LOG_SIZE:
                log.popleft();
            print ("Episode reward: ", episode_reward, "100 mean:", np.mean(log), "Buffer: ", len(replay_memory))
            episode_reward = 0
        # Not over yet
        else:
            state = next_state

    # total steps
    total_t = 0

    for episode in range(TRAINING_EPISODES):

        # Reset the environment
        observation = env.reset()
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward

        for t in itertools.count():
            action = None
            if random.random() <= epsilon:
                action = random.randrange(4)
            else:
                if episode % 2 == 0:
                    action = tdqn.select_action(sess, state)
                else:
                    state_code = tdqn.get_code([state])[0]
                    action = np.argmax([value.GetValue(state_code) for value in qValue])
            if epsilon > FINAL_EPSILON:
                epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # save the transition to replay buffer
            replay_memory.append((state, action, reward, next_state, done))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)

                state_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                next_state_batch = [sample[3] for sample in samples]
                done_batch = [sample[4] for sample in samples]

			    # Update network

                loss = tdqn.update(sess, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                if total_t % 1000 == 0:
                    print('dqn loss: ', loss)

                if total_t % 16 == 0:
                    state_code_batch = tdqn.get_code(state_batch)
                    next_state_code_batch = tdqn.get_code(next_state_batch)
                    loss = 0
                    for i in range(BATCH_SIZE):
                        replay_state_code = state_code_batch[i]
                        replay_action = action_batch[i]
                        replay_reward = reward_batch[i]
                        replay_done = done_batch[i]
                        replay_next_state_code = next_state_code_batch[i]
                        if replay_done:
                            delta = replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code)
                            loss += delta ** 2
                            qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * delta)
                        else:
                            next_max = GAMMA * np.max([value.GetValue(replay_next_state_code) for value in qValue])
                            delta = replay_reward + next_max - qValue[replay_action].GetValue  (replay_state_code)
                            loss += delta ** 2
                            qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * delta)
                    loss /= BATCH_SIZE
                    if total_t % 1000 == 0:
                        print('tn loss: ', loss)
            # Update target network every FREQ_UPDATE_TARGET_Q steps
            if total_t % FREQ_UPDATE_TARGET_Q == 0:
                tdqn.update_target_network(sess)

            if done:
                log.append(episode_reward);
                if len(log) > SCORE_LOG_SIZE:
                    log.popleft();
                print ("Episode reward: ", episode_reward, "100 mean:", np.mean(log), 'episode = ', episode, 'total_t = ', total_t, 'epsilon: ', epsilon)
                with open('tranningResult', 'a') as file:
                    file.writelines(str(episode) + "\t" + str(total_t) + "\t" + str(episode_reward) + "\n")
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()