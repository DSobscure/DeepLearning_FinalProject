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
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game
import math

# Hyper Parameters:
GAMMA = 0.99

# Epsilon
INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001
EXPLORE_STPES = 500000

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 500000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

MONITOR_PATH = 'breakout_videos/'

# Valid actions for breakout: ['NOOP', 'JUMP']
VALID_ACTIONS = [0, 1]

SCORE_LOG_SIZE = 100

CODE_SIZE = 64
Q_LEARNING_RATE = 0.01

def elu(value):
    if value >= 0:
        return value
    else:
        return math.exp(value) - 1

def inverse_elu(value):
    if value < 0:
        return value
    else:
        return -math.exp(-value) + 1

def process_state(state):
    state = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return np.array(state)

def main(_):
    # make game eviornment
    env = Game.GameState()
    qValue = np.array([TupleNetwork(), TupleNetwork()])

    # The replay memory
    replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    tdqn = TDQN(len(VALID_ACTIONS), CODE_SIZE, GAMMA)

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = Game.GameState()
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing) 
    observation = process_state(observation)
    state = np.stack([observation] * 4, axis=2) 

    epsilon = INITIAL_EPSILON
    episode_reward = 0    
    episode_replay_memory = []

    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= epsilon:
            action = np.random.randint(2)
        else:
            action = tdqn.select_action(sess, state)
        actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        episode_replay_memory.append((state, action, reward, next_state, done))

        episode_reward += reward

        # Current game episode is over
        if done:
            average = np.mean(log)
            deviation = np.std(log) + 0.01
            for episode_replay in episode_replay_memory:
                _state, _action, _reward, _next_state, _done = episode_replay
                transfer_reward = _reward * (1 + elu((episode_reward - average) / deviation)) if (_reward >= 0) else _reward * (1 - inverse_elu((episode_reward - average) / deviation))
                replay_memory.append((_state_code, _action, transfer_reward, _done, _next_state_code));
            observation = Game.GameState()
            do_nothing = np.zeros(2)
            do_nothing[0] = 1
            observation, _, _ = env.frame_step(do_nothing)
            observation = process_state(observation)
            state = np.stack([observation] * 4, axis=2)
            log.append(episode_reward);
            if len(log) > SCORE_LOG_SIZE:
                log.popleft();
            print ("Episode reward: ", episode_reward, "100 mean:", np.mean(log), "Buffer: ", len(replay_memory))
            episode_reward = 0
            episode_replay_memory = []
        # Not over yet
        else:
            state = next_state

    # total steps
    total_t = 0

    for episode in range(TRAINING_EPISODES):

        episode_replay_memory = []

        # Reset the environment
        observation = Game.GameState()
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward

        for t in itertools.count():
            actions = np.zeros([2])
            if random.random() <= epsilon:
                action = np.random.randint(2)
            else:
                if episode % 2 == 0:
                    action = tdqn.select_action(sess, state)
                else:
                    state_code = tdqn.get_code([state])[0]
                    action = np.argmax([value.GetValue(state_code) for value in qValue])
            actions[action] = 1
            if epsilon > FINAL_EPSILON:
                epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            # save the transition to replay buffer
            episode_replay_memory.append((state, action, reward, next_state, done))
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
                average = np.mean(log)
                deviation = np.std(log) + 0.01
                for episode_replay in episode_replay_memory:
                    _state, _action, _reward, _next_state, _done = episode_replay
                    transfer_reward = _reward * (1 + elu((episode_reward - average) / deviation)) if (_reward >= 0) else _reward * (1 - inverse_elu((episode_reward - average) / deviation))
                    if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                        replay_memory.popleft();
                    replay_memory.append((_state_code, _action, transfer_reward, _done, _next_state_code));
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