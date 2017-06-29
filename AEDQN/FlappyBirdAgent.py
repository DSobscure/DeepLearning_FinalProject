import itertools
import numpy as np
import random
import sys
import tensorflow as tf
from collections import deque
from AEDQN import AEDQN
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
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 50000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

SCORE_LOG_SIZE = 100

def process_state(state):
    state = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return np.array(state)

def main(_):
    # make game eviornment
    env = Game.GameState()

    # The replay memory
    replay_memory = deque()
    log = deque()
    log.append(0)

    # Behavior Network & Target Network
    dqn = AEDQN(2, GAMMA)

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

    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= epsilon:
            action = np.random.randint(2)
        else:
            action = dqn.select_action(sess, state)
        actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((state, action, reward, next_state, done))

        episode_reward += reward

        # Current game episode is over
        if done:
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
                action = dqn.select_action(sess, state)
            actions[action] = 1
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            # save the transition to replay buffer
            replay_memory.append((state, action, reward, next_state, done))
            if len(replay_memory) > REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)

                state_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                next_state_batch = [sample[3] for sample in samples]
                done_batch = [sample[4] for sample in samples]

			    # Update network

                loss = dqn.update(sess, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                if total_t % 1000 == 0:
                    print('dqn loss: ', loss)

            # Update target network every FREQ_UPDATE_TARGET_Q steps
            if total_t % FREQ_UPDATE_TARGET_Q == 0:
                dqn.update_target_network(sess)

            if done:
                log.append(episode_reward);
                if len(log) > SCORE_LOG_SIZE:
                    log.popleft();
                print ("Episode reward: ", episode_reward, "100 mean:", np.mean(log), 'episode = ', episode, 'total_t = ', total_t, 'epsilon: ', epsilon)
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()