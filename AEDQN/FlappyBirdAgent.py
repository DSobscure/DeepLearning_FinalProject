import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import sys
import math
from AEDQN import AEDQN
sys.path.append("game/")
import wrapped_flappy_bird as Game

GAMMA = 0.99

INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001
EXPLORE_STPES = 100000

# replay memory
INIT_REPLAY_MEMORY_SIZE =10000
REPLAY_MEMORY_SIZE = 100000

BATCH_SIZE = 32

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
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state)

def main(_):
    env = Game.GameState()
    dqn = AEDQN(GAMMA, 2)

    replay_memory = deque()
    log = deque()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    epsilon = INITIAL_EPSILON
    total_t = 0

    episode_reward = 0
    observation = Game.GameState()
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing)                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)

    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= epsilon:
            action = np.random.randint(2)
            actions[action] = 1
        else:    
            action = 0
            actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        
        replay_memory.append((state, action, reward, done, next_state))

        episode_reward += reward
        
        # Current game episode is over
        if done:
            observation = Game.GameState()
            do_nothing = np.zeros(2)
            do_nothing[0] = 1
            observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
            observation = process_state(observation)
            state = np.stack([observation] * 4, axis=2)
            log.append(episode_reward)
            if len(log) > 100:
                log.popleft()
            print ("Episode reward: ", episode_reward, '100 mean: ', np.mean(log), ' dev: ', np.std(log), " Buffer: ", len(replay_memory))
            episode_reward = 0
        else:
            state = next_state

    for episode in range(1000000):
        observation = Game.GameState()
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0
        episode_replay_memory = []

        for t in itertools.count():
            actions = np.zeros([2])    
            if random.random() <= epsilon:
                action = np.random.randint(2)
            else:    
                action = dqn.select_action(state)
            actions[action] = 1
            
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            episode_reward += reward

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES
            
            episode_replay_memory.append((state, action, reward, done, next_state))

            if len(replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 8 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                state_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_batch = [sample[4] for sample in samples]

                dqn_loss, rec_loss = dqn.update(sess, state_batch, action_batch, reward_batch, done_batch, next_state_batch)

                if total_t % 1000 == 0:
                    print("dqn loss:", dqn_loss)   
                    print("rec loss:", rec_loss)           
            if total_t % 10000 == 0:
                dqn.update_target_network(sess)

            if done or t > 10000:     
                average = np.mean(log)
                deviation = np.std(log) + 0.01
                for episode_replay in episode_replay_memory:
                    _state, _action, _reward, _done, _next_state = episode_replay
                    transfer_reward = _reward * (1 + elu((episode_reward - average) / deviation)) if (_reward >= 0) else _reward * (1 - inverse_elu((episode_reward - average) / deviation))
                    if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                        replay_memory.popleft();
                    replay_memory.append((_state, _action, transfer_reward, _done, _next_state));
                               
                log.append(episode_reward)
                if len(log) > 100:
                    log.popleft()
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t, '100 mean: ', np.mean(log), ' dev: ', np.std(log))
                total_t += 1
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()