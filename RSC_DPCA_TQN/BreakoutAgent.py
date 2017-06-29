import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from StateCodeGenerator import SCG
from TupleNetwork import TupleNetwork
import gym
import math

GAMMA = 0.99

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORE_STPES = 500000


# replay memory
INIT_REPLAY_MEMORY_SIZE =50000
REPLAY_MEMORY_SIZE = 500000

BATCH_SIZE = 32
CODE_SIZE = 8
CODE_LEVEL = 2

Q_LEARNING_RATE = 0.0025

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state).reshape([84,84,1])

def main(_):
    env = gym.envs.make("Breakout-v0")
    qValue = np.array([TupleNetwork(), TupleNetwork(), TupleNetwork(), TupleNetwork()])

    # The replay memory
    replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    scg = SCG(CODE_SIZE, CODE_LEVEL)
    code_set = set()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    epsilon = INITIAL_EPSILON
    total_t = 0

    for episode in range(1000000):
        observation = env.reset()                       # retrive first env image
        observation = process_state(observation)
        state = np.stack([observation] * 2) 
        state_code = scg.get_code([state])[0]
        episode_reward = 0

        for t in itertools.count():
            if random.random() <= epsilon:
                action = random.randrange(4)
            else:
                action = np.argmax([value.GetValue(state_code) for value in qValue])
            
            # execute the action
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_state(next_observation)
            next_state = np.append(state[1:], np.expand_dims(next_observation, 0), axis=0)
            next_state_code = scg.get_code([next_state])[0]
            code_set.add(state_code)                   
            episode_reward += reward

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES

            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            replay_memory.append((state, action, reward, done, next_state))
            
            if len(replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                state_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_batch = [sample[4] for sample in samples]

                scg.update_code(sess, state_batch)

                state_code_batch = scg.get_code(state_batch)
                next_state_code_batch = scg.get_code(next_state_batch)

                q_loss_sum = 0
                for i in range(BATCH_SIZE):
                    replay_state_code = state_code_batch[i]
                    replay_action = action_batch[i]
                    replay_reward = reward_batch[i]
                    replay_done = done_batch[i]
                    replay_next_state_code = next_state_code_batch[i]
                    if replay_done:
                        q_loss = replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code)
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * q_loss)
                    else:
                        next_max = GAMMA * np.max([value.GetValue(replay_next_state_code) for value in qValue])
                        q_loss = replay_reward + next_max - qValue[replay_action].GetValue(replay_state_code)
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * q_loss)
                    q_loss_sum += abs(q_loss)

                if total_t % 1000 == 0:
                    print("Code Set: ", len(code_set))
                    print("rec loss:", scg.get_loss(state_batch))           
                    print("q loss:", q_loss_sum / BATCH_SIZE)           

            if done:                
                log.append(episode_reward)
                if len(log) > 100:
                    log.popleft()
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t, '100 mean: ', np.mean(log), ' dev: ', np.std(log))
                total_t += 1
                break

            state = next_state
            state_code = next_state_code
            total_t += 1


if __name__ == '__main__':
    tf.app.run()