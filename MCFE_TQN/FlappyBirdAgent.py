import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from StateCodeGenerator import SCG
from TupleNetwork import TupleNetwork
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game
import math

GAMMA = 0.99

INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001
EXPLORE_STPES = 250000
ENCODING_STEPS = 50000
INITIAL_LIFE_STPES = 500000
LIFE_STPES_INCREASE_FACTOR = 10

INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 100000

BATCH_SIZE = 32

CODE_SIZE = 24

Q_LEARNING_RATE = 0.0025

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

def random_code():
    result = np.zeros(CODE_SIZE)
    for i in range(CODE_SIZE):
        result[i] = np.random.randint(2)
    return result

def process_state(state):
    state = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return np.array(state).reshape([84,84,1])

def get_initial_state(env):
    observation = Game.GameState()
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing)
    return observation

def main(_):
    env = Game.GameState()
    scg = SCG(CODE_SIZE)
    qValue = np.array([TupleNetwork(), TupleNetwork()])

    state_replay_memory = deque()
    rl_replay_memory = deque()
    heritage_replay_memory = deque()
    log = deque()
    code_set = set()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1000)

    episode_reward = 0    
    epsilon = INITIAL_EPSILON
    life_steps = INITIAL_LIFE_STPES

    observation = process_state(get_initial_state(env))
    state = np.stack([observation] * 2)

    while len(state_replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= epsilon:
            action = np.random.randint(2)
        else:    
            action = 0
        actions[action] = 1      

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.array([state[1], next_observation])
        state_replay_memory.append((state, random_code()))
        episode_reward += reward

        if done:
            observation = process_state(get_initial_state(env))
            state = np.stack([observation] * 2)

            log.append(episode_reward)
            if len(log) > 100:
                log.popleft()
            print ("Episode reward: ", episode_reward, '100 mean: ', np.mean(log), ' dev: ', np.std(log), " Buffer: ", len(state_replay_memory))
            episode_reward = 0
        else:
            state = next_state

    for i in range(ENCODING_STEPS):
        samples = random.sample(state_replay_memory, BATCH_SIZE)
        state_batch = [sample[0] for sample in samples]
        random_code_batch = [sample[1] for sample in samples]
        code_loss, difference_code_loss = scg.update_code(sess, state_batch, random_code_batch)
        if i % 1000 == 0:
            print("generate code...", i)
            print("code loss: ", code_loss)
            print("difference code loss: ", difference_code_loss)

    total_t = 0
    for episode in range(1000000):
        episode_reward = 0
        episode_replay_memory = []

        observation = process_state(get_initial_state(env))
        state = np.stack([observation] * 2)
        state_code = scg.get_code([state])[0]

        for t in itertools.count():
            if total_t > life_steps:
                code_set.clear()
                rl_replay_memory.clear()
                qValue = np.array([TupleNetwork(), TupleNetwork()])
                sess.run(tf.global_variables_initializer())
                life_steps *= LIFE_STPES_INCREASE_FACTOR
                epsilon = INITIAL_EPSILON
                for i in range(ENCODING_STEPS):
                    samples = random.sample(state_replay_memory, BATCH_SIZE)
                    state_batch = [sample[0] for sample in samples]
                    random_code_batch = [sample[1] for sample in samples]
                    code_loss, difference_code_loss = scg.update_code(sess, state_batch, random_code_batch)
                    if i % 1000 == 0:
                        print("generate code...", i)
                        print("code loss: ", code_loss)
                        print("difference code loss: ", difference_code_loss)

            code_set.add(state_code)
            actions = np.zeros([2])    
            if random.random() <= epsilon:
                action = np.random.randint(2)
            else:    
                action = np.argmax([value.GetValue(state_code) for value in qValue])
            actions[action] = 1

            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.array([state[1], next_observation])
            next_state_code = scg.get_code([next_state])[0]
            episode_reward += reward
                               
            episode_replay_memory.append((state_code, action, reward, done, next_state_code))

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES

            if len(state_replay_memory) >= REPLAY_MEMORY_SIZE:
                state_replay_memory.popleft();
            state_replay_memory.append((state, random_code()))
            
            if len(rl_replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 2 == 0:
                if total_t % 1000 == 0:
                    print("Code Set: ", len(code_set))

                samples = random.sample(rl_replay_memory, BATCH_SIZE)
                state_code_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_code_batch = [sample[4] for sample in samples]
                for i in range(BATCH_SIZE):
                    replay_state_code = state_code_batch[i]
                    replay_action = action_batch[i]
                    replay_reward = reward_batch[i]
                    replay_done = done_batch[i]
                    replay_next_state_code = next_state_code_batch[i]
                    if replay_done:
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code)))
                    else:
                        next_max = GAMMA * np.max([value.GetValue(replay_next_state_code) for value in qValue])
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + next_max - qValue[replay_action].GetValue  (replay_state_code)))

            if done:
                average = np.mean(log)
                deviation = np.std(log) + 0.01
                for episode_replay in episode_replay_memory:
                    _state_code, _action, _reward, _done, _next_state_code = episode_replay
                    transfer_reward = _reward * (1 + elu((episode_reward - average) / deviation)) if (_reward >= 0) else _reward * (1 - inverse_elu((episode_reward - average) / deviation))
                    if len(rl_replay_memory) >= REPLAY_MEMORY_SIZE:
                        rl_replay_memory.popleft();
                    rl_replay_memory.append((_state_code, _action, transfer_reward, _done, _next_state_code));
                
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