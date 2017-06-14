import gym
import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from StateCodeGenerator import SCG
from TupleNetwork import TupleNetwork
import math

GAMMA = 0.99

INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORE_STPES = 500000
CHILD_EXPLORE_STPES = 100000
LIFE_STPES = 1000000


# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 500000

BATCH_SIZE = 32

CODE_SIZE = 24
WINDOW_SIZE = 2

Q_LEARNING_RATE = 0.1

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
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state).reshape([84,84,1])

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")
    qValue = np.array([TupleNetwork(), TupleNetwork(), TupleNetwork(), TupleNetwork()])

    # The replay memory
    # The replay memory
    state_replay_memory = deque()
    rl_replay_memory = deque()
    heritage_replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    scg = SCG(CODE_SIZE)
    code_set = set()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1000)

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * WINDOW_SIZE)
    state_replay_memory.append((state[0], random_code(), state[1], random_code()))
    state_code = scg.get_code([state[0]], [state[1]])
    initial_state = state

    episode_reward = 0    
    epsilon = INITIAL_EPSILON

    while len(state_replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = random.randrange(4)

        next_observation, reward, done, _ = env.step(action)
        next_observation = process_state(next_observation)      
        next_state = np.array([state[1], next_observation]) 
        state_replay_memory.append((next_state[0], random_code(), next_state[1], random_code()))
        episode_reward += reward

        # Current game episode is over
        if done:
            observation = env.reset()                 # retrive first env image
            observation = process_state(observation)
            state = np.stack([observation] * WINDOW_SIZE)

            log.append(episode_reward)
            if len(log) > 100:
                log.popleft()
            print ("Episode reward: ", episode_reward, '100 mean: ', np.mean(log), ' dev: ', np.std(log), " Buffer: ", len(state_replay_memory))
            episode_reward = 0
        else:
            state = next_state

    # total steps
    total_t = 0

    for episode in range(1000000):

        # Reset the environment
        observation = env.reset()                 # retrive first env image
        observation = process_state(observation)
        state = np.stack([observation] * WINDOW_SIZE)

        state_code = scg.get_code([state[0]], [state[1]])
        episode_reward = 0

        episode_replay_memory = []
        episode_heritage_replay_memory = []

        for t in itertools.count():

            if total_t % LIFE_STPES == 0:
                rl_replay_memory.clear()
                code_set.clear()
                qValue = np.array([TupleNetwork(), TupleNetwork(), TupleNetwork(), TupleNetwork()])
                scg = SCG(CODE_SIZE)
                sess.run(tf.global_variables_initializer())
                epsilon += (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES * CHILD_EXPLORE_STPES
                
                for i in range(50000):
                    samples = random.sample(state_replay_memory, BATCH_SIZE)
                    state_batch = [sample[0] for sample in samples]
                    state_code_batch = [sample[1] for sample in samples]
                    next_state_batch = [sample[2] for sample in samples]
                    difference_code_batch = [sample[3] for sample in samples]
                    scg.update_code(sess, state_batch, state_code_batch, next_state_batch, difference_code_batch)
                    if i % 1000 == 0:
                        print("generate code...", i)
                        print(scg.get_code_batch(state_batch, next_state_batch))
                        print(scg.get_code([initial_state[0]], [initial_state[1]]))
                if len(heritage_replay_memory) > BATCH_SIZE:
                    print("we start with heritage!")
                    for j in range(50000):
                        if j % 1000 == 0:
                            print("inherit progress...", j)
                        samples = random.sample(heritage_replay_memory, BATCH_SIZE)
                        state0_batch = [sample[0][0] for sample in samples]
                        state1_batch = [sample[0][1] for sample in samples]
                        action_batch = [sample[1] for sample in samples]
                        reward_batch = [sample[2] for sample in samples]
                        done_batch = [sample[3] for sample in samples]
                        next_state0_batch = [sample[4][0] for sample in samples]
                        next_state1_batch = [sample[4][1] for sample in samples]
                        state_code_batch = scg.get_code_batch(state0_batch, state1_batch)
                        next_state_code_batch = scg.get_code_batch(next_state0_batch, next_state1_batch)
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
                heritage_replay_memory.clear()

            if random.random() <= epsilon:
                action = random.randrange(4)
            else:    
                x = [value.GetValue(state_code) for value in qValue]               
                action = random.sample(np.argwhere(x == np.max(x)).flatten().tolist(), 1)[0]

            # execute the action
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_state(next_observation)
            next_state = np.array([state[1], next_observation])
            next_state_code = scg.get_code([next_state[0]], [next_state[1]])
            code_set.add(state_code)                   
            episode_reward += reward

            episode_replay_memory.append((state_code, action, reward, done, next_state_code))

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES
            else:
                episode_heritage_replay_memory.append((state, action, reward, done, next_state))

            if len(state_replay_memory) >= REPLAY_MEMORY_SIZE:
                state_replay_memory.popleft();
            state_replay_memory.append((next_state[0], random_code(), next_state[1], random_code()))
            
            if len(rl_replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 4 == 0:
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
                for episode_heritage_replay in episode_heritage_replay_memory:
                    _state, _action, _reward, _done, _next_state = episode_heritage_replay
                    transfer_reward = _reward * (1 + elu((episode_reward - average) / deviation)) if (_reward >= 0) else _reward * (1 - inverse_elu((episode_reward - average) / deviation))
                    if len(heritage_replay_memory) >= REPLAY_MEMORY_SIZE:
                        heritage_replay_memory.popleft();
                    heritage_replay_memory.append((_state, _action, transfer_reward, _done, _next_state))
                
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