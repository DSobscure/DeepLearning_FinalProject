import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import sys
import math
from StateCodeGenerator import SCG
from TupleNetwork import TupleNetwork
sys.path.append("game/")
import wrapped_flappy_bird as Game

GAMMA = 0.99

INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.001
EXPLORE_STPES = 500000
ENCODING_STPES = 40000
INITIAL_LIFE_STPES = 500000
LIFE_STPES_INCREASE_FACTOR = 5

# replay memory
INIT_REPLAY_MEMORY_SIZE =10000
REPLAY_MEMORY_SIZE = 100000

BATCH_SIZE = 32
CODE_SIZE = 24

TN_Q_LEARNING_RATE = 0.0025

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

def random_code(seed):
    result = np.zeros(CODE_SIZE)
    block_mask_index = np.random.randint(int(CODE_SIZE / 4))
    for i in range(CODE_SIZE):
        if block_mask_index * 4 <=  i and i < (block_mask_index + 1) * 4:
            result[i] = np.random.randint(2)
        else:
            result[i] = seed[i]
    return result

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state).reshape([84,84,1])

def main(_):
    env = Game.GameState()
    qValue = np.array([TupleNetwork(), TupleNetwork()])
    scg = SCG(CODE_SIZE)

    state_replay_memory = deque()
    rl_replay_memory = deque()
    log = deque()
    code_set = set()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    epsilon = INITIAL_EPSILON
    total_t = 0
    life_steps = INITIAL_LIFE_STPES

    episode_reward = 0
    observation = Game.GameState()
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing)                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4)
    code_seed = np.zeros(CODE_SIZE)

    state_replay_memory.append((observation, code_seed))
    code_seed = random_code(code_seed)

    while len(state_replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= epsilon:
            action = np.random.randint(2)
            actions[action] = 1
        else:    
            action = 0
            actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.array([state[1], state[2], state[3], next_observation])
        
        state_replay_memory.append((next_observation, code_seed))
        code_seed = random_code(code_seed)       

        episode_reward += reward
        
        # Current game episode is over
        if done:
            observation = Game.GameState()
            do_nothing = np.zeros(2)
            do_nothing[0] = 1
            observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
            observation = process_state(observation)
            state = np.stack([observation] * 4)
            state_replay_memory.append((observation, code_seed))
            code_seed = random_code(code_seed)
            log.append(episode_reward)
            if len(log) > 100:
                log.popleft()
            print ("Episode reward: ", episode_reward, '100 mean: ', np.mean(log), ' dev: ', np.std(log), " Buffer: ", len(state_replay_memory))
            episode_reward = 0
        else:
            state = next_state

    for i in range(ENCODING_STPES):
        samples = random.sample(state_replay_memory, BATCH_SIZE)
        state_batch = [sample[0] for sample in samples]
        random_code_batch = [sample[1] for sample in samples]
        code_loss = scg.update_code(sess, state_batch, random_code_batch)
        if i % 1000 == 0:
            print("generate code...", i)
            print("code loss: ", code_loss)

    for episode in range(1000000):
        observation = Game.GameState()
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)
        observation = process_state(observation)
        state = np.stack([observation] * 4)
        state_code = scg.get_code([state])[0]
        code_seed = np.zeros(CODE_SIZE)

        state_replay_memory.append((observation, code_seed))
        code_seed = random_code(code_seed)
        episode_reward = 0
        episode_replay_memory = []

        for t in itertools.count():
            if total_t > life_steps:
                code_set.clear()
                sess.run(tf.global_variables_initializer())
                qValue = np.array([TupleNetwork(), TupleNetwork()])
                life_steps *= LIFE_STPES_INCREASE_FACTOR
                for i in range(ENCODING_STPES):
                    samples = random.sample(state_replay_memory, BATCH_SIZE)
                    state_batch = [sample[0] for sample in samples]
                    random_code_batch = [sample[1] for sample in samples]
                    code_loss = scg.update_code(sess, state_batch, random_code_batch)
                    if i % 1000 == 0:
                        print("generate code...", i)
                        print("code loss: ", code_loss)

            actions = np.zeros([2])
            code_set.add(state_code)
            if random.random() <= epsilon:
                action = np.random.randint(2)
            else:    
                action = np.argmax([value.GetValue(state_code) for value in qValue])
            actions[action] = 1
            
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.array([state[1], state[2], state[3], next_observation])
            next_state_code = scg.get_code([next_state])[0]
            episode_reward += reward

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES
            
            episode_replay_memory.append((state_code, action, reward, done, next_state_code))
            code_seed = random_code(code_seed)

            if len(rl_replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 8 == 0:
                samples = random.sample(rl_replay_memory, BATCH_SIZE)
                state_code_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_code_batch = [sample[4] for sample in samples]            

                tn_q_loss_sum = 0
                for i in range(BATCH_SIZE):
                    replay_state_code = state_code_batch[i]
                    replay_action = action_batch[i]
                    replay_reward = reward_batch[i]
                    replay_done = done_batch[i]
                    replay_next_state_code = next_state_code_batch[i]
                    if replay_done:
                        tn_q_loss = replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code)
                        qValue[replay_action].UpdateValue(replay_state_code, TN_Q_LEARNING_RATE * tn_q_loss)
                    else:
                        next_max = GAMMA * np.max([value.GetValue(replay_next_state_code) for value in qValue])
                        tn_q_loss = replay_reward + next_max - qValue[replay_action].GetValue(replay_state_code)
                        qValue[replay_action].UpdateValue(replay_state_code, TN_Q_LEARNING_RATE * tn_q_loss)
                    tn_q_loss_sum += abs(tn_q_loss)

                if total_t % 1000 == 0:
                    print("Code Set: ", len(code_set))
                    print("tn q loss:", tn_q_loss_sum / BATCH_SIZE)           

            if done or t > 10000:     
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