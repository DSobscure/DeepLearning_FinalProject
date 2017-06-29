import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from StateCodeGenerator import SCG
from TupleNetwork import TwoStageTupleNetwork as TupleNetwork
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game
import math

GAMMA = 0.99

INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001
EXPLORE_STPES = 500000
CHILD_EXPLORE_STPES = 100000
LIFE_STPES = 10000000
ENCODE_STEPS = 25000


# replay memory
INIT_REPLAY_MEMORY_SIZE =10000
REPLAY_MEMORY_SIZE = 50000

BATCH_SIZE = 32

CODE_SIZE = 16
FEATURE_COUNT = 8
WINDOW_SIZE = 4

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

def process_state(state):
    state = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return np.array(state)

def main(_):
    env = Game.GameState()
    qValue = np.array([TupleNetwork(CODE_SIZE, FEATURE_COUNT, 5), TupleNetwork(CODE_SIZE, FEATURE_COUNT, 5)])

    # The replay memory
    state_replay_memory = deque()
    rl_replay_memory = deque()
    heritage_replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    scg = SCG(CODE_SIZE, FEATURE_COUNT)
    code_set = set()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = Game.GameState()
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing)                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2) 
    state_code = scg.get_code([state])[0]
    initial_state = state

    episode_reward = 0    
    epsilon = INITIAL_EPSILON

    while len(state_replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= epsilon:
            action = np.random.randint(2)
            actions[action] = 1
        else:    
            action = np.argmax([value.GetValue(state_code, episode_reward) for value in qValue])
            actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        state_replay_memory.append(next_state)
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
            print ("Episode reward: ", episode_reward, '100 mean: ', np.mean(log), ' dev: ', np.std(log), " Buffer: ", len(state_replay_memory))
            episode_reward = 0
        else:
            state = next_state

    # total steps
    total_t = 0

    for episode in range(1000000):
        # Reset the environment
        observation = Game.GameState()
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)                 # retrive first env image
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2) 
        state_code = scg.get_code([state])[0]
        episode_reward = 0

        episode_replay_memory = []
        episode_heritage_replay_memory = []

        for t in itertools.count():
            if total_t % LIFE_STPES == 0:
                rl_replay_memory.clear()
                code_set.clear()
                qValue = np.array([TupleNetwork(CODE_SIZE, FEATURE_COUNT, 5), TupleNetwork(CODE_SIZE, FEATURE_COUNT, 5)])
                epsilon += (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES * CHILD_EXPLORE_STPES
                sess.run(tf.global_variables_initializer())
                for i in range(ENCODE_STEPS):
                    state_batch = random.sample(state_replay_memory, BATCH_SIZE)
                    scg.update_code(sess, state_batch)
                    if i % 1000 == 0:
                        print("generate code...", i)
                        print(scg.get_code(state_batch))
                        print(scg.get_code([initial_state])[0])
                if len(heritage_replay_memory) > BATCH_SIZE:
                    print("we start with heritage!")
                    for j in range(ENCODE_STEPS):
                        if j % 1000 == 0:
                            print("inherit progress...", j)
                        samples = random.sample(heritage_replay_memory, BATCH_SIZE)
                        state_batch = [sample[0] for sample in samples]
                        action_batch = [sample[1] for sample in samples]
                        reward_batch = [sample[2] for sample in samples]
                        done_batch = [sample[3] for sample in samples]
                        next_state_batch = [sample[4] for sample in samples]
                        score_batch = [sample[5] for sample in samples]
                        current_score_batch = [sample[6] for sample in samples]
                        state_code_batch = scg.get_code(state_batch)
                        next_state_code_batch = scg.get_code(next_state_batch)
                        average = np.mean(log)
                        deviation = np.std(log) + 0.01
                        for i in range(BATCH_SIZE):
                            replay_state_code = state_code_batch[i]
                            replay_action = action_batch[i]
                            replay_reward = reward_batch[i] * (1 + elu((score_batch[i] - average) / deviation)) if (reward_batch[i] >= 0) else reward_batch[i] * (1 - inverse_elu((score_batch[i] - average) / deviation))
                            replay_done = done_batch[i]
                            replay_next_state_code = next_state_code_batch[i]
                            if replay_done:
                                qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code)), current_score_batch[i])
                            else:
                                next_max = GAMMA * np.max([value.GetValue(replay_next_state_code, current_score_batch[i]) for value in qValue])
                                qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + next_max - qValue[replay_action].GetValue(replay_state_code, current_score_batch[i])), current_score_batch[i])
                heritage_replay_memory.clear()
            actions = np.zeros([2])    
            if random.random() <= epsilon:
                action = np.random.randint(2)
                actions[action] = 1
            else:    
                action = np.argmax([value.GetValue(state_code, episode_reward) for value in qValue])
                actions[action] = 1
            
            # execute the action
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            next_state_code = scg.get_code([next_state])[0]
            code_set.add(state_code)                   
            

            episode_replay_memory.append((state_code, action, reward, done, next_state_code, episode_reward))

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES
            else:
                episode_heritage_replay_memory.append((state, action, reward, done, next_state, episode_reward))
            episode_reward += reward
            if len(state_replay_memory) >= REPLAY_MEMORY_SIZE:
                state_replay_memory.popleft();
            state_replay_memory.append(next_state)
            
            if len(rl_replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 1 == 0:
                if total_t % 1000 == 0:
                    print("Code Set: ", len(code_set))

                samples = random.sample(rl_replay_memory, 4)
                state_code_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_code_batch = [sample[4] for sample in samples]
                score_batch = [sample[5] for sample in samples]
                current_score_batch = [sample[6] for sample in samples]
                for i in range(4):
                    replay_state_code = state_code_batch[i]
                    replay_action = action_batch[i]
                    replay_reward = reward_batch[i] * (1 + elu((score_batch[i] - average) / deviation)) if (reward_batch[i] >= 0) else reward_batch[i] * (1 - inverse_elu((score_batch[i] - average) / deviation))
                    replay_done = done_batch[i]
                    replay_next_state_code = next_state_code_batch[i]
                    if replay_done:
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code, current_score_batch[i])), current_score_batch[i])
                    else:
                        next_max = GAMMA * np.max([value.GetValue(replay_next_state_code, current_score_batch[i]) for value in qValue])
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + next_max - qValue[replay_action].GetValue(replay_state_code, current_score_batch[i])), current_score_batch[i])

            if done:
                average = np.mean(log)
                deviation = np.std(log) + 0.01
                for episode_replay in episode_replay_memory:
                    _state_code, _action, _reward, _done, _next_state_code, _current_score = episode_replay
                    if len(rl_replay_memory) >= REPLAY_MEMORY_SIZE:
                        rl_replay_memory.popleft();
                    rl_replay_memory.append((_state_code, _action, _reward, _done, _next_state_code, episode_reward, _current_score));
                for episode_heritage_replay in episode_heritage_replay_memory:
                    _state, _action, _reward, _done, _next_state, _current_score = episode_heritage_replay
                    if len(heritage_replay_memory) >= REPLAY_MEMORY_SIZE:
                        heritage_replay_memory.popleft();
                    heritage_replay_memory.append((_state, _action, _reward, _done, _next_state, episode_reward, _current_score))
                
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