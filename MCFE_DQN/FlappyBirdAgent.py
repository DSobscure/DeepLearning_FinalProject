import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from DQN import DQN
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
EXPLORE_STPES = 200000

Q_EXPLORE_STPES = 500000
Q_INITIAL_EPSILON = 0.5
Q_FINAL_EPSILON = 0

ENCODING_STEPS = 25000
INITIAL_LIFE_STPES = 1000000
LIFE_STPES_INCREASE_FACTOR = 10

INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 100000

BATCH_SIZE = 32

CODE_SIZE = 24
FEATURE_LEVEL = 1

Q_LEARNING_RATE = 1.0


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
    return np.array(state)

def get_initial_state(env):
    observation = Game.GameState()
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    observation, _, _ = env.frame_step(do_nothing)
    return observation

def main(_):
    env = Game.GameState()
    scg = SCG(CODE_SIZE, FEATURE_LEVEL)
    qValue = np.array([TupleNetwork(CODE_SIZE, FEATURE_LEVEL), TupleNetwork(CODE_SIZE, FEATURE_LEVEL)])
    dqn = DQN(GAMMA, 2)

    replay_memory = deque()
    rl_replay_memory = deque()
    log = deque()
    code_set = set()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    episode_reward = 0    
    epsilon = INITIAL_EPSILON
    q_epsilon = Q_INITIAL_EPSILON
    life_steps = INITIAL_LIFE_STPES

    observation = process_state(get_initial_state(env))
    state = np.stack([observation] * 2, axis=2)
    episode_replay_memory = []

    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        actions = np.zeros([2])
        if random.random() <= epsilon:
            action = np.random.randint(2)
        else:    
            action = 0
        actions[action] = 1

        next_observation, reward, done = env.frame_step(actions)
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        
        episode_replay_memory.append((state, action, reward, done, next_state, random_code()))
        episode_reward += reward
        
        # Current game episode is over
        if done:
            observation = process_state(get_initial_state(env))
            state = np.stack([observation] * 2, axis=2)

            for episode_replay in episode_replay_memory:
                _state, _action, _reward, _done, _next_state, _random_code = episode_replay
                replay_memory.append((_state, _action, _reward, _done, _next_state, _random_code, episode_reward));
            episode_replay_memory = []
            log.append(episode_reward)
            if len(log) > 100:
                log.popleft()
            print ("Episode reward: ", episode_reward, '100 mean: ', np.mean(log), ' dev: ', np.std(log), " Buffer: ", len(replay_memory))
            episode_reward = 0
        else:
            state = next_state

    for i in range(ENCODING_STEPS):
        samples = random.sample(replay_memory, BATCH_SIZE)
        state_batch = [sample[0] for sample in samples]
        random_code_batch = [sample[5] for sample in samples]
        scg.update_code(sess, state_batch, random_code_batch)
        if i % 1000 == 0:
            print("generate code...", i)

    total_t = 0
    for episode in range(1000000):
        episode_reward = 0
        episode_replay_memory = []

        observation = process_state(get_initial_state(env))
        state = np.stack([observation] * 2, axis=2)
        state_code = scg.get_code([state])[0]

        for t in itertools.count():
            if total_t > life_steps:
                code_set.clear()
                rl_replay_memory.clear()
                qValue = np.array([TupleNetwork(CODE_SIZE, FEATURE_LEVEL), TupleNetwork(CODE_SIZE, FEATURE_LEVEL)])
                life_steps *= LIFE_STPES_INCREASE_FACTOR
                q_epsilon = Q_INITIAL_EPSILON
                for i in range(ENCODING_STEPS):
                    samples = random.sample(replay_memory, BATCH_SIZE)
                    state_batch = [sample[0] for sample in samples]
                    random_code_batch = [sample[5] for sample in samples]
                    scg.update_code(sess, state_batch, random_code_batch)
                    if i % 1000 == 0:
                        print("generate code...", i)

            code_set.add(state_code)
            actions = np.zeros([2])    
            if random.random() <= epsilon:
                action = np.random.randint(2)
            else:    
                if random.random() < q_epsilon:
                    action = dqn.select_action(state)
                else:
                    action = np.argmax([value.GetValue(state_code) for value in qValue])
            actions[action] = 1

            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            next_state_code = scg.get_code([next_state])[0]
            episode_reward += reward
                               
            episode_replay_memory.append((state, state_code, action, reward, done, next_state, next_state_code, random_code()))

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES
            if q_epsilon > Q_FINAL_EPSILON:
                q_epsilon -= (Q_INITIAL_EPSILON - Q_FINAL_EPSILON) / Q_EXPLORE_STPES
            
            if total_t % 1000 == 0:
                print("Code Set: ", len(code_set))

            if len(rl_replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 4 == 0:
                samples = random.sample(rl_replay_memory, BATCH_SIZE)
                state_code_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_code_batch = [sample[4] for sample in samples]
                score_batch = [sample[5] for sample in samples]

                average = np.mean(log)
                deviation = np.std(log) + 0.01
                for i in range(BATCH_SIZE):
                    replay_state_code = state_code_batch[i]
                    replay_action = action_batch[i]
                    replay_reward = reward_batch[i] * (1 + elu((score_batch[i] - average) / deviation)) if (reward_batch[i] >= 0) else reward_batch[i] * (1 - inverse_elu((score_batch[i] - average) / deviation))
                    replay_done = done_batch[i]
                    replay_next_state_code = next_state_code_batch[i]
                    if replay_done:
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code)))
                    else:
                        next_max = GAMMA * np.max([value.GetValue(replay_next_state_code) for value in qValue])
                        qValue[replay_action].UpdateValue(replay_state_code, Q_LEARNING_RATE * (replay_reward + next_max - qValue[replay_action].GetValue  (replay_state_code)))
            if len(replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                state_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_batch = [sample[4] for sample in samples]
                score_batch = [sample[6] for sample in samples]
                average = np.mean(log)
                deviation = np.std(log) + 0.01
                for i in range(BATCH_SIZE):
                    reward_batch[i] = reward_batch[i] * (1 + elu((score_batch[i] - average) / deviation)) if (reward_batch[i] >= 0) else reward_batch[i] * (1 - inverse_elu((score_batch[i] - average) / deviation))
                dqn.update(sess, state_batch, action_batch, reward_batch, done_batch, next_state_batch)
            
            if total_t % 10000 == 0:
                dqn.update_target_network(sess)

            if done or t >= 5000:
                average = np.mean(log)
                deviation = np.std(log) + 0.01
                episode_replay_memory.reverse()
                for episode_replay in episode_replay_memory:
                    _state, _state_code, _action, _reward, _done, _next_state, _next_state_code, _random_code = episode_replay
                    
                    if len(rl_replay_memory) >= REPLAY_MEMORY_SIZE:
                        rl_replay_memory.popleft();
                    rl_replay_memory.append((_state_code, _action, _reward, _done, _next_state_code, episode_reward))

                    if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                        replay_memory.popleft();
                    replay_memory.append((_state, _action, _reward, _done, _next_state, _random_code, episode_reward))

                    transfer_reward = _reward * (1 + elu((episode_reward - average) / deviation)) if (_reward >= 0) else _reward * (1 - inverse_elu((episode_reward - average) / deviation))
                    if _done:
                        qValue[_action].UpdateValue(_state_code, Q_LEARNING_RATE * (transfer_reward + 0 - qValue[_action].GetValue(_state_code)))
                    else:
                        next_max = GAMMA * np.max([value.GetValue(_next_state_code) for value in qValue])
                        qValue[_action].UpdateValue(_state_code, Q_LEARNING_RATE * (transfer_reward + next_max - qValue[_action].GetValue(_state_code)))
                
                log.append(episode_reward)
                if len(log) > 100:
                    log.popleft()
                    
                print ("Score: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t, '100 mean: ', np.mean(log), ' dev: ', np.std(log), "Epsilon: ", epsilon, " Q Epsilon: ", q_epsilon)
                total_t += 1
                break

            state = next_state
            state_code = next_state_code
            total_t += 1


if __name__ == '__main__':
    tf.app.run()