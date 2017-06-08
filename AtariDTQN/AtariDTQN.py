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

GAMMA = 0.99

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORE_STPES = 200000

# replay memory
INIT_REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SIZE = 500000

BATCH_SIZE = 32

CODE_SIZE = 12
WINDOW_SIZE = 2

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
    state_replay_memory = deque()
    rl_replay_memory = deque()
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

            print ("Episode reward: ", episode_reward, "Buffer: ", len(state_replay_memory))
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

        for t in itertools.count():
            #if total_t > 10000:
            #    print([value.GetValue(state_code) for value in qValue])
            #    env.render();
            if random.random() <= epsilon:
                action = random.randrange(4)
            else:    
                action = np.argmax([value.GetValue(state_code) for value in qValue])
            if epsilon > FINAL_EPSILON:
                epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_state(next_observation)
            next_state = np.array([state[1], next_observation])
            next_state_code = scg.get_code([next_state[0]], [next_state[1]])
            if total_t > 10000:
                code_set.add(next_state_code)    
            episode_reward += reward

            if len(state_replay_memory) >= REPLAY_MEMORY_SIZE:
                state_replay_memory.popleft();
            state_replay_memory.append((next_state[0], random_code(), next_state[1], random_code()))

            if len(rl_replay_memory) >= REPLAY_MEMORY_SIZE:
                rl_replay_memory.popleft();
            rl_replay_memory.append((state_code, action, reward, done, next_state_code));

            
            if total_t % 1 == 0:
                if total_t % 1000 == 0:
                    print("Code Set: ", len(code_set))

                if total_t > 10000:
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
                            qValue[replay_action].UpdateValue(replay_state_code, 0.01 * (replay_reward + 0 - qValue[replay_action].GetValue(replay_state_code)))
                        else:
                            next_max = GAMMA * np.max([value.GetValue(replay_next_state_code) for value in qValue])
                            qValue[replay_action].UpdateValue(replay_state_code, 0.01 * (replay_reward + next_max - qValue[replay_action].GetValue  (replay_state_code)))
                if total_t < 10000:
                    samples = random.sample(state_replay_memory, BATCH_SIZE)
                    state_batch = [sample[0] for sample in samples]
                    state_code_batch = [sample[1] for sample in samples]
                    next_state_batch = [sample[2] for sample in samples]
                    difference_code_batch = [sample[3] for sample in samples]
                    scg.update_code(sess, state_batch, state_code_batch, next_state_batch, difference_code_batch)
                    if total_t % 1000 == 0:
                        print(scg.get_code_batch(state_batch, next_state_batch))
                        print(scg.get_code([initial_state[0]], [initial_state[1]]))

            if done:
                print ("Episode reward: ", episode_reward, 'episode = ', episode, 'total_t = ', total_t)
                break

            state = next_state
            state_code = next_state_code
            total_t += 1


if __name__ == '__main__':
    tf.app.run()