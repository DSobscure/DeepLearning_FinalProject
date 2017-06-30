import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
from StateCodeGenerator import SCG
from TupleNetwork import TupleNetwork
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game

GAMMA = 0.99

INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001
EXPLORE_STPES = 500000
ENCODING_STPES = 50000

# replay memory
INIT_REPLAY_MEMORY_SIZE =10000
REPLAY_MEMORY_SIZE = 50000

BATCH_SIZE = 32
Q_BATCH_SIZE = 8
CODE_SIZE = 16

Q_LEARNING_RATE = 0.0025
CODE_DIVERSITY = 0.25

def random_code(seed):
    result = np.zeros(CODE_SIZE)
    for i in range(CODE_SIZE):
        if random.random() <= CODE_DIVERSITY:
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

    # The replay memory
    replay_memory = deque()
    log = deque()

    # Behavior Network & Target Network
    scg = SCG(CODE_SIZE)
    code_set = set()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    epsilon = INITIAL_EPSILON
    total_t = 0

    for episode in range(1000000):
        observation = Game.GameState()
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        observation, _, _ = env.frame_step(do_nothing)                       # retrive first env image
        observation = process_state(observation)
        state = np.stack([observation] * 2) 
        state_code = scg.get_code([state])[0]
        episode_reward = 0
        code_seed = np.zeros(CODE_SIZE)

        for t in itertools.count():
            actions = np.zeros([2])    
            if random.random() <= epsilon:
                action = np.random.randint(2)
                actions[action] = 1
            else:    
                action = np.argmax([value.GetValue(state_code) for value in qValue])
                actions[action] = 1
            
            # execute the action
            next_observation, reward, done = env.frame_step(actions)
            next_observation = process_state(next_observation)
            next_state = np.append(state[1:], np.expand_dims(next_observation, 0), axis=0)
            next_state_code = scg.get_code([next_state])[0]
            if len(replay_memory) > INIT_REPLAY_MEMORY_SIZE:
                code_set.add(state_code)                   
            episode_reward += reward

            if len(replay_memory) > INIT_REPLAY_MEMORY_SIZE and epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STPES

            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();           
            code_seed = random_code(code_seed)
            replay_memory.append((state, action, reward, done, next_state, code_seed))
            
            if len(replay_memory) == INIT_REPLAY_MEMORY_SIZE:
                for i in range(ENCODING_STPES):
                    samples = random.sample(replay_memory, BATCH_SIZE)
                    state_batch = [sample[0] for sample in samples]
                    random_code_batch = [sample[5] for sample in samples]
                    scg.update_code(sess, state_batch, random_code_batch)
                    if i % 1000 == 0:
                        print('encoding...', i)
                        scg.print_loss(state_batch, random_code_batch)
            if len(replay_memory) > INIT_REPLAY_MEMORY_SIZE and total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)
                state_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                done_batch = [sample[3] for sample in samples]
                next_state_batch = [sample[4] for sample in samples]

                state_code_batch = scg.get_code(state_batch)
                next_state_code_batch = scg.get_code(next_state_batch)

                q_loss_sum = 0
                for i in range(Q_BATCH_SIZE):
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
                    print("q loss:", q_loss_sum / Q_BATCH_SIZE)           

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