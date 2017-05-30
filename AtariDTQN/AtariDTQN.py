import gym
import itertools
import numpy as np
import random
import tensorflow as tf
from collections import deque
from PIL import Image, ImageOps
from TupleNetwork import TupleNetwork
from CodeGenerator import CodeGenerator

EXPLORE_STPES = 500000

INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 1000000

def process_state(state):
    state = Image.fromarray(state)
    state = ImageOps.fit(state, (84,84), centering=(0.5,0.7))
    state = state.convert('L')      
    return np.array(state)

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")

    # The replay memory
    replay_memory = deque()

    # Behavior Network & Target Network
    tn = TupleNetwork()

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = process_state(observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times

    epsilon = INITIAL_EPSILON
    episode_reward = 0    
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = None
        if random.random() <= epsilon:
            action = random.randrange(4)
        else:
            action = dqn.select_action(sess, state)

        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_observation = process_state(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append((state, action, reward, next_state, done))

        episode_reward += reward

        # Current game episode is over
        if done:
            observation = env.reset()
            observation = process_state(observation)
            state = np.stack([observation] * 4, axis=2)
            log.append(episode_reward);
            if len(log) > SCORE_LOG_SIZE:
                log.popleft();
            print ("Episode reward: ", episode_reward, "100 mean:", np.mean(log), "Buffer: ", len(replay_memory))
            episode_reward = 0
        # Not over yet
        else:
            state = next_state

    # record videos
    env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda count: count % 100 == 0, resume=True)

    # total steps
    total_t = 0

    for episode in range(TRAINING_EPISODES):

        # Reset the environment
        observation = env.reset()
        observation = process_state(observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward
        '''
        How to update episode reward:
        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        episode_reward += reward
        '''

        for t in itertools.count():

            # choose a action
            action = None
            if random.random() <= epsilon:
                action = random.randrange(4)
            else:
                action = dqn.select_action(sess, state)
            if epsilon > FINAL_EPSILON:
                epsilon -= (1 - FINAL_EPSILON) / EXPLORE_STPES
            # execute the action
            next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_observation = process_state(next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.popleft();
            # save the transition to replay buffer
            replay_memory.append((state, action, reward, next_state, done))
            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            if total_t % 4 == 0:
                samples = random.sample(replay_memory, BATCH_SIZE)

                state_batch = [sample[0] for sample in samples]
                action_batch = [sample[1] for sample in samples]
                reward_batch = [sample[2] for sample in samples]
                next_state_batch = [sample[3] for sample in samples]
                done_batch = [sample[4] for sample in samples]

			    # Update network
                dqn.update(sess, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            # Update target network every FREQ_UPDATE_TARGET_Q steps
            if total_t % FREQ_UPDATE_TARGET_Q == 0:
                dqn.update_target_network(sess)

            if done:
                log.append(episode_reward);
                if len(log) > SCORE_LOG_SIZE:
                    log.popleft();
                print ("Episode reward: ", episode_reward, "100 mean:", np.mean(log), 'episode = ', episode, 'total_t = ', total_t, 'epsilon: ', epsilon)
                with open('tranningResult', 'a') as file:
                    file.writelines(str(episode) + "\t" + str(total_t) + "\t" + str(episode_reward) + "\n")
                break

            state = next_state
            total_t += 1
        if (episode+1) % 100 == 0:
            savePath = saver.save(sess, "train/model.ckpt", global_step=episode)
            print("Model saved in file: %s" % savePath)


if __name__ == '__main__':
    tf.app.run()