#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import  keras
from keras.optimizers import Adam
from keras import backend as K
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import os
import gc
import datetime
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print("CUDA 版本:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN 版本:", tf.sysconfig.get_build_info()["cudnn_version"])
    if gpus:
        print(f"检测到 {len(gpus)} 个 GPU 可用:")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"已启用显存自增长，逻辑 GPU 数量: {len(logical_gpus)}")
        except RuntimeError as e:
            print(f"设置显存自增长失败: {e}")
    else:
        print("未检测到 GPU，使用 CPU 训练。")
        
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000.
EXPLORE = 300000.
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 5000

def createNetwork():
    model = keras.Sequential([
        keras.Input(shape=(80, 80, 4)),
        keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_normal'),
        keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal'),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(ACTIONS, activation='linear', kernel_initializer='he_normal')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

def trainNetwork(model):
    target_model = createNetwork()
    target_model.set_weights(model.get_weights())

    log_dir = os.path.join("logs_tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir)

    game_state = game.FlappyBirdEnvWrapper(num_stack=4, gray=True)
    D = deque()

    s_t = game_state.reset()
    epsilon = INITIAL_EPSILON
    t = 0
    episode_reward = 0
    episode_count = 0

    while True:
        state = s_t.astype('float32').reshape(1, 80, 80, 4) / 255.0
        readout_t = model.predict(state, verbose=0)
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        s_t1, r_t, terminal = game_state.step(a_t)
        episode_reward += r_t

        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH, 80, 80, 4) / 255.0
            next_state_batch = np.array(s_j1_batch).astype('float32').reshape(BATCH, 80, 80, 4) / 255.0

            current_q_values = model.predict(state_batch, verbose=0)
            next_q_values = target_model.predict(next_state_batch, verbose=0)

            target_q_values = current_q_values.copy()

            for i in range(BATCH):
                terminal_flag = minibatch[i][4]
                action_i = np.argmax(a_batch[i])
                if terminal_flag:
                    target_q_values[i][action_i] = r_batch[i]
                else:
                    target_q_values[i][action_i] = r_batch[i] + GAMMA * np.max(next_q_values[i])

            model.train_on_batch(state_batch, target_q_values)

        s_t = s_t1
        t += 1

        # 每步写入当前 epsilon 和 Q 值信息
        with summary_writer.as_default():
            tf.summary.scalar("epsilon", epsilon, step=t)

        if terminal:
            print("Episode finished after {} timesteps".format(t))
            episode_count += 1
            with summary_writer.as_default():
                tf.summary.scalar("episode_reward", episode_reward, step=episode_count)
            print(f"[Episode {episode_count}] Total Reward: {episode_reward:.2f}")
            episode_reward = 0

        if t % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
            print(f"Target network updated at timestep {t}")

        if t % 10000 == 0:
            model.save('test.h5')
            print(f"Model saved at timestep {t}")

        state_str = "observe" if t <= OBSERVE else "explore" if t <= OBSERVE + EXPLORE else "train"
        print("TIMESTEP", t, "/ STATE", state_str,
              "/ EPSILON", round(epsilon, 4), "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %.4f" % np.max(readout_t),
              "/ Q_MIN %.4f" % np.min(readout_t))

        if t % 1000 == 0:
            K.clear_session()
            gc.collect()

def playGame():
    model = createNetwork()
    trainNetwork(model)

def main():
    setup_gpu()
    playGame()

if __name__ == "__main__":
    main()
