#!/usr/bin/env python
from __future__ import print_function
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import os
import keyboard
import gc

import wandb
#rom wandb.keras import WandbCallback

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 10000.  # timesteps to observe before training
EXPLORE = 500000.  # 减少探索时间，原来是2000000太长了
FINAL_EPSILON = 0.0001  # 提高最终epsilon，保持一定探索
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # 减小batch size，提高训练频率
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4  # 提高学习率，原来1e-6太小
TARGET_UPDATE_FREQ = 1000


def createNetwork():
    model = tensorflow.keras.Sequential([
        keras.Input(shape=(80, 80, 4)),
        keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu',
                        kernel_initializer='he_normal'),
        keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal'),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                        kernel_initializer='he_normal'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(ACTIONS, activation='linear', kernel_initializer='he_normal')
    ])
    model.compile(loss='mse',
                optimizer=Adam(learning_rate=LEARNING_RATE))
    return model


def trainNetwork(model):
    target_model = createNetwork()
    target_model.set_weights(model.get_weights())
    log_dir = "logs_" + GAME
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    game_state = game.FlappyBirdEnvWrapper(num_stack=4, gray=True)
    D = deque()

    a_file = open(os.path.join(log_dir, "readout.txt"), 'w')
    h_file = open(os.path.join(log_dir, "hidden.txt"), 'w')

    s_t = game_state.reset()

    try:
        model.load_weights('test.h5')
        target_model.set_weights(model.get_weights())
        print("Successfully loaded weights")
    except:
        print("Could not find old network weights")

    # 初始化 wandb
    wandb.init(project="flappybird-dqn", config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH,
        "gamma": GAMMA,
        "actions": ACTIONS,
        "epsilon_start": INITIAL_EPSILON,
        "epsilon_final": FINAL_EPSILON,
    })

    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # print(s_t.shape)
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

        if t % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
            print(f"Target network updated at timestep {t}")

        if t % 10000 == 0:
            model.save('test.h5')
            print(f"Model saved at timestep {t}")

        # wandb 记录日志
        wandb.log({
            "timestep": t,
            "reward": r_t,
            "epsilon": epsilon,
            "q_max": float(np.max(readout_t)),
            "q_min": float(np.min(readout_t)),
        })

        state_str = ""
        if t <= OBSERVE:
            state_str = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state_str = "explore"
        else:
            state_str = "train"

        print("TIMESTEP", t, "/ STATE", state_str,
              "/ EPSILON", round(epsilon, 4), "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %.4f" % np.max(readout_t),
              "/ Q_MIN %.4f" % np.min(readout_t))

        if t % 1000 == 0:
            K.clear_session()
            gc.collect()

    wandb.finish()


def playGame():
    model = createNetwork()
    trainNetwork(model)


def main():
    playGame()


if __name__ == "__main__":
    main()
