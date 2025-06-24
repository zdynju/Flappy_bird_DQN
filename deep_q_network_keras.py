#!/usr/bin/env python
from __future__ import print_function
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal, RandomUniform
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
                        kernel_initializer='he_normal'),  # 改用he_normal初始化
        keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal'),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                        kernel_initializer='he_normal'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(ACTIONS, activation='linear', kernel_initializer='he_normal')  # 改为linear激活
    ])
    model.compile(loss='mse',
                optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

def trainNetwork(model):
    target_model = createNetwork()
    target_model.set_weights(model.get_weights())
    # open up a game state to communicate with emulator
    log_dir = "logs_" + GAME
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # open up a game state to communicate with emulator
    game_state = game.FlappyBirdEnvWrapper(num_stack=4,gray=True)

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open(os.path.join(log_dir, "readout.txt"), 'w')
    h_file = open(os.path.join(log_dir, "hidden.txt"), 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    s_t = game_state.reset()
    print(s_t.shape)
    # x_t, r_0, terminal = game_state.step(do_nothing)
    # x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    try:
        model.load_weights('test.h5')   
        target_model.set_weights(model.get_weights())
        print("Successfully loaded weights")
    except:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        print(s_t.shape)
        # choose an action epsilon greedily
        state = s_t.astype('float32').reshape(1, 80, 80, 4) / 255.0  # 归一化输入
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

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        
        s_t1, r_t, terminal = game_state.step(a_t)
        print("s_t1:",s_t1.shape)
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch] 
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # 归一化状态
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH, 80, 80, 4) / 255.0
            next_state_batch = np.array(s_j1_batch).astype('float32').reshape(BATCH, 80, 80, 4) / 255.0
            
            # 获取当前Q值和下一状态Q值
            current_q_values = model.predict(state_batch, verbose=0)
            next_q_values = target_model.predict(next_state_batch, verbose=0)
            
            # 准备目标值
            target_q_values = current_q_values.copy()
          
            for i in range(BATCH):
                terminal = minibatch[i][4]
                action_index = np.argmax(a_batch[i])
                
                if terminal:
                    target_q_values[i][action_index] = r_batch[i]
                else:
                    target_q_values[i][action_index] = r_batch[i] + GAMMA * np.max(next_q_values[i])

            # 训练网络
            model.train_on_batch(state_batch, target_q_values)

        # update the old values
        s_t = s_t1
        t += 1

        if t % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
            print(f"Target network updated at timestep {t}")
        # save progress every 10000 iterations
        if t % 10000 == 0:
            model.save('test.h5')
            print(f"Model saved at timestep {t}")

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
            "/ EPSILON", round(epsilon, 4), "/ ACTION", action_index, "/ REWARD", r_t,
            "/ Q_MAX %.4f" % np.max(readout_t),
            "/ Q_MIN %.4f" % np.min(readout_t))

        # 减少内存清理频率
        if t % 1000 == 0:
            K.clear_session()
            gc.collect()

def playGame():
    model = createNetwork()
    trainNetwork(model)

def main():
    playGame()

if __name__ == "__main__":
    main()
