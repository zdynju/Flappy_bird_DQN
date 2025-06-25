import os
import sys
import random
import numpy as np
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
from collections import deque
import cv2

# 设置 FPS 和窗口尺寸
FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

# 设置环境变量以支持无头模式（无图形界面）
if not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
FPSCLOCK = pygame.time.Clock()
pygame.display.set_mode((1, 1)) 

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    def __init__(self, render_mode='server'):
        self.render_mode = render_mode


        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        else:
            self.screen = pygame.Surface((SCREENWIDTH, SCREENHEIGHT))

        pygame.display.set_caption('Flappy Bird')
        
        self.reset_game()
        
        
    def reset_game(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        self.pipeVelX = -4
        self.playerVelY = 0
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False
        
    def frame_step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Only one action can be active at a time.')

        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        self.playery = max(self.playery, 0)

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        isCrash = checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)
        if isCrash:
            terminal = True
            self.reset_game()
            reward = -1

        self.screen.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.screen.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.screen.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        self.screen.blit(IMAGES['base'], (self.basex, BASEY))
        self.screen.blit(IMAGES['player'][self.playerIndex],
                         (self.playerx, self.playery))

        if self.render_mode == 'human':
            pygame.display.update()

        image_data = pygame.surfarray.array3d(self.screen)
        FPSCLOCK.tick(FPS)

        return image_data, reward, terminal


class FlappyBirdEnvWrapper(GameState):
    def __init__(self, num_stack=4, gray=True, resize=(80, 80), render_mode='server'):
        self.render_mode = render_mode
        super().__init__(render_mode=render_mode)
        self.num_stack = num_stack
        self.gray = gray
        self.resize = resize
        self.frames = deque(maxlen=num_stack)
        self.reset()

    def reset(self):
        self.reset_game()
        obs, _, _ = self.frame_step(np.array([1, 0]))  # 静止动作
        frame = self._preprocess(obs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(frame)
        return self._get_obs()

    def step(self, action):
        obs, reward, done = self.frame_step(action)
        frame = self._preprocess(obs)
        self.frames.append(frame)
        return self._get_obs(), reward, done

    def _preprocess(self, img):
        img = np.transpose(img, (1, 0, 2))  # (W,H,C) → (H,W,C)
        img = cv2.resize(img, self.resize)
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2) if self.num_stack > 1 else self.frames[-1]


def getRandomPipe():
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    gapY = random.choice(gapYs) + int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10
    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]


def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    if player['y'] + player['h'] >= BASEY - 1:
        return True

    playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        lRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        pMask = HITMASKS['player'][pi]
        uMask = HITMASKS['pipe'][0]
        lMask = HITMASKS['pipe'][1]

        if pixelCollision(playerRect, uRect, pMask, uMask) or pixelCollision(playerRect, lRect, pMask, lMask):
            return True
    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
