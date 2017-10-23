from pygame.locals import *
from random import random, randint
from time import sleep

import math
import numpy as np    
import pygame
import sys  

BIRD_UP = -1
BIRD_DOWN = 1

class NeuralNetwork(object):
    """
    Neural Network with two inputs and one output. 
    1 hidden layer with, 8 neurons
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_to_hidden_weights =  np.random.uniform(-1,1,(2,8))
        self.hidden_to_output_weights = np.random.uniform(-1,1,(8,1))
        self.state = 0
    
    def sigmoid(self, matrix):
        def _sigmoid(z):
            return 1 / (1 + math.exp(-z))
        return np.array([[_sigmoid(z) for z in row] for row in matrix])

    def output(self, inputs):
        inputs = [inputs]
        z = self.sigmoid(np.matmul(inputs, self.input_to_hidden_weights)) # 1x2 * 2x8 = 1x8
        z = np.matmul(z, self.hidden_to_output_weights)[0] # 1x8 * 8x1 = 1x1        
        return 1 / (1 + math.exp(-z))

    def copy(self):
        nn = NeuralNetwork()
        nn.input_to_hidden_weights = np.copy(self.input_to_hidden_weights)
        nn.hidden_to_output_weights = np.copy(self.hidden_to_output_weights)
        return nn

    def mate(self, other, w1, w2):        
        nn = NeuralNetwork()
        nn.input_to_hidden_weights = (w1 * self.input_to_hidden_weights + w2 * other.input_to_hidden_weights) / 2
        nn.hidden_to_output_weights = (w1 * self.hidden_to_output_weights + w2 * other.hidden_to_output_weights) /2
        return nn

    def mutate(self):
        indices = [randint(0,24) for i in range(4)]
        for index in indices:
            if index < 16:
                i = index / 8
                j = index % 8
                self.input_to_hidden_weights[i][j] = (random()*2)-1
            else :
                i = index % 8
                self.hidden_to_output_weights[i][0] = (random()*2)-1
    
    def __repr__(self):
        return "\nLayer 1 : " + self.input_to_hidden_weights.__repr__() + "\nLayer 2 : " + self.hidden_to_output_weights.__repr__()


g = 1
class Bird(object):
    """docstring for Bird"""
    def __init__(self):
        super(Bird, self).__init__()
        self.nn = NeuralNetwork()
        self.direction = BIRD_UP
        self.alive = True
        self.score = 0
        self.v = 0


    def set_pos(self, pos):
        self.current_pos = pos

    def is_alive(self):
        return self.alive

    def kill(self):
        self.alive = False;

    def tick(self, pipe):
        self.score += 1
        #h_dist = (0, self.pipe_dist)
        #v_dist = (-self.height, self.height)
        # h_dist = (abs(pipe[0] - self.x()))
        # v_dist = (((pipe[1][0] + pipe[1][1]) / 2) - self.y())
        # print h_dist, v_dist        
        h_dist = self.map_h_dist(abs(pipe[0] - self.x()))
        v_dist = self.map_v_dist(((pipe[1][0] + pipe[1][1]) / 2) - self.y())
        # print h_dist, v_dist        
        decision = self.nn.output([h_dist, v_dist])
        # print decision
        return decision > 0.5

    def map_h_dist(self, d):
        pipe_dist = 250
        return -1 + (1.0 - (-1))/(pipe_dist - 0) * (d)

    def map_v_dist(self, d):
        height = 600
        return -1 + (1.0 - (-1)) / (height - (-height)) * (d - (-height))

    def move(self, offset, height_limit):        
        if offset[1] < 0:
            self.direction = BIRD_UP
            self.v = -2*g
            # self.current_pos = np.add(self.current_pos, offset)
        else:
            self.direction = BIRD_DOWN
            self.v += g
        self.current_pos = np.add(self.current_pos, [offset[0], self.v])
        self.current_pos[1] = self.current_pos[1] % height_limit

    def x(self):
        return self.current_pos[0]
    def y(self):
        return self.current_pos[1]

    def copy(self):
        newBird = Bird()
        newBird.nn = self.nn.copy()
        return newBird

    def mate(self, other):
        newBird = Bird()
        w = float(self.score + other.score)
        w1 = 0.5
        w2 = 0.5
        if w != 0:
            w1 = self.score / w
        if w != 0:
            w2 = other.score / w
        newBird.nn = self.nn.mate(other.nn, w1, w2)
        return newBird

    def mutate(self):
        self.nn.mutate()
        return self

        
class Game(object):
    """docstring for Game"""
    def __init__(self):        
        super(Game, self).__init__()
        self.tick_count = 0
        self.generataion = 0
        self.nr_birds = 10
        self.width  = 800
        self.height = 600
        self.pipe_dist = 250
        self.birds = [Bird() for i in range(self.nr_birds)]
        self.reset()

    def reset(self):
        self.generataion += 1
        if self.generataion % 100 == 0:
            print self.generataion
        for bird in self.birds:
            bird.set_pos([20, self.height/2])
        self.pipes = []
        self.alive_birds = filter(lambda bird: bird.is_alive(), self.birds)
        self.generatePipes()
        self.pipe_width = 20
        self.bird_width = 32
        self.bird_height = 32

    def generatePipes(self):
        curr_pipe = self.pipe_dist
        # self.pipes = [(curr_pipe, (self.height/2-25, self.height/2+25))]
        curr_pipe += self.pipe_dist
        while curr_pipe <= self.width:
            gap = randint(0, self.height)
            self.pipes.append(self.newPipe(curr_pipe))
            curr_pipe += self.pipe_dist

    def newPipe(self, x):
        if len(self.pipes) == 0:
            last_gap = randint(100, self.height-100)
        else:
            last_gap = self.pipes[-1][1][0] + 25
        diff = 240
        gap = last_gap
        while abs(gap - last_gap) < 150:
            gap = randint(last_gap-diff,last_gap+diff)
        if gap > self.height-50:
            gap = self.height-50
        if gap < 50:
            gap = 50
    
        return (x,(gap-25, gap+25))

    def debug(self):
        print self.bird.x(), self.bird.y()

    def tick(self, tick_count):
        self.alive_birds = filter(lambda bird: bird.is_alive(), self.birds)        
        for bird in self.alive_birds:
            direction = 1
            if bird.tick(self.pipes[0]):                
                direction = -1            
            bird.move([0,direction], self.height)
            
            if self.collisonWithPipe(bird, self.pipes[0]):
                bird.kill()
            elif self.outOfBounds(bird):
                bird.kill()
                # bird.score /= 2
            elif bird.score > 250*100:
                print bird.nn.__repr__()
                bird.kill()

        if len(self.alive_birds) == 0:
            print sorted(map(lambda b : b.score, self.birds), reverse=True)
            self.generateNewGeneratation(self.birds)
            self.reset()
            return
            #raise Exception("All dead")

        for i in range(len(self.pipes)):
            pipe = self.pipes[i]
            self.pipes[i] = (pipe[0]-2, pipe[1])

        if self.passed(self.alive_birds[0], self.pipes[0]):
            self.removeFirstPipe()

        # if self.pipes[0][0] < self.pipe_dist:
            self.addNewPipe()

        self.tick_count += 1


    def generateNewGeneratation(self, old):
        old.sort(key = lambda b: b.score, reverse=True)
        new = [b.copy() for b in old[:3]] #copy 3 best birds to new generataions

        for (i,j) in [(1,2), (2,3), (1,3)]:
            new.append(old[i].mate(old[j]))

        for i in range(4):
            new.append(old[i].copy().mutate())

        self.birds = new

    def outOfBounds(self, bird):
        return (bird.y() + self.bird_height/2 > self.height) or (bird.y() - self.bird_height/2 < 0)

    def collisonWithPipe(self, bird, pipe):
        collison_with_pipe = (bird.x() + self.bird_width/2 > pipe[0] - self.pipe_width/2 and (bird.y() <= pipe[1][0] or bird.y() >= pipe[1][1]))
        return collison_with_pipe
        
    def passed(self, bird, pipe):
        return bird.x() > pipe[0]

    def removeFirstPipe(self):
        self.pipes = self.pipes[1:]

    def addNewPipe(self):        
        last_pipe = self.pipes[-1]
        self.pipes.append(self.newPipe(last_pipe[0] + self.pipe_dist))


class GameWindow(object):
    """docstring for GameWindow"""
    def __init__(self, game):
        super(GameWindow, self).__init__()
        pygame.init()
        self.game = game
        self.screen = pygame.display.set_mode((self.game.width, self.game.height))
        self.bird_up = pygame.image.load('bird_up.png')
        self.bird_down = pygame.image.load('bird_down.png')
        self.pipe = pygame.image.load('pipe.png')
        #self.birds = [self.bird_up for i in range(game.nr_birds)]

    def loop(self, fps):
        loop_delay = 1.0 / fps
        tick_count = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            self.game.tick(tick_count)
            self.draw()
            # self.debug()
            sleep(loop_delay)
            tick_count += 1
        
    def draw(self):
        self.screen.fill((0,0,0))        
        for bird in self.game.alive_birds:
            image = self.bird_up if (bird.direction == BIRD_UP) else self.bird_down
            self.screen.blit(image, (bird.x() - self.game.bird_width/2, bird.y() - self.game.bird_height/2))
        for pipe in self.game.pipes:
            y = pipe[1][0]
            while y >= 0:
                self.screen.blit(self.pipe, (pipe[0]-16, y - 32))
                y -= 32

            y = pipe[1][1]
            while y < self.game.height:
                self.screen.blit(self.pipe, (pipe[0]-16, y))
                y += 32
            
        pygame.display.flip()

    def debug(self):
        pass

def main():
    game = Game()
    window = GameWindow(game)
    window.loop(500)

if __name__ == '__main__':
    main()