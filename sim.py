from random import random
import sys
import numpy as np
import pygame

from perceptron import Perceptron


def text_to_screen(screen, text, x, y, size = 10, color = (200, 000, 000)):
    text = str(text)
    font = pygame.font.SysFont('Arial', size)
    text = font.render(text, True, color)
    screen.blit(text, (x, y))

pygame.init()
height = 600
width = 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
p = Perceptron(verbose=False)

def random_points(n_points):
    points = [(2 * random() - 1, 2 * random() - 1) for _ in range(n_points)]
    targets = [point[0] > point[1] for point in points]

    return np.array(points), np.array(targets)

def map_value(oldvalue, oldmax=1, oldmin=-1, newmax=1, newmin=0):
    oldrange = (oldmax - oldmin)
    newrange = newmax - newmin
    return (((oldvalue - oldmin) * newrange) / oldrange) + newmin

def draw_point(point, target, predict, screen=screen):
    # print(point)
    px = map_value(point[0], newmax=width)
    py = map_value(point[1], newmax=height)

    rect = pygame.Rect(px, py, 15, 15)
    color = "green" if predict else "red"
    outline = "blue" if target else "orange"
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, outline, rect, 4)

def line_func(x):
    # mx + b
    return 0.4 * x + 0.4

def draw_line(p1, p2, color="yellow", width=4, screen=screen):
    x1 = map_value(-1, newmax=width)
    y1 = map_value(line_func(x1), newmax=height)
    x2 = map_value(1, newmax=width)
    y2 = map_value(line_func(x2), newmax=height)
    p1 = x1, x2
    p2 = y1, y2

    print(p1, p2)

    pygame.draw.line(screen, color, p1, p2, width)

x, y = random_points(50)
# p.train(inputs=x, targets=y)

while 1:
    # background
    screen.fill("gray")

    # get mouse position
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()

    # p.train(x, y)
    for point, target in zip(x, y):
        predict = p.predict(point, target)
        draw_point(point, target, predict==target)
    
    draw_line((0, 0), (width, height))

    for point, target in zip(x, y):
        p.fit(point, target)
        predict = p.predict(point, target)
    pygame.display.flip()
    clock.tick(15)