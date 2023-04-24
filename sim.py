import sys
import numpy as np
import pygame
from pygame.math import Vector2

from perceptron import Perceptron, random_points

from line_plot import plot_to_image


def text_to_screen(screen, text, x, y, size = 10, color = (200, 000, 000)):
    text = str(text)
    font = pygame.font.SysFont('Arial', size)
    text = font.render(text, True, color)
    screen.blit(text, (x, y))

pygame.init()
HEIGHT = 500
WIDTH = 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
p = Perceptron(verbose=False)
plot_img = plot_to_image(WIDTH/100, HEIGHT/100)
plot_img = pygame.transform.scale(plot_img, (WIDTH, HEIGHT))

def map_value(oldvalue, oldmax=1, oldmin=-1, newmax=1, newmin=0):
    oldrange = (oldmax - oldmin)
    newrange = newmax - newmin
    return (((oldvalue - oldmin) * newrange) / oldrange) + newmin

def draw_point(point, target, predict, screen=screen):
    px = map_value(point[0], newmax=WIDTH)
    py = map_value(point[1], newmax=HEIGHT)

    rect = pygame.Rect(px, py, 15, 15)
    color = "green" if predict else "red"
    outline = "blue" if target else "orange"
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, outline, rect, 4)

def draw_line_function(x):
    CENTER = Vector2(screen.get_rect().center)

    x = np.linspace(-1, 1, WIDTH)
    x = map_value(x, newmax=WIDTH)
    y = line_func(x)
    p1 = CENTER
    p2 = Vector2(x[-1], y[-1])
    diff =  (p1-p2).as_polar()
    diff = Vector2(diff).rotate(180)
    other = Vector2()
    other.from_polar(diff)

    print(p1 ,p2, diff, other)
    pygame.draw.line(screen, "purple", p1, p2, 5)

    p2 = p2 - (p1-p2).as_polar()
    pygame.draw.line(screen, "purple", p1, p2, 5)

def line_func(x):
    # mx + b
    m = 2
    b = 0.5
    mx = m*x
    y = mx + b
    return y

def draw_line(p1, p2, color="yellow", width=4, screen=screen):
    x1 = map_value(-1, newmax=width)
    y1 = map_value(line_func(x1), newmax=HEIGHT)
    x2 = map_value(1, newmax=width)
    y2 = map_value(line_func(x2), newmax=HEIGHT)
    p1 = x1, x2
    p2 = y1, y2

    pygame.draw.line(screen, color, p1, p2, width)
    return p1, p2

x, y = random_points(50)
# p.train(inputs=x, targets=y)

while 1:
    # background
    screen.fill("gray")
    screen.blit(plot_img, (0,0))

    # get mouse position
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()

    # for point, target in zip(x, y):
    #     predict = p.predict(point, target)
    #     draw_point(point, target, predict==target)
    
    draw_line((0, 0), (WIDTH, HEIGHT))
    draw_line_function(0)
    draw_line_function(1)
    draw_line_function(2)

    for point, target in zip(x, y):
        p.fit(point, target)
        predict = p.predict(point, target)
        draw_point(point, target, predict==target)
    pygame.display.flip()
    clock.tick(15)
