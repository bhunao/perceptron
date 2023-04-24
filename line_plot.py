import sys
import matplotlib
import matplotlib.backends.backend_agg as agg
import numpy as np
from pygame import Surface, image
from pygame.math import Vector2
from matplotlib import pyplot as plt
import pygame


def map_value(oldvalue, oldmax=1, oldmin=-1, newmax=1, newmin=0):
    oldrange = (oldmax - oldmin)
    newrange = newmax - newmin
    return (((oldvalue - oldmin) * newrange) / oldrange) + newmin

def line_func(x, t):
    # mx + b
    m = 0 + .1 * t
    b = 0.1
    mx = m*x
    y = mx + b
    return y

def draw_line_function(t, screen: Surface):
    rect = screen.get_rect()
    WIDTH, HEIGHT = rect.size
    CENTER = Vector2(rect.center)

    x = np.linspace(-1, 1, WIDTH)
    x = map_value(x, newmax=WIDTH)
    y = line_func(x, t)
    p1 = CENTER
    p2 = Vector2(x[-1], y[-1])
    angle =  (p1-p2).as_polar()
    angle = Vector2(angle).rotate(0)
    other = Vector2()
    other.from_polar(angle)
    p3 = CENTER + other

    print(p1 ,p2, angle, other)
    pygame.draw.aaline(screen, "purple", p1, p2)
    pygame.draw.aaline(screen, "purple", p1, p3)


def plot_to_image(x=5, y=5) -> Surface:
    print(f"{x=}{y=}")
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(x,y))
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=1, hspace=1)

    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    size = int(x*100), int(y*100)
    return image.fromstring(raw_data, size, "RGB")

def main():
    pygame.init()
    HEIGHT = 500
    WIDTH = 500
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    plot_img = plot_to_image(WIDTH/100, HEIGHT/100)
    plot_img = pygame.transform.scale(plot_img, (WIDTH, HEIGHT))
    t = 0

    screen.blit(plot_img, (0,0))
    while 1:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

        draw_line_function(t, screen)
        t += 1
        pygame.display.flip()
        clock.tick(15)


if __name__ == "__main__":
    main()