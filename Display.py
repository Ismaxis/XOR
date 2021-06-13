import pygame as pg
from numpy import random
from pygame.constants import KEYDOWN, QUIT


DEF_COLOR = (255, 255, 255)


def wait():
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                quit()
            elif event.type == KEYDOWN:
                pg.quit()
                quit()


def draw(win, values, matrix_size, size_of_node):
    clock = pg.time.Clock()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        for column in range(matrix_size[0]):
            for line in range(matrix_size[1]):
                rect = (size_of_node * column, size_of_node *
                        (matrix_size[1] - line - 1), size_of_node, size_of_node)
                color = [255 * values[column][line]] * 3
                pg.draw.rect(win, color, rect)

        pg.display.update()
        clock.tick(3)


'''
# random generating of values
values = []
for i in range(10):
    values.append([])
    for j in range(10):
        values[i].append(random.randint(0, 100) / 100)

draw(WIN, values, (10, 10), 70)
'''
