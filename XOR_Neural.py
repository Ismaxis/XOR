import numpy as np
import pygame as pg
from Functions import generating_net, forward, back_prop
from Display import draw, wait
pg.font.init()

# display settings
WIN_SIZE = (800, 800)
SIZE_OF_NODE = 100
MATRIX_SIZE = (WIN_SIZE[0] // SIZE_OF_NODE, WIN_SIZE[1] // SIZE_OF_NODE)
WIN = pg.display.set_mode(WIN_SIZE)
STAT_FONT = pg.font.SysFont("comics", 50)


# file for logging
File = open("Weights.txt", "a")

# generating struct and random weights
struct = (2, 3, 1)

# in func code weights named log. 55 str
Neurons, weights = generating_net(struct)

log = ''

for i in range(len(weights)):
    for j in range(len(weights[i])):
        log += f'{round(weights[i][j], 2)} '
    log += '| '

File.write(log)
File.write(' -Start weights\n')

lmb = 0.5

# dataset
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]

amount_of_cicles = 0
amount_right = 0

while True:
    amount_of_cicles += 1

    # random number of cur trained set
    cur_set = np.random.randint(0, len(inputs))

    output = forward(inputs[cur_set], struct, Neurons)

    e = output - outputs[cur_set]

    Neurons = back_prop(Neurons, e, struct, lmb)

    if abs(e) < 0.4:
        amount_right += 1

    rate = amount_right / amount_of_cicles
    if rate >= 0.9 and amount_of_cicles > 1000:
        # console
        for cur_set in range(len(inputs)):
            output = forward(inputs[cur_set], struct, Neurons)

            print(f'\u001b[37mNetWork result: {inputs[cur_set]} => {output}')
            print(f'Required result: {inputs[cur_set]} => {outputs[cur_set]}')

            if output >= 0.5:
                output = 1
            else:
                output = 0

            if output == outputs[cur_set]:
                print('\u001b[32mpassed')
            else:
                print('\u001b[31mnot passed')

        # file logging
        log = ''
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                log += f'{round(weights[i][j], 2)} '
            log += '| '

        File.write(log)
        File.write(' -Trained weights\n')
        File.write(f'{amount_of_cicles}\n')

        File.close()

        # display
        # passing values from 0 to 1 throuht net
        values = []
        for i in range(MATRIX_SIZE[0]):
            values.append([])
            for j in range(MATRIX_SIZE[1]):
                inp = (i * 0.1, j * 0.1)
                values[i].append(forward(inp, struct, Neurons))

        draw(WIN, values, MATRIX_SIZE, SIZE_OF_NODE)
        break

    if rate < 0.85 and amount_of_cicles > 60000:
        print('\u001b[31mLearning failed. \u001b[37mPlease retry')

        WIN.fill((0, 0, 0))
        label = STAT_FONT.render(
            'Learning failed. Please retry', True, (255, 255, 255))
        WIN.blit(label, (WIN_SIZE[0] // 2 -
                 label.get_width() // 2, WIN_SIZE[1] // 2))
        pg.display.update()

        wait()

        break

    if amount_of_cicles % 1000 == 0:
        # console
        print("------------------------")
        print(f'{round(rate * 100)} %')

        # display
        WIN.fill((0, 0, 0))
        label = STAT_FONT.render(
            f'{round(rate * 100)} %', True, (255, 255, 255))
        WIN.blit(label, (WIN_SIZE[0] // 2 -
                 label.get_width() // 2, WIN_SIZE[1] // 2))
        pg.display.update()


print(f'\u001b[37mNetWork were trained for {amount_of_cicles} cicles')
