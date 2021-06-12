import numpy as np
from Functions import generating_net, forward, back_prop


# file for logging
File = open(r"F:/python/XOR/Weights.txt", "a")

# generating struct and random weights
struct = (2, 3, 1)

Neurons, weights = generating_net(struct) # in func code weights named log. 55 str

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

        break
    
    if rate < 0.85 and amount_of_cicles > 60000:
        print('\u001b[31mLearning failed. \u001b[37mPlease retry')
        break

    if amount_of_cicles % 1000 == 0:
        print("------------------------")
        print(f'{round(rate * 100)} %')


print(f'\u001b[37mNetWork were trained for {amount_of_cicles} cicles')
