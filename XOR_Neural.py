import numpy as np
from Functions import generating_neurons, forward


def sigmoid(arg):
    return 1/(1 + np.exp(-arg))


# generating struct and random weights
struct = (2, 2, 1,)

Neurons, weights = generating_neurons(struct)

lmb = 0.1

# dataset
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]

studied = False

amount_of_cicles = 0
amount_right = 0

while not studied:
    amount_of_cicles += 1

    # random number of cur trained set
    cur_set = np.random.randint(0, len(inputs))

    output = forward(inputs[cur_set], struct, Neurons)

    break

print(output)
