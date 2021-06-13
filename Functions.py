import numpy as np


class Neu:
    def __init__(self, struct, weights, layer, current_neu, Neurons):
        self.value = None
        self.In = []  # weights which INcome in neuron
        # self.Out = [] #weights which OUTcome from neuron
        self.Parents = []

        if layer != 0:
            for i in range(struct[layer - 1]):
                self.In.append(weights[layer - 1][i][current_neu])
                self.Parents.append(Neurons[layer - 1][i])
        else:
            self.Parents = None
            self.In = None


def sigmoid(arg):
    return 1/(1 + np.exp(-arg))


def der(arg):
    return arg * (1 - arg)


#!!!
# return an array of arrays, where every neuron discribes like array of weights
#!!!
def generating_weights(struct):
    weights = []
    for layer in range(len(struct) - 1):
        weights.append([])
        for neu in range(struct[layer]):
            weights[layer].append([])
            for i in range(struct[layer + 1]):
                weights[layer][neu].append(np.random.randint(-100, 100)/100)

    return weights


def generating_net(struct):
    weights = generating_weights(struct)
    Neurons = []
    log = []
    for layer in range(len(struct)):
        Neurons.append([])
        for neuron in range(struct[layer]):
            new_neu = Neu(struct, weights, layer, neuron, Neurons)
            Neurons[layer].append(new_neu)
            if new_neu.In != None:
                log.append(new_neu.In)

    return Neurons, log  # in main code named weigths. 13 str


def forward(inp, struct, Neurons):
    # pushing input
    for i in range(len(Neurons[0])):
        Neurons[0][i].value = inp[i]

    for layer in range(1, len(struct)):
        for neuron in Neurons[layer]:
            val = 0

            # checking all inputs
            for weight in range(len(neuron.In)):
                val += neuron.In[weight] * neuron.Parents[weight].value

            neuron.value = sigmoid(val)

    return Neurons[len(struct) - 1][0].value


def back_prop(Neurons, e, struct, lmb):

    delta = e * der(Neurons[2][0].value)

    for i in range(len(Neurons[1])):
        Neurons[2][0].In[i] -= lmb * delta * Neurons[1][i].value

    for i in range(len(Neurons[1])):
        delta2 = delta * Neurons[2][0].In[i] * der(Neurons[1][i].value)
        for j in range(len(Neurons[0])):
            Neurons[1][i].In[j] -= lmb * delta2 * Neurons[0][j].value

    return Neurons
