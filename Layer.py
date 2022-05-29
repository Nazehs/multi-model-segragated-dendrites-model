import numpy as np


class Layer:
    def __init__(self, net, m):
        '''
        Initialize the layer.

        Arguments:
            net (Network) : The network that the layer belongs to.
            m (int)       : The layer number, eg. m = 0 for the first layer.
        '''

        self.net = net
        self.m = m
        self.size = self.net.n[m]

    def spike(self):
        '''
        Generate Poisson spikes based on the firing rates of the neurons.
        '''

        self.S_hist = np.concatenate(
            [self.S_hist[:, 1:], np.random.poisson(self.lambda_C)], axis=-1)
