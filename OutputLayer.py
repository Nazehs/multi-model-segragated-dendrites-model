from __future__ import print_function
import numpy as np
from Layer import Layer
"""
NOTE: In the paper, we denote the output layer's somatic & dendritic potentials
      as U and V. Here, we use C & B purely in order to simplify the code.
"""


class finalLayer(Layer):
    def __init__(self, net, m, f_input_size, integration_time, use_backprop,
                 use_conductances, record_backprop_angle, k_D, g_D, kappas, mem, dt,
                 E_E, g_L, E_I, use_spiking_feedforward, P_final, lambda_max):
        '''
        Initialize the final layer.

        Arguments:
            net (Network)      : The network that the layer belongs to.
            m (int)            : The layer number, ie. m = M - 1 where M is the total number of layers.
            f_input_size (int) : The size of feedforward input. This is the same as the
                                 the number of units in the previous layer.
        '''
        Layer.__init__(self, net, m)
        self.f_input_size = f_input_size
        self.mem = mem

        self.B = np.zeros((self.size, 1))
        self.I = np.zeros((self.size, 1))
        self.C = np.zeros((self.size, 1))
        self.lambda_C = np.zeros((self.size, 1))

        self.S_hist = np.zeros((self.size, self.mem), dtype=np.int8)

        self.E = np.zeros((self.size, 1))
        self.delta_W = np.zeros(self.net.W[self.m].shape)
        self.delta_b = np.zeros((self.size, 1))

        self.average_C_f = np.zeros((self.size, 1))
        self.average_C_t = np.zeros((self.size, 1))
        self.average_lambda_C_f = np.zeros((self.size, 1))
        self.average_lambda_C_t = np.zeros((self.size, 1))
        self.average_PSP_B_f = np.zeros((self.f_input_size, 1))
        self.integration_time = integration_time
        self.use_spiking_feedforward = use_spiking_feedforward
        self.use_conductances = use_conductances
        self.P_final = P_final
        self.lambda_max = lambda_max
        self.record_backprop_angle = record_backprop_angle
        self.use_backprop = use_backprop
        self.kappas = kappas
        # self.calc_E_bp = calc_E_bp
        self.k_D = k_D
        self.E_E = E_E
        self.g_D = g_D
        self.g_L = g_L
        self.E_I = E_I
        self.dt = dt
        # set integration counter
        self.integration_counter = 0

        # create integration variables
        self.create_integration_vars()

    def create_integration_vars(self):
        self.PSP_B_hist = np.zeros((self.f_input_size, self.integration_time))
        self.C_hist = np.zeros((self.size, self.integration_time))
        self.lambda_C_hist = np.zeros((self.size, self.integration_time))

    def clear_vars(self):
        '''
        Clear all layer variables.
        '''

        self.B *= 0
        self.I *= 0
        self.C *= 0
        self.lambda_C *= 0

        self.S_hist *= 0
        self.PSP_B_hist *= 0
        self.C_hist *= 0
        self.lambda_C_hist *= 0

        self.E *= 0
        self.delta_W *= 0
        self.delta_b *= 0

        self.average_C_f *= 0
        self.average_C_t *= 0
        self.average_lambda_C_f *= 0
        self.average_lambda_C_t *= 0
        self.average_PSP_B_f *= 0

        self.integration_counter = 0

    def update_W(self):
        '''
        Update feedforward weights.
        '''

        self.E = (self.average_lambda_C_t - self.lambda_max*self.net.sigma(self.average_C_f)
                  )*-self.k_D*self.lambda_max*self.net.deriv_sigma(self.average_C_f)

        if self.use_backprop or (self.record_backprop_angle):
            self.E_bp = (self.average_lambda_C_t - self.lambda_max*self.net.sigma(self.average_C_f)
                         )*-self.k_D*self.lambda_max*self.net.deriv_sigma(self.average_C_f)

        self.delta_W = np.dot(self.E, self.average_PSP_B_f.T)
        # equation 29 a - updating output layer weight using gradient descent
        self.net.W[self.m] += -self.net.f_etas[self.m] * \
            self.P_final*self.delta_W
        # equation 29 b for bias
        self.delta_b = self.E
        self.net.b[self.m] += -self.net.f_etas[self.m] * \
            self.P_final*self.delta_b

    def update_B(self, f_input):
        '''
        Update basal potentials.

        Arguments:
            f_input (ndarray) : Feedforward input.
        '''

        if self.use_spiking_feedforward:
            self.PSP_B = np.dot(f_input, self.kappas)
        else:
            self.PSP_B = f_input

        self.PSP_B_hist[:, self.integration_counter %
                        self.integration_time] = self.PSP_B[:, 0]
        # equation 2a - the synaptic weight from the input layer
        self.B = np.dot(self.net.W[self.m], self.PSP_B) + self.net.b[self.m]

    def update_I(self, b_input=None):
        '''
        Update injected perisomatic currents.

        Arguments:
            b_input (ndarray) : Target input, eg. if the target label is 8,
                                b_input = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).
        '''

        if b_input is None:
            self.I *= 0
        else:
            self.g_E = b_input
            self.g_I = -self.g_E + 1
            if self.use_conductances:
                # equation 20 - a somatic current aimed at influencing the output neuron towards desired somatic voltage
                self.I = self.g_E*(self.E_E - self.C) + \
                    self.g_I*(self.E_I - self.C)
            else:
                self.k_D2 = self.g_D / \
                    (self.g_L + self.g_D + self.g_E + self.g_I)
                self.k_E = self.g_E/(self.g_L + self.g_D + self.g_E + self.g_I)
                self.k_I = self.g_I/(self.g_L + self.g_D + self.g_E + self.g_I)

    def update_C(self, phase):
        '''
        Update somatic potentials & calculate firing rates.

        Arguments:
            phase (string) : Current phase of the network, "forward" or "target".
        '''

        if self.use_conductances:
            if phase == "forward":
                self.C_dot = -self.g_L*self.C + self.g_D*(self.B - self.C)
            elif phase == "target":
                # equation 19 - the filtered presynaptic spike trains at synapses that receives feedforward input from the hidden layer
                self.C_dot = -self.g_L*self.C + \
                    self.g_D*(self.B - self.C) + self.I
            self.C += self.C_dot*self.dt
        else:
            if phase == "forward":
                self.C = self.k_D*self.B
            elif phase == "target":
                self.C = self.k_D2*self.B + self.k_E*self.E_E + self.k_I*self.E_I

        self.C_hist[:, self.integration_counter %
                    self.integration_time] = self.C[:, 0]

        self.lambda_C = self.lambda_max*self.net.sigma(self.C)
        self.lambda_C_hist[:, self.integration_counter %
                           self.integration_time] = self.lambda_C[:, 0]

    def out_f(self, f_input, b_input):
        '''
        Perform a forward phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Target input. b_input = None during this phase.
        '''

        self.update_B(f_input)
        self.update_I(b_input)
        self.update_C(phase="forward")
        self.spike()

        self.integration_counter = (
            self.integration_counter + 1) % self.integration_time

    def out_t(self, f_input, b_input):
        '''
        Perform a target phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Target input.
        '''

        self.update_B(f_input)
        self.update_I(b_input)
        self.update_C(phase="target")
        self.spike()

        self.integration_counter = (
            self.integration_counter + 1) % self.integration_time

    def calc_averages(self, phase):
        '''
        Calculate averages of dynamic variables. This is done at the end of each
        forward & target phase.

        Arguments:
            phase (string) : Current phase of the network, "forward" or "target".
        '''

        if phase == "forward":
            self.average_C_f = np.mean(self.C_hist, axis=-1)[:, np.newaxis]

            self.average_lambda_C_f = np.mean(
                self.lambda_C_hist, axis=-1)[:, np.newaxis]
            self.average_PSP_B_f = np.mean(
                self.PSP_B_hist, axis=-1)[:, np.newaxis]
        elif phase == "target":
            self.average_C_t = np.mean(self.C_hist, axis=-1)[:, np.newaxis]
            self.average_lambda_C_t = np.mean(
                self.lambda_C_hist, axis=-1)[:, np.newaxis]
