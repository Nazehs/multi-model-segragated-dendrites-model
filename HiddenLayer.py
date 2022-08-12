from __future__ import print_function
import numpy as np
from LayerHidden import Layer


class hiddenLayer(Layer):
    def __init__(self, net, m, f_input_size, b_input_size, use_feedback_bias, update_feedback_weights, use_spiking_feedback, integration_time,
                 use_conductances, record_backprop_angle, k_B, g_D, use_backprop, kappas, mem, use_apical_conductance, g_A, g_L, g_B, dt, use_spiking_feedforward, P_hidden, lambda_max):
        '''
        Initialize the hidden layer.

        Arguments:
            net (Network)      : The network that the layer belongs to.
            m (int)            : The layer number, eg. m = 0 for the first hidden layer.
            f_input_size (int) : The size of feedforward input, eg. 784 for MNIST input.
            b_input_size (int) : The size of feedback input. This is the same as the
                                 the number of units in the next layer.
        '''

        Layer.__init__(self, net, m)
        self.f_input_size = f_input_size
        self.b_input_size = b_input_size
        self.mem = mem

        self.A_LEFT = np.zeros((self.size, 1))
        self.B_LEFT = np.zeros((self.size, 1))
        self.C_LEFT = np.zeros((self.size, 1))
        # LEFT INPUT
        self.A_RIGHT = np.zeros((self.size, 1))
        self.B_RIGHT = np.zeros((self.size, 1))
        self.C_RIGHT = np.zeros((self.size, 1))

        self.lambda_C_left = np.zeros((self.size, 1))
        self.lambda_C_right = np.zeros((self.size, 1))

        self.S_hist_left = np.zeros((self.size, self.mem), dtype=np.int8)

        self.S_hist_right = np.zeros((self.size, self.mem), dtype=np.int8)

        self.E_LEFT = np.zeros((self.size, 1))
        self.delta_W_left = np.zeros(self.net.W_left[self.m].shape)
        self.delta_Y_left = np.zeros(self.net.Y_left[self.m].shape)
        self.delta_b_left = np.zeros((self.size, 1))

        self.delta_W_right = np.zeros(self.net.W_right[self.m].shape)
        self.delta_Y_right = np.zeros(self.net.Y_right[self.m].shape)
        self.delta_b_right = np.zeros((self.size, 1))

        self.average_C_f_left = np.zeros((self.size, 1))
        self.average_C_t_left = np.zeros((self.size, 1))
        self.average_A_f_left = np.zeros((self.size, 1))
        self.average_A_t_left = np.zeros((self.size, 1))
        self.average_lambda_C_f_left = np.zeros((self.size, 1))
        self.average_PSP_B_f_left = np.zeros((self.f_input_size, 1))

        self.average_C_f_right = np.zeros((self.size, 1))
        self.average_C_t_right = np.zeros((self.size, 1))
        self.average_A_f_right = np.zeros((self.size, 1))
        self.average_A_t_right = np.zeros((self.size, 1))
        self.average_lambda_C_f_right = np.zeros((self.size, 1))
        self.average_PSP_B_f_right = np.zeros((self.f_input_size, 1))

        self.integration_time = integration_time
        self.use_spiking_feedforward = use_spiking_feedforward
        self.use_conductances = use_conductances
        self.P_hidden = P_hidden
        self.lambda_max = lambda_max
        self.record_backprop_angle = record_backprop_angle
        self.use_backprop = use_backprop
        self.update_feedback_weights = update_feedback_weights
        self.use_spiking_feedback = use_spiking_feedback
        self.use_feedback_bias = use_feedback_bias
        self.use_apical_conductance = use_apical_conductance
        self.kappas = kappas
        self.g_D = g_D
        self.g_L = g_L
        self.g_B = g_B
        self.g_A = g_A
        self.k_B = k_B
        self.dt = dt

        if update_feedback_weights:
            self.average_PSP_A_f_left = np.zeros((self.b_input_size, 1))
            self.average_PSP_A_f_right = np.zeros((self.b_input_size, 1))

        self.alpha_f_left = np.zeros((self.size, 1))
        self.alpha_f_right = np.zeros((self.size, 1))
        self.alpha_t_left = np.zeros((self.size, 1))
        self.alpha_t_right = np.zeros((self.size, 1))

        # set integration counter
        self.integration_counter = 0

        # create integration variables
        self.create_integration_vars()

    def create_integration_vars(self):
        self.A_hist_left = np.zeros((self.size, self.integration_time))
        self.A_hist_right = np.zeros((self.size, self.integration_time))
        self.PSP_A_hist_left = np.zeros(
            (self.b_input_size, self.integration_time))
        self.PSP_B_hist_left = np.zeros(
            (self.f_input_size, self.integration_time))
        self.PSP_A_hist_right = np.zeros(
            (self.b_input_size, self.integration_time))
        self.PSP_B_hist_right = np.zeros(
            (self.f_input_size, self.integration_time))
        self.C_hist_left = np.zeros((self.size, self.integration_time))
        self.C_hist_right = np.zeros((self.size, self.integration_time))
        self.lambda_C_hist_left = np.zeros((self.size, self.integration_time))
        self.lambda_C_hist_right = np.zeros((self.size, self.integration_time))

    def clear_vars(self):
        '''
        Clear all layer variables.
        '''

        # left input variable
        self.A_LEFT *= 0
        self.B_LEFT *= 0
        self.C_LEFT *= 0
        #  right input variable
        self.A_RIGHT *= 0
        self.B_RIGHT *= 0
        self.C_RIGHT *= 0

        self.lambda_C_left *= 0
        self.lambda_C_right *= 0

        self.S_hist_left *= 0
        self.S_hist_right *= 0
        self.A_hist_left *= 0
        self.A_hist_right *= 0
        self.PSP_A_hist_left *= 0
        self.PSP_B_hist_left *= 0
        self.PSP_A_hist_right *= 0
        self.PSP_B_hist_right *= 0
        self.C_hist_left *= 0
        self.C_hist_right *= 0
        self.lambda_C_hist_left *= 0
        self.lambda_C_hist_right *= 0
        self.E_LEFT *= 0
        self.E_RIGHT *= 0
        self.delta_W_left *= 0
        self.delta_Y_left *= 0
        self.delta_b_left *= 0

        self.delta_W_right *= 0
        self.delta_Y_right *= 0
        self.delta_b_right *= 0

        self.average_C_f_left *= 0
        self.average_C_t_left *= 0
        self.average_A_f_left *= 0
        self.average_A_t_left *= 0
        self.average_lambda_C_f_left *= 0
        self.average_PSP_B_f_left *= 0

        self.average_C_f_right *= 0
        self.average_C_t_right *= 0
        self.average_A_f_right *= 0
        self.average_A_t_right *= 0
        self.average_lambda_C_f_right *= 0
        self.average_PSP_B_f_right *= 0

        if self.update_feedback_weights:
            self.average_PSP_A_f_left *= 0
            self.average_PSP_A_f_right *= 0

        self.alpha_f_left *= 0
        self.alpha_f_right *= 0
        self.alpha_t_left *= 0
        self.alpha_t_right *= 0

        self.integration_counter = 0

    def update_W(self):
        '''
        Update feedforward weights.
        '''

        if not self.use_backprop:
            self.E_LEFT = (self.alpha_t_left - self.alpha_f_left)*-self.k_B * \
                self.lambda_max*self.net.deriv_sigma(self.average_C_f_left)

            self.E_RIGHT = (self.alpha_t_right - self.alpha_f_right)*-self.k_B * \
                self.lambda_max*self.net.deriv_sigma(self.average_C_f_right)

            if self.record_backprop_angle and not self.use_backprop and self.calc_E_bp_left:
                if self.m >= self.net.M - 2:
                    self.E_bp_left = (np.dot(
                        self.net.W_left[self.m+1].T, self.net.l[self.m+1].E_bp)*self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_left))
                    self.E_bp_right = (np.dot(
                        self.net.W_right[self.m+1].T, self.net.l[self.m+1].E_bp)*self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_right))
                else:
                    self.E_bp_left = (np.dot(
                        self.net.W_left[self.m+1].T, self.net.l[self.m+1].E_bp_left)*self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_left))
                    self.E_bp_right = (np.dot(
                        self.net.W_right[self.m+1].T, self.net.l[self.m+1].E_bp_right)*self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_right))

        else:
            # print("m", self.m, self.net.M)
            if self.m >= self.net.M - 2:

                self.E_bp_left = (np.dot(self.net.W[self.m+1].T, self.net.l[self.m+1].E_bp)
                                  * self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_left))
                self.E_LEFT = self.E_bp_left

                self.E_bp_right = (np.dot(self.net.W[self.m+1].T, self.net.l[self.m+1].E_bp)
                                   * self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_right))
                self.E_RIGHT = self.E_bp_right
            else:

                self.E_bp_left = (np.dot(self.net.W_left[self.m+1].T, self.net.l[self.m+1].E_bp_left)
                                  * self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_left))
                self.E_LEFT = self.E_bp_left

                self.E_bp_right = (np.dot(self.net.W_right[self.m+1].T, self.net.l[self.m+1].E_bp_right)
                                   * self.k_B*self.lambda_max*self.net.deriv_sigma(self.average_C_f_right))
                self.E_RIGHT = self.E_bp_right

        if self.record_backprop_angle and (not self.use_backprop):
            self.delta_b_bp_left = self.E_bp_left
            self.delta_b_bp_right = self.E_bp_right

        self.delta_W_left = np.dot(self.E_LEFT, self.average_PSP_B_f_left.T)
        # equation 10 - updating the weight of the layer using stochastic gradient descent
        # layer learning rate and weight matrix
        self.net.W_left[self.m] += -self.net.f_etas[self.m] * \
            self.P_hidden*self.delta_W_left

        self.delta_W_right = np.dot(self.E_RIGHT, self.average_PSP_B_f_right.T)
        self.net.W_right[self.m] += -self.net.f_etas[self.m] * \
            self.P_hidden*self.delta_W_right

        self.delta_b_left = self.E_LEFT
        self.net.b_left[self.m] += -self.net.f_etas[self.m] * \
            self.P_hidden*self.delta_b_left

        self.delta_b_right = self.E_RIGHT
        self.net.b_right[self.m] += -self.net.f_etas[self.m] * \
            self.P_hidden*self.delta_b_right

    def update_Y(self):
        '''
        Update feedback weights.
        '''

        E_inv_left = (self.lambda_max*self.net.sigma(self.average_C_f_left) -
                      self.alpha_f_left)*-self.net.deriv_sigma(self.average_A_f_left)

        self.delta_Y_left = np.dot(E_inv_left, self.average_PSP_A_f_left.T)
        self.net.Y_left[self.m] += -self.net.b_etas[self.m]*self.delta_Y_left

        E_inv_right = (self.lambda_max*self.net.sigma(self.average_C_f_right) -
                       self.alpha_f_right)*-self.net.deriv_sigma(self.average_A_f_right)

        self.delta_Y_right = np.dot(E_inv_right, self.average_PSP_A_f_right.T)
        self.net.Y_right[self.m] += -self.net.b_etas[self.m]*self.delta_Y_right

        if self.use_feedback_bias:
            self.delta_c_left = E_inv_left
            self.net.c_left[self.m] += - \
                self.net.b_etas[self.m]*self.delta_c_left

            self.delta_c_right = E_inv_right
            self.net.c_right[self.m] += - \
                self.net.b_etas[self.m]*self.delta_c_right

    def update_A(self, b_input_left, b_input_right):
        '''
        Update apical potentials.

        Arguments:
            b_input (ndarray) : Feedback input.
        '''

        if self.use_spiking_feedback:
            self.PSP_A_LEFT = np.dot(b_input_left, self.kappas)
            self.PSP_A_RIGHT = np.dot(b_input_right, self.kappas)
        else:
            self.PSP_A_LEFT = b_input_left
            self.PSP_A_RIGHT = b_input_right

        self.PSP_A_hist_left[:, self.integration_counter %
                             self.integration_time] = self.PSP_A_LEFT[:, 0]
        self.PSP_A_hist_right[:, self.integration_counter %
                              self.integration_time] = self.PSP_A_RIGHT[:, 0]

        if self.use_feedback_bias:
            self.A_LEFT = np.dot(self.net.Y_left[self.m],
                                 self.PSP_A_LEFT) + self.net.c_left[self.m]
            self.A_RIGHT = np.dot(self.net.Y_right[self.m],
                                  self.PSP_A_RIGHT) + self.net.c_right[self.m]
        else:
            # equation 2b - the synaptic weight from the output layer
            self.A_LEFT = np.dot(self.net.Y_left[self.m], self.PSP_A_LEFT)
            self.A_RIGHT = np.dot(self.net.Y_right[self.m], self.PSP_A_RIGHT)

        self.A_hist_left[:, self.integration_counter %
                         self.integration_time] = self.A_LEFT[:, 0]

        self.A_hist_right[:, self.integration_counter %
                          self.integration_time] = self.A_RIGHT[:, 0]

    def update_B(self, f_input_left, f_input_right):
        '''
        Update basal potentials.

        Arguments:
            f_input (ndarray) : Feedforward input.
        '''

        if self.use_spiking_feedforward:
            self.PSP_B_RIGHT = np.dot(f_input_right, self.kappas)
            self.PSP_B_LEFT = np.dot(f_input_left, self.kappas)
        else:
            self.PSP_B_LEFT = f_input_left
            self.PSP_B_RIGHT = f_input_left

        self.PSP_B_hist_left[:, self.integration_counter %
                             self.integration_time] = self.PSP_B_LEFT[:, 0]
        self.PSP_B_hist_right[:, self.integration_counter %
                              self.integration_time] = self.PSP_B_RIGHT[:, 0]

        # equation 2a - the synaptic weight from the input layer
        self.B_LEFT = np.dot(self.net.W_left[self.m],
                             self.PSP_B_LEFT) + self.net.b_left[self.m]
        self.B_RIGHT = np.dot(self.net.W_right[self.m],
                              self.PSP_B_RIGHT) + self.net.b_right[self.m]

    def update_C(self):
        '''
        Update somatic potentials & calculate firing rates.
        '''

        if self.use_conductances:
            if self.use_apical_conductance:

                self.C_dot_left = -self.g_L*self.C_LEFT + self.g_B * \
                    (self.B_LEFT - self.C_LEFT) + \
                    self.g_A*(self.A_LEFT - self.C_LEFT)

                self.C_dot_right = -self.g_L*self.C_RIGHT + self.g_B * \
                    (self.B_RIGHT - self.C_RIGHT) + \
                    self.g_A*(self.A_RIGHT - self.C_RIGHT)
            else:
                self.C_dot_left = -self.g_L*self.C_LEFT + \
                    self.g_B*(self.B_LEFT - self.C_LEFT)

                self.C_dot_right = -self.g_L*self.C_RIGHT + \
                    self.g_B*(self.B_RIGHT - self.C_RIGHT)
            # equation 1 - updating the neuron in the hidden layer
            self.C_LEFT += self.C_dot_left*self.dt
            # equation 1 - updating the neuron in the hidden layer
            self.C_RIGHT += self.C_dot_right*self.dt
        else:
            self.C_LEFT = self.k_B*self.B_LEFT
            self.C_RIGHT = self.k_B*self.B_RIGHT

        self.C_hist_left[:, self.integration_counter %
                         self.integration_time] = self.C_LEFT[:, 0]
        self.C_hist_right[:, self.integration_counter %
                          self.integration_time] = self.C_RIGHT[:, 0]

        # equation 3 - determine the rate of fire using a non-linear sigmoid function applied on the somatic
        self.lambda_C_left = self.lambda_max*self.net.sigma(self.C_LEFT)
        # print("self.lambda_C_left)", self.lambda_C_left)

        self.lambda_C_hist_left[:, self.integration_counter %
                                self.integration_time] = self.lambda_C_left[:, 0]

        self.lambda_C_right = self.lambda_max*self.net.sigma(self.C_RIGHT)

        self.lambda_C_hist_right[:, self.integration_counter %
                                 self.integration_time] = self.lambda_C_right[:, 0]

    def out_f(self, f_input_left, f_input_right, b_input):
        '''
        Perform a forward phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Feedback input.
        '''

        self.update_B(f_input_left, f_input_right)
        self.update_A(b_input, b_input)
        self.update_C()
        self.spike()

        self.integration_counter = (
            self.integration_counter + 1) % self.integration_time

    def out_t(self, f_input_left, f_input_right, b_input):
        '''
        Perform a target phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Feedback input.
        '''

        self.update_B(f_input_left, f_input_right)
        self.update_A(b_input, b_input)
        self.update_C()
        self.spike()

        self.integration_counter = (
            self.integration_counter + 1) % self.integration_time

    def plateau_f(self, plateau_indices):
        # print("self.A_hist_left", self.A_hist_left.shape, plateau_indices)
        '''
        Calculate forward phase apical plateau potentials.

        Arguments:
            plateau_indices (ndarray) : Indices of neurons that are undergoing apical plateau potentials.
        '''

        # calculate average apical potentials for neurons undergoing plateau potentials - equation 21
        self.average_A_f_left[plateau_indices] = np.mean(
            self.A_hist_left[plateau_indices], axis=-1)[:, np.newaxis]

        self.average_A_f_right[plateau_indices] = np.mean(
            self.A_hist_right[plateau_indices], axis=-1)[:, np.newaxis]

        # calculate apical calcium spike nonlinearity
        self.alpha_f_left[plateau_indices] = self.net.sigma(
            self.average_A_f_left[plateau_indices])

        self.alpha_f_right[plateau_indices] = self.net.sigma(
            self.average_A_f_right[plateau_indices])

    def plateau_t(self, plateau_indices):
        '''
        Calculate target phase apical plateau potentials.

        Arguments:
            plateau_indices (ndarray) : Indices of neurons that are undergoing apical plateau potentials.
        '''

        # calculate average apical potentials for neurons undergoing plateau potentials
        self.average_A_t_left[plateau_indices] = np.mean(
            self.A_hist_left[plateau_indices], axis=-1)[:, np.newaxis]

        self.average_A_t_right[plateau_indices] = np.mean(
            self.A_hist_right[plateau_indices], axis=-1)[:, np.newaxis]

        # calculate apical calcium spike nonlinearity
        self.alpha_t_left[plateau_indices] = self.net.sigma(
            self.average_A_t_left[plateau_indices])

        self.alpha_t_right[plateau_indices] = self.net.sigma(
            self.average_A_t_right[plateau_indices])

    def calc_averages(self, phase):
        '''
        Calculate averages of dynamic variables. This is done at the end of each
        forward & target phase.

        Arguments:
            phase (string) : Current phase of the network, "forward" or "target".
        '''

        if phase == "forward":
            self.average_C_f_left = np.mean(
                self.C_hist_left, axis=-1)[:, np.newaxis]

            self.average_C_f_right = np.mean(
                self.C_hist_right, axis=-1)[:, np.newaxis]

            self.average_lambda_C_f_left = np.mean(
                self.lambda_C_hist_left, axis=-1)[:, np.newaxis]

            self.average_lambda_C_f_right = np.mean(
                self.lambda_C_hist_right, axis=-1)[:, np.newaxis]

            self.average_PSP_B_f_left = np.mean(
                self.PSP_B_hist_left, axis=-1)[:, np.newaxis]

            self.average_PSP_B_f_right = np.mean(
                self.PSP_B_hist_right, axis=-1)[:, np.newaxis]

            if self.update_feedback_weights:

                self.average_PSP_A_f_left = np.mean(
                    self.PSP_A_hist_left, axis=-1)[:, np.newaxis]

                self.average_PSP_A_f_right = np.mean(
                    self.PSP_A_hist_right, axis=-1)[:, np.newaxis]

        elif phase == "target":
            self.average_C_t_left = np.mean(
                self.C_hist_left, axis=-1)[:, np.newaxis]

            self.average_C_t_right = np.mean(
                self.C_hist_right, axis=-1)[:, np.newaxis]

            self.average_lambda_C_t_left = np.mean(
                self.lambda_C_hist_left, axis=-1)[:, np.newaxis]

            self.average_lambda_C_t_right = np.mean(
                self.lambda_C_hist_right, axis=-1)[:, np.newaxis]

            if self.update_feedback_weights:
                self.average_PSP_A_t_left = np.mean(
                    self.PSP_A_hist_left, axis=-1)[:, np.newaxis]

                self.average_PSP_A_t_right = np.mean(
                    self.PSP_A_hist_right, axis=-1)[:, np.newaxis]
