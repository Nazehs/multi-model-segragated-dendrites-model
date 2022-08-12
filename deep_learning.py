'''
This is a master's thesis that extended the paper of "Towards deep learning with segregated dendrites", arXiv:1610.00161
     Author: Nazeh Abel
     E-mail: nazeaabel@gmail.com
       Date: May 29, 2022
Institution: University of Wolverhampton, UK
Copyright (C) 2022 Nazeh Abel
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import copy
import datetime
import os
import pdb
import sys
import time
import shutil
import json
from scipy.special import expit
from HiddenLayer import hiddenLayer

from OutputLayer import finalLayer

if sys.version_info >= (3,):
    xrange = range

n_full_test = 10000  # number of examples to use for full tests  (every epoch)
# number of examples to use for quick tests (every 1000 examples)
n_quick_test = 100
# whether to run in non-spiking mode (real-valued outputs)
nonspiking_mode = True

# use random phase lengths (chosen from Wald distribution)
use_rand_phase_lengths = True
# randomly sample the time of each neuron's apical plateau potential
use_rand_plateau_times = False
use_conductances = True  # use conductances between dendrites and soma
# use broadcast (ie. feedback to all layers comes from output layer)
use_broadcast = True
use_spiking_feedback = True  # use spiking feedback
use_spiking_feedforward = True  # use spiking feedforward input

use_symmetric_weights = False  # enforce symmetric weights
noisy_symmetric_weights = False  # add noise to symmetric weights

use_sparse_feedback = False  # use sparse feedback weights
update_feedback_weights = False  # update feedback weights
use_backprop = False  # use error backpropagation
# use attenuated conductance from apical dendrite to soma
use_apical_conductance = False
use_weight_optimization = True  # attempt to optimize initial weights
use_feedback_bias = False  # use biases in feedback paths
initial_test = False  # whether to do an initial test on the test set prior to training

# record angle b/w hidden layer error signals and backprop-generated error signals
record_backprop_angle = False
record_loss = True  # record final layer loss during training
record_training_error = True  # record training error during training
# record labels of images that were shown during training
record_training_labels = True
record_phase_times = False  # record phase transition times across training
# record plateau potential times for each neuron across training
record_plateau_times = False
# record voltages of neurons during training (huge arrays for long simulations!)
record_voltages = False

# --- Jacobian testing --- #
record_eigvals = False  # record maximum eigenvalues for Jacobians
# record Jacobian product & weight product matrices (huge arrays for long simulations!)
record_matrices = False
plot_eigvals = False  # dynamically plot maximum eigenvalues for Jacobians

# folder in which to save simulations (edit accordingly)
default_simulations_folder = 'Simulations/'
weight_cmap = 'bone'         # color map to use for weight plotting

dt = 1.0        # time step (ms)
# spike memory (time steps) - used to limit PSP integration of past spikes (for performance)
mem = int(10/dt)

l_f_phase = int(50/dt)  # length of forward phase (time steps)
l_t_phase = int(50/dt)  # length of target phase (time steps)
l_f_phase_test = int(250/dt)  # length of forward phase for tests (time steps)

# time steps of integration of neuronal variables used for plasticity
integration_time = l_f_phase - int(30/dt)
# time steps of integration of neuronal variables during testing
integration_time_test = l_f_phase_test - int(30/dt)

if nonspiking_mode:
    print("* ------------ Running in non-spiking mode. ------------ *")

    # set parameters for non-spiking mode
    use_rand_phase_lengths = False
    use_rand_plateau_times = False
    use_conductances = False
    use_spiking_feedforward = False
    use_spiking_feedback = False
    record_phase_times = False
    record_plateau_times = False
    record_voltages = False

    l_f_phase = 2
    l_t_phase = 2
    l_f_phase_test = 2
    integration_time = 1
    integration_time_test = 1
    mem = 1

if use_rand_phase_lengths:
    # set minimum phase lengths
    min_l_f_phase = l_f_phase
    min_l_t_phase = l_t_phase

lambda_max = 0.2*dt  # maximum spike rate (spikes per time step)

# kernel parameters
tau_s = 3.0  # synaptic time constant
tau_L = 10.0  # leak time constant

# conductance parameters
g_B = 0.6                                   # basal conductance
g_A = 0.05 if use_apical_conductance else 0  # apical conductance
g_L = 1.0/tau_L                             # leak conductance
g_D = g_B                                   # dendritic conductance in output layer

E_E = 8  # excitation reversal potential
E_I = -8  # inhibition reversal potential

# steady state constants
k_B = g_B/(g_L + g_B + g_A)
k_D = g_D/(g_L + g_D)
k_I = 1.0/(g_L + g_D)

# weight update constants
P_hidden = 20.0/lambda_max      # hidden layer error signal scaling factor
P_final = 20.0/(lambda_max**2)  # final layer error signal scaling factor
# ---------------------------------------------------------------
"""                     Functions                             """
# ---------------------------------------------------------------

# --- sigmoid function --- #


def sigma(x):
    return expit(x)


def deriv_sigma(x):
    return expit(x)*(1.0 - expit(x))

# --- kernel function --- #


def kappa(x):
    return (np.exp(-x/tau_L) - np.exp(-x/tau_s))/(tau_L - tau_s)


def get_kappas(n=mem):
    return np.array([kappa(i+1) for i in xrange(n)])


kappas = np.flipud(get_kappas(mem))[:, np.newaxis]  # initialize kappas array

# ---------------------------------------------------------------
"""                     Network class                         """
# ---------------------------------------------------------------


class Network:
    def __init__(self, n):
        '''
        Initialize the network. Note: This also loads the MNIST dataset.

        Arguments:
            n (tuple) : Number of units in each layer of the network, eg. (500, 100, 10).
        '''

        if type(n) == int:
            n = (n,)

        self.n = n           # layer sizes - eg. (500, 100, 10)
        self.M = len(self.n)  # number of layers

        self.n_neurons_per_category = int(self.n[-1]/10)

        # load MNIST
        self.x_train, self.t_train, self.x_test, self.t_test = load_MNIST()

        self.x_train_input_left = self.x_train[:self.x_train.shape[0]//2, :]
        self.x_train_input_right = self.x_train[self.x_train.shape[0]//2:, :]

        self.n_in = self.x_train_input_left.shape[0]  # input size
        self.n_out = self.n[-1]            # output size

        # initialize input spike history for network with atleast 1 hidden layer
        self.x_hist_left = np.zeros((self.n_in, mem))
        self.x_hist_right = np.zeros((self.n_in, mem))
        # for a network without any hidden layer
        if self.M == 1:
            self.x_hist = np.zeros((self.n_in * 2, mem))
        self.current_epoch = None  # current epoch of simulation

        print("Creating network with {} layers.".format(self.M))
        print("--------------------------------")

        self.init_weights()
        self.init_layers()

    def init_weights(self):
        '''
        Initialize the weights of the network.
        '''

        if use_weight_optimization:
            # initial weight optimization parameters
            V_avg = 3                  # desired average of dendritic potential
            V_sd = 3                  # desired standard deviation of dendritic potential
            b_avg = 0.8                # desired average of bias
            b_sd = 0.001              # desired standard deviation of bias
            nu = lambda_max*0.25    # slope of linear region of activation function
            V_sm = V_sd**2 + V_avg**2  # second moment of dendritic potential

        # initialize lists of weight matrices & bias vectors
        self.W = [0]*self.M
        self.b = [0]*self.M
        self.Y = [0]*(self.M-1)

        self.W_left = [0]*self.M
        self.b_left = [0]*self.M
        self.Y_left = [0]*(self.M-1)

        self.W_right = [0]*self.M
        self.b_right = [0]*self.M
        self.Y_right = [0]*(self.M-1)

        if use_feedback_bias:
            self.c = [0]*(self.M-1)
            self.c_left = [0]*(self.M-1)
            self.c_right = [0]*(self.M-1)

        if use_sparse_feedback:
            # initialize list of indices of zeroed-out weights
            self.Y_dropout_indices = [0]*(self.M-1)
            self.Y_dropout_indices_left = [0]*(self.M-1)
            self.Y_dropout_indices_right = [0]*(self.M-1)

        for m in xrange(self.M-1, -1, -1):
            # get number of units in the layer below
            if m != 0:
                N = self.n[m-1]
                if self.M == 1:
                    N = self.n[m-1] * 2
            else:
                N = self.n_in

            # generate feedforward weights & biases
            if use_weight_optimization:
                # calculate weight variables needed to get desired average & standard deviations of somatic potentials
                W_avg = (V_avg - b_avg)/(nu*N*V_avg)
                W_sm = (V_sm + (nu**2)*(N - N**2)*(W_avg**2)*(V_avg**2) -
                        2*N*nu*b_avg*V_avg*W_avg - (b_avg**2))/(N*(nu**2)*V_sm)
                W_sd = np.sqrt(W_sm - W_avg**2)
                if self.M == 1:
                    self.W[m] = W_avg + 3.465*W_sd * \
                        np.random.uniform(-1, 1, size=(self.n[m], N * 2))
                    self.b[m] = b_avg + 3.465*b_sd * \
                        np.random.uniform(-1, 1, size=(self.n[m], 1))
                else:
                    self.b[m] = b_avg + 3.465*b_sd * \
                        np.random.uniform(-1, 1, size=(self.n[m], 1))
                    self.W[m] = W_avg + 3.465*W_sd * \
                        np.random.uniform(-1, 1, size=(self.n[m], N))

                self.W_left[m] = W_avg + 3.465*W_sd * \
                    np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b_left[m] = b_avg + 3.465*b_sd * \
                    np.random.uniform(-1, 1, size=(self.n[m], 1))

                self.W_right[m] = W_avg + 3.465*W_sd * \
                    np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b_right[m] = b_avg + 3.465*b_sd * \
                    np.random.uniform(-1, 1, size=(self.n[m], 1))
            else:
                self.W_left[m] = 0.1 * \
                    np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b_left[m] = 1.0 * \
                    np.random.uniform(-1, 1, size=(self.n[m], 1))

                self.W_right[m] = 0.1 * \
                    np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b_right[m] = 1.0 * \
                    np.random.uniform(-1, 1, size=(self.n[m], 1))

                if self.M == 1:
                    self.W[m] = W_avg + 3.465*W_sd * \
                        np.random.uniform(-1, 1, size=(self.n[m], N * 2))
                    self.b[m] = b_avg + 3.465*b_sd * \
                        np.random.uniform(-1, 1, size=(self.n[m], 1))
                else:
                    self.W[m] = 0.1 * \
                        np.random.uniform(-1, 1, size=(self.n[m], N))
                    self.b[m] = 1.0 * \
                        np.random.uniform(-1, 1, size=(self.n[m], 1))

            # generate feedback weights & biases; in the paper, we do not use feedback biases
            if m != 0:

                if use_broadcast:
                    if use_weight_optimization:
                        self.Y[m-1] = W_avg + 3.465*W_sd * \
                            np.random.uniform(-1, 1, size=(N, self.n[-1]))
                        self.Y_left[m-1] = W_avg + 3.465*W_sd * \
                            np.random.uniform(-1, 1, size=(N, self.n[-1]))
                        self.Y_right[m-1] = W_avg + 3.465*W_sd * \
                            np.random.uniform(-1, 1, size=(N, self.n[-1]))

                        if use_feedback_bias:
                            self.c[m-1] = b_avg + 3.465*b_sd * \
                                np.random.uniform(-1, 1, size=(N, 1))
                            self.c_left[m-1] = b_avg + 3.465*b_sd * \
                                np.random.uniform(-1, 1, size=(N, 1))
                            self.c_right[m-1] = b_avg + 3.465*b_sd * \
                                np.random.uniform(-1, 1, size=(N, 1))
                    else:
                        self.Y[m -
                               1] = np.random.uniform(-1, 1, size=(N, self.n[-1]))

                        self.Y_left[m -
                                    1] = np.random.uniform(-1, 1, size=(N, self.n[-1]))
                        self.Y_right[m -
                                     1] = np.random.uniform(-1, 1, size=(N, self.n[-1]))

                        if use_feedback_bias:
                            self.c[m-1] = np.random.uniform(-1, 1, size=(N, 1))
                            self.c_left[m -
                                        1] = np.random.uniform(-1, 1, size=(N, 1))
                            self.c_right[m -
                                         1] = np.random.uniform(-1, 1, size=(N, 1))
                else:
                    if use_weight_optimization:
                        self.Y[m-1] = W_avg + 3.465*W_sd * \
                            np.random.uniform(-1, 1, size=(N, self.n[m]))

                        self.Y_left[m-1] = W_avg + 3.465*W_sd * \
                            np.random.uniform(-1, 1, size=(N, self.n[m]))
                        self.Y_right[m-1] = W_avg + 3.465*W_sd * \
                            np.random.uniform(-1, 1, size=(N, self.n[m]))

                        if use_feedback_bias:
                            self.c[m-1] = b_avg + 3.465*b_sd * \
                                np.random.uniform(-1, 1, size=(N, 1))
                            self.c_left[m-1] = b_avg + 3.465*b_sd * \
                                np.random.uniform(-1, 1, size=(N, 1))
                            self.c_right[m-1] = b_avg + 3.465*b_sd * \
                                np.random.uniform(-1, 1, size=(N, 1))
                    else:

                        self.Y[m -
                               1] = np.random.uniform(-1, 1, size=(N, self.n[m]))

                        self.Y_left[m -
                                    1] = np.random.uniform(-1, 1, size=(N, self.n[m]))
                        self.Y_right[m -
                                     1] = np.random.uniform(-1, 1, size=(N, self.n[m]))

                        if use_feedback_bias:
                            self.c[m -
                                   1] = np.random.uniform(-1, 1, size=(N, 1))
                            self.c_left[m -
                                        1] = np.random.uniform(-1, 1, size=(N, 1))
                            self.c_right[m -
                                         1] = np.random.uniform(-1, 1, size=(N, 1))

        if use_symmetric_weights:
            # enforce symmetric weights
            self.make_weights_symmetric()

        if use_sparse_feedback:
            # randomly zero out 80% of weights, increase magnitude of surviving weights to keep desired average voltages
            for m in xrange(self.M-1):
                self.Y_dropout_indices[m] = np.random.choice(
                    len(self.Y[m].ravel()), int(0.8*len(self.Y[m].ravel())), False)

                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0
                self.Y[m] *= 5

                self.Y_dropout_indices_left[m] = np.random.choice(
                    len(self.Y_left[m].ravel()), int(0.8*len(self.Y_left[m].ravel())), False)
                self.Y_left[m].ravel()[self.Y_dropout_indices_left[m]] = 0
                self.Y_left[m] *= 5

                self.Y_dropout_indices_right[m] = np.random.choice(
                    len(self.Y_right[m].ravel()), int(0.8*len(self.Y_right[m].ravel())), False)
                self.Y_right[m].ravel()[
                    self.Y_dropout_indices_right[m]] = 0
                self.Y_right[m] *= 5

        # print initial weights info
        self.print_weights()

        print("--------------------------------")

    def print_weights(self):
        print("Current network weights:")
        print("--------------------------------")
        for m in xrange(self.M-1, -1, -1):
            if m >= self.M - 1:
                print("Layer {0} -- {1} units.".format(m, self.n[m]))
                print("\tW_avg: {0:.6f},\tW_sd: {1:.6f}.".format(
                    np.mean(self.W[m]), np.std(self.W[m])))
                print("\tb_avg: {0:.6f},\tb_sd: {1:.6f}.".format(
                    np.mean(self.b[m]), np.std(self.b[m])))
            else:
                print("Layer {0} -- {1} units.".format(m, self.n[m]))
                print("\tW_avg_left: {0:.6f},\tW_sd_left: {1:.6f}.".format(
                    np.mean(self.W_left[m]), np.std(self.W_left[m])))
                print("\tb_avg_left: {0:.6f},\tb_sd_left: {1:.6f}.".format(
                    np.mean(self.b_left[m]), np.std(self.b_left[m])))
                print("\tW_avg_right: {0:.6f},\tW_sd_right: {1:.6f}.".format(
                    np.mean(self.W_right[m]), np.std(self.W_right[m])))
                print("\tb_avg_right: {0:.6f},\tb_sd_right: {1:.6f}.".format(
                    np.mean(self.b_right[m]), np.std(self.b_right[m])))
            if m != self.M-1:
                print("\tY_avg_left: {0:.6f},\tY_sd_left: {1:.6f}.".format(
                    np.mean(self.Y_left[m]), np.std(self.Y_left[m])))
                print("\tY_avg_right: {0:.6f},\tY_sd_right: {1:.6f}.".format(
                    np.mean(self.Y_right[m]), np.std(self.Y_right[m])))
                if use_feedback_bias:
                    print("\tc_avg: {0:.6f},\tc_sd: {1:.6f}.".format(
                        np.mean(self.c_left[m]), np.std(self.c_left[m])))

    def make_weights_symmetric(self):
        '''
        Make the feedback weights in the network symmetric to (transposes of) the feedforward weights.
        '''

        if use_broadcast:
            for m in xrange(self.M-2, -1, -1):
                # make a copy if we're altering the feedback weights after
                if use_sparse_feedback:
                    W_above = self.W[m+1].T.copy()
                    W_above_left = self.W_left[m+1].T.copy()
                    W_above_right = self.W_right[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T
                    W_above_left = self.W_left[m+1].T
                    W_above_right = self.W_right[m+1].T

                if m == self.M - 2:
                    # for final hidden layer, use feedforward weights of output layer
                    if noisy_symmetric_weights:
                        self.Y[m] = W_above + \
                            np.random.normal(0, 0.05, size=W_above.shape)
                        self.Y_left[m] = W_above_left + \
                            np.random.normal(0, 0.05, size=W_above.shape)
                        self.Y_right[m] = W_above_right + \
                            np.random.normal(0, 0.05, size=W_above.shape)
                    else:
                        self.Y[m] = W_above
                        self.Y_left[m] = W_above_left
                        self.Y_right[m] = W_above_right
                else:
                    # for other hidden layers, use product of all feedforward weights downstream
                    if noisy_symmetric_weights:
                        self.Y[m] = np.dot(
                            W_above + np.random.normal(0, 0.05, size=W_above.shape), self.Y[m+1])
                        self.Y_left[m] = np.dot(
                            W_above_left + np.random.normal(0, 0.05, size=W_above_left.shape), self.Y_left[m+1])
                        self.Y_right[m] = np.dot(
                            W_above_right + np.random.normal(0, 0.05, size=W_above_right.shape), self.Y_right[m+1])
                    else:
                        self.Y[m] = np.dot(W_above, self.Y[m+1])
                        self.Y_left[m] = np.dot(W_above_left, self.Y_left[m+1])
                        self.Y_right[m] = np.dot(
                            W_above_right, self.Y_right[m+1])
        else:
            for m in xrange(self.M-2, -1, -1):
                # make a copy if we're altering the feedback weights after
                if use_sparse_feedback:
                    W_above = self.W[m+1].T.copy()
                    W_above_left = self.W_left[m+1].T.copy()
                    W_above_right = self.W_right[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T
                    W_above_left = self.W_left[m+1].T
                    W_above_right = self.W_right[m+1].T

                # use feedforward weights of the layer downstream
                if noisy_symmetric_weights:
                    self.Y[m] = W_above + np.random.normal(0, 0.05)
                    self.Y_left[m] = W_above_left + np.random.normal(0, 0.05)
                    self.Y_right[m] = W_above_right + np.random.normal(0, 0.05)
                else:
                    self.Y_left[m] = W_above_left
                    self.Y_right[m] = W_above_right
                    self.Y[m] = W_above

    def init_layers(self):
        '''
        Create the layers of the network.
        '''
        # initialize layers list
        self.l = []

        # create all layers
        if self.M == 1:
            self.l.append(finalLayer(net=self, m=-1,
                          f_input_size=self.n_in * 2, mem=mem, dt=dt, g_D=g_D, E_E=E_E, g_L=g_L, k_D=k_D, E_I=E_I, record_backprop_angle=record_backprop_angle, integration_time=integration_time,
                          lambda_max=lambda_max, P_final=P_final,   use_spiking_feedforward=use_spiking_feedforward, kappas=kappas,
                                     use_backprop=use_backprop, use_conductances=use_conductances))
        else:
            if use_broadcast:
                self.l.append(hiddenLayer(
                    net=self, m=0, f_input_size=self.n_in, b_input_size=self.n[-1], use_feedback_bias=use_feedback_bias, mem=mem, dt=dt, g_B=g_B, g_A=g_A, k_B=k_B, g_D=g_D, g_L=g_L,   integration_time=integration_time,
                    use_spiking_feedforward=use_spiking_feedforward, kappas=kappas, use_apical_conductance=use_apical_conductance,
                    lambda_max=lambda_max, P_hidden=P_hidden, record_backprop_angle=record_backprop_angle,
                    use_spiking_feedback=use_spiking_feedback, update_feedback_weights=update_feedback_weights, use_backprop=use_backprop, use_conductances=use_conductances))
                for m in xrange(1, self.M-1):
                    self.l.append(hiddenLayer(
                        net=self, m=m, f_input_size=self.n[m -
                                                           1], b_input_size=self.n[-1], mem=mem, dt=dt, kappas=kappas, use_feedback_bias=use_feedback_bias, g_B=g_B, g_D=g_D, g_L=g_L, g_A=g_A, k_B=k_B,  integration_time=integration_time,
                        use_spiking_feedforward=use_spiking_feedforward, use_apical_conductance=use_apical_conductance,
                        lambda_max=lambda_max, P_hidden=P_hidden, record_backprop_angle=record_backprop_angle,
                        use_spiking_feedback=use_spiking_feedback, update_feedback_weights=update_feedback_weights, use_backprop=use_backprop, use_conductances=use_conductances))
            else:
                self.l.append(hiddenLayer(
                    net=self, m=0, f_input_size=self.n_in, b_input_size=self.n[1], mem=mem, dt=dt, kappas=kappas, use_feedback_bias=use_feedback_bias, g_B=g_B,  g_D=g_D, g_L=g_L,  g_A=g_A, k_B=k_B,  integration_time=integration_time,
                    use_spiking_feedforward=use_spiking_feedforward, use_apical_conductance=use_apical_conductance,
                    lambda_max=lambda_max, P_hidden=P_hidden, record_backprop_angle=record_backprop_angle,
                    use_spiking_feedback=use_spiking_feedback, update_feedback_weights=update_feedback_weights, use_backprop=use_backprop, use_conductances=use_conductances))
                for m in xrange(1, self.M-1):
                    self.l.append(hiddenLayer(
                        net=self, m=m, f_input_size=self.n[m-1], b_input_size=self.n[m +
                                                                                     1], kappas=kappas, mem=mem, dt=dt, use_feedback_bias=use_feedback_bias, g_B=g_B, g_A=g_A, k_B=k_B, g_D=g_D, g_L=g_L,    integration_time=integration_time,
                        use_spiking_feedforward=use_spiking_feedforward, use_apical_conductance=use_apical_conductance,
                        lambda_max=lambda_max, P_hidden=P_hidden, record_backprop_angle=record_backprop_angle,
                        use_spiking_feedback=use_spiking_feedback, update_feedback_weights=update_feedback_weights, use_backprop=use_backprop, use_conductances=use_conductances))
            self.l.append(finalLayer(net=self, m=self.M -
                          1, f_input_size=self.n[-2], mem=mem, dt=dt, g_D=g_D, E_E=E_E, g_L=g_L, k_D=k_D, E_I=E_I, kappas=kappas,
                          record_backprop_angle=record_backprop_angle, integration_time=integration_time,
                                     use_spiking_feedforward=use_spiking_feedforward,
                          lambda_max=lambda_max, P_final=P_final,
                                     use_backprop=use_backprop, use_conductances=use_conductances))

    def out_f(self, training=False):
        '''
        Perform a forward phase pass through the network.

        Arguments:
            training (bool) : Whether the network is in training (True) or testing (False) mode.
        '''

        if use_spiking_feedforward:
            x_left = self.x_hist_left
            x_right = self.x_hist_right
            if self.M == 1:
                x = self.x_hist
        else:
            x_left = self.x_left
            x_right = self.x_right
            if self.M == 1:
                x = np.concatenate([x_left, x_right], axis=0)

        if self.M == 1:
            self.l[0].out_f(x, None)
        else:
            if use_broadcast:
                if use_spiking_feedback:
                    # taking the left and right input
                    self.l[0].out_f(x_left, x_right, self.l[-1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            # taking the left and right input
                            self.l[m].out_f(self.l[m-1].S_hist,
                                            self.l[-1].S_hist)
                        else:
                            # taking the left and right input
                            self.l[m].out_f(
                                self.l[m-1].lambda_C, self.l[-1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(np.sum([self.l[-2].S_hist_left,
                                         self.l[-2].S_hist_right], axis=0), None)
                    else:
                        self.l[-1].out_f(np.sum([self.l[-2].lambda_C_left,
                                         self.l[-2].lambda_C_right], axis=0), None)
                else:
                    # taking the left and right input
                    self.l[0].out_f(x_left, x_right, self.l[-1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_f(self.l[m-1].S_hist,
                                            self.l[-1].lambda_C)
                        else:
                            # taking the left and right input -- important
                            self.l[m].out_f(self.l[m-1].lambda_C_left, self.l[m-1].lambda_C_right,
                                            self.l[-1].lambda_C)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(np.sum([self.l[-2].S_hist_left,
                                         self.l[-2].S_hist_right], axis=0), None)
                    else:
                        # concatenating the left and right input -- important
                        self.l[-1].out_f(np.sum([self.l[-2].lambda_C_left,
                                                 self.l[-2].lambda_C_right], axis=0), None)
            else:
                if use_spiking_feedback:
                    self.l[0].out_f(x_left, x_right, self.l[1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_f(self.l[m-1].S_hist,

                                            self.l[m+1].S_hist)
                        else:
                            self.l[m].out_f(
                                self.l[m-1].lambda_C, self.l[m+1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(
                            self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(
                            self.l[-2].lambda_C, None)
                else:
                    # taking the left and right input
                    self.l[0].out_f(x_left, x_right, self.l[1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_f(self.l[m-1].S_hist,
                                            self.l[m+1].lambda_C)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C,
                                            self.l[m+1].lambda_C)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(
                            self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(
                            self.l[-2].lambda_C, None)

    def out_t(self):
        '''
        Perform a target phase pass through the network. This is the same as a forward phase pass,
        but with a target introduced at the top layer.
        '''

        # same as forward pass, but with a target introduced at the top layer
        if use_spiking_feedforward:
            x_left = self.x_hist_left
            x_right = self.x_hist_right
            if self.M == 1:
                x = self.x_hist
        else:
            x_left = self.x_left
            x_right = self.x_right
            if self.M == 1:
                x = np.concatenate([x_left, x_right], axis=0)
        if self.M == 1:
            self.l[0].out_t(x, self.t)
        else:
            if use_broadcast:
                if use_spiking_feedback:
                    self.l[0].out_t(x_left, x_right, self.l[-1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[-1].S_hist)
                        else:
                            self.l[m].out_t(
                                self.l[m-1].lambda_C, self.l[-1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(np.concatenate([self.l[-2].S_hist_left,
                                         self.l[-2].S_hist_right], axis=0), self.t)
                    else:
                        self.l[-1].out_t(np.concatenate([self.l[-2].lambda_C_left,
                                         self.l[-2].lambda_C_right], axis=0), self.t)
                else:
                    # taking the left and right input of the data
                    self.l[0].out_t(x_left, x_right, self.l[-1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[-1].lambda_C)
                        else:
                            # taking the left and right of the input
                            self.l[m].out_t(self.l[m-1].lambda_C_left, self.l[m-1].lambda_C_right,
                                            self.l[-1].lambda_C)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(np.sum([self.l[-2].S_hist_left,
                                         self.l[-2].S_hist_right], axis=0), self.t)
                    else:
                        # concatenating the left and right
                        self.l[-1].out_t(np.sum([self.l[-2].lambda_C_left,
                                         self.l[-2].lambda_C_right], axis=0), self.t)
            else:
                if use_spiking_feedback:
                    self.l[0].out_t(x_left, x_right, self.l[1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[m+1].S_hist)
                        else:
                            self.l[m].out_t(
                                self.l[m-1].lambda_C, self.l[m+1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)
                else:
                    self.l[0].out_t(x_left, x_right, self.l[1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[m+1].lambda_C)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C,
                                            self.l[m+1].lambda_C)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)

    def f_phase(self, x_left, x_right,  training_num, training=False):
        '''
        Perform a forward phase.

        Arguments:
            x (ndarray)        : Input array of size (X, 1) where X is the size of the input, eg. (784, 1).
            training_num (int) : Number (from start of the epoch) of the training example being shown.
            training (bool)    : Whether the network is in training (True) or testing (False) mode.
        '''

        if record_voltages and training:
            # initialize voltage arrays
            self.A_hists = [np.zeros((l_f_phase, self.l[m].size))
                            for m in xrange(self.M-1)]
            self.B_hists = [np.zeros((l_f_phase, self.l[m].size))
                            for m in xrange(self.M)]
            self.C_hists = [np.zeros((l_f_phase, self.l[m].size))
                            for m in xrange(self.M)]

        for time in xrange(l_f_phase):
            # update input spike history

            if self.M == 1:
                self.x_hist = np.concatenate(
                    [self.x_hist[:, 1:], np.random.poisson(np.concatenate([x_left, x_right], axis=0))], axis=-1)
            else:
                self.x_hist_left = np.concatenate(
                    [self.x_hist_left[:, 1:], np.random.poisson(x_left)], axis=-1)

                self.x_hist_right = np.concatenate(
                    [self.x_hist_right[:, 1:], np.random.poisson(x_right)], axis=-1)
            # do a forward pass
            self.out_f(training=training)

            if use_rand_plateau_times and training:
                # calculate plateau potentials for hidden layer neurons
                for m in xrange(self.M-2, -1, -1):
                    plateau_indices = np.nonzero(
                        time == self.plateau_times_f[m][training_num])

                    self.l[m].plateau_f(plateau_indices=plateau_indices)

            if record_voltages and training:
                # record voltages for this timestep
                for m in xrange(self.M):
                    if m != self.M-1:
                        self.A_hists[m][time, :] = self.l[m].A[:, 0]
                    self.B_hists[m][time, :] = self.l[m].B[:, 0]
                    self.C_hists[m][time, :] = self.l[m].C[:, 0]

        if (not use_rand_plateau_times) or (not training):
            for m in xrange(self.M-2, -1, -1):
                plateau_indices = np.arange(self.n[m])

                # calculate plateau potentials for hidden layer neurons
                self.l[m].plateau_f(plateau_indices=plateau_indices)

        for m in xrange(self.M-1, -1, -1):
            # calculate averages
            self.l[m].calc_averages(phase="forward")

        if record_eigvals:
            # calculate Jacobians & update lists of last 100 Jacobians
            if len(self.J_betas) >= 100:
                self.J_betas = self.J_betas[1:]
                self.J_gammas = self.J_gammas[1:]

            self.J_betas.append(np.multiply(
                lambda_max*deriv_sigma(self.l[-1].average_C_f), k_D*self.W[-1]))
            if self.M == 1:
                self.J_gammas.append(np.multiply(deriv_sigma(
                    np.dot(self.Y[-2], lambda_max*sigma(self.l[-1].average_C_f))), self.Y[-2]))
            else:
                # needs to concatenate the left and right here
                self.J_gammas.append(np.multiply(deriv_sigma(
                    np.dot(np.sum([self.Y_left[-2], self.Y_right[-2]], axis=0), lambda_max*sigma(self.l[-1].average_C_f))), np.sum([self.Y_left[-2], self.Y_right[-2]], axis=0)))

        if record_voltages and training:
            # append voltages to files
            for m in xrange(self.M):
                if m != self.M-1:
                    with open(os.path.join(self.simulation_path, 'A_hist_{}.csv'.format(m)), 'a') as A_hist_file:
                        np.savetxt(A_hist_file, self.A_hists[m])
                with open(os.path.join(self.simulation_path, 'B_hist_{}.csv'.format(m)), 'a') as B_hist_file:
                    np.savetxt(B_hist_file, self.B_hists[m])
                with open(os.path.join(self.simulation_path, 'C_hist_{}.csv'.format(m)), 'a') as C_hist_file:
                    np.savetxt(C_hist_file, self.C_hists[m])

    def t_phase(self, x_left, x_right,  training_num):
        '''
        Perform a target phase.

        Arguments:
            x (ndarray)        : Input array of size (X, 1) where X is the size of the input, eg. (784, 1).
            training_num (int) : Number (from start of the epoch) of the training example being shown.
        '''

        if record_voltages:
            # initialize voltage arrays
            self.A_hists = [np.zeros((l_t_phase, self.l[m].size))
                            for m in xrange(self.M-1)]
            self.B_hists = [np.zeros((l_t_phase, self.l[m].size))
                            for m in xrange(self.M)]
            self.C_hists = [np.zeros((l_t_phase, self.l[m].size))
                            for m in xrange(self.M)]

        for time in xrange(l_t_phase):
            # update input history

            if self.M == 1:
                self.x_hist = np.concatenate(
                    [self.x_hist[:, 1:], np.random.poisson(np.concatenate([x_left, x_right], axis=0))], axis=-1)
            else:
                self.x_hist_left = np.concatenate(
                    [self.x_hist_left[:, 1:], np.random.poisson(x_left)], axis=-1)
                self.x_hist_right = np.concatenate(
                    [self.x_hist_right[:, 1:], np.random.poisson(x_right)], axis=-1)

            # calculate backprop angle at the end of the target phase
            calc_E_bp = record_backprop_angle and time == l_t_phase - 1
            # calculate backprop angle at the end of the target phase
            # do a target pass
            self.out_t()
            if use_rand_plateau_times:
                # calculate plateau potentials & perform weight updates
                for m in xrange(self.M-2, -1, -1):
                    plateau_indices = np.nonzero(
                        time == self.plateau_times_t[m][training_num])

                    self.l[m].plateau_t(plateau_indices=plateau_indices)

            if record_voltages:
                # record voltages for this timestep
                for m in xrange(self.M):
                    if m != self.M-1:
                        self.A_hists[m][time, :] = self.l[m].A[:, 0]
                    self.B_hists[m][time, :] = self.l[m].B[:, 0]
                    self.C_hists[m][time, :] = self.l[m].C[:, 0]

        if not use_rand_plateau_times:
            for m in xrange(self.M-2, -1, -1):
                plateau_indices = np.arange(self.n[m])

                # calculate plateau potentials for hidden layer neurons
                self.l[m].plateau_t(plateau_indices=plateau_indices)

        for m in xrange(self.M-1, -1, -1):
            # calculate averages
            self.l[m].calc_averages(phase="target")

            if update_feedback_weights and m < self.M-1:
                # update feedback weights
                self.l[m].update_Y()

            # update weights
            self.l[m].update_W()

        if record_loss:
            # take note of this (the loss function equation 7) - calculated by taking the difference between the
            # average forward phase activity and the target phase on the output layer
            self.loss = ((self.l[-1].average_lambda_C_t -
                          lambda_max*sigma(self.l[-1].average_C_f)) ** 2).mean()

        for m in xrange(self.M-1, -1, -1):
            # reset averages

            if m == self.M-1:
                self.l[m].average_lambda_C_f *= 0
                self.l[m].average_lambda_C_t *= 0
                self.l[m].average_C_f *= 0
                self.l[m].average_C_t *= 0
                self.l[m].average_PSP_B_f *= 0
            else:
                self.l[m].average_C_f_left *= 0
                self.l[m].average_C_t_left *= 0
                self.l[m].average_PSP_B_f_left *= 0
                self.l[m].average_A_f_left *= 0
                self.l[m].average_A_t_left *= 0
                self.l[m].average_lambda_C_f_left *= 0
                self.l[m].average_lambda_C_t_left *= 0
                self.l[m].average_lambda_C_t_left *= 0
                # the right input reset
                self.l[m].average_C_f_right *= 0
                self.l[m].average_lambda_C_t_right *= 0
                self.l[m].average_lambda_C_t_right *= 0
                self.l[m].average_C_t_right *= 0
                self.l[m].average_PSP_B_f_right *= 0
                self.l[m].average_A_f_right *= 0
                self.l[m].average_A_t_right *= 0
                self.l[m].average_lambda_C_f_right *= 0

            if self.M == 1:
                self.x_hist *= 0

                if update_feedback_weights:
                    self.l[m].average_PSP_A_f_left *= 0
                    self.l[m].average_PSP_A_f_right *= 0

        if use_symmetric_weights:
            # make feedback weights symmetric to new feedforward weights
            self.make_weights_symmetric()

        if use_sparse_feedback and (use_symmetric_weights or update_feedback_weights):
            for m in xrange(self.M-1):
                # zero out the inactive weights
                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0
                # increase magnitude of surviving weights
                self.Y[m] *= 5
                # zero out the inactive weights
                self.Y_left[m].ravel()[self.Y_dropout_indices_left[m]] = 0
                self.Y_right[m].ravel()[self.Y_dropout_indices_right[m]] = 0
                # increase magnitude of surviving weights
                self.Y_left[m] *= 5
                self.Y_right[m] *= 5

        if record_voltages:
            # append voltages to files
            for m in xrange(self.M):
                if m != self.M-1:
                    with open(os.path.join(self.simulation_path, 'A_hist_{}.csv'.format(m)), 'a') as A_hist_file:
                        np.savetxt(A_hist_file, self.A_hists[m])
                with open(os.path.join(self.simulation_path, 'B_hist_{}.csv'.format(m)), 'a') as B_hist_file:
                    np.savetxt(B_hist_file, self.B_hists[m])
                with open(os.path.join(self.simulation_path, 'C_hist_{}.csv'.format(m)), 'a') as C_hist_file:
                    np.savetxt(C_hist_file, self.C_hists[m])

    def train(self, f_etas, b_etas, n_epochs, n_training_examples, save_simulation, simulations_folder=default_simulations_folder, folder_name="", overwrite=True, simulation_notes=None, current_epoch=None):
        '''
        Train the network. Checkpoints will be saved at the end of every epoch if save_simulation is True.

        Arguments:
            f_etas (tuple)              : Learning rates for each layer's feedforward weights, eg. (0.21, 0.21).
            b_etas (tuple/None)         : Learning rates for each layer's feedback weights.
                                          If None, no backward weight updates occur.
            n_epochs (int)              : Number of epochs of training.
            n_training_examples (int)   : Number of training examples per epoch.
            save_simulation (bool)      : Whether to save data from this simulation.
            simulations_folder (string) : Name of the parent folder that can contain data from multiple simulations.
            folder_name (string)        : Name of the subfolder in the parent folder that will contain data from this simulation.
            overwrite (bool)            : Whether to overwrite the folder given by folder_name if it already exists.
            simulation_notes (string)   : Notes about this simulation to save in the parameters text file that will be generated.
            current_epoch (int/None)    : The current epoch of this simulation. This sets the value of the network's current_epoch attribute.
                                          If 0, this is a new simulation.
                                          If > 0, this is a continuation of a previously-started simulation.
                                          If None, the current value of the network's 'current_epoch' attribute
                                          determines the state of the simulation.
        '''

        print("Starting training.\n")

        if b_etas == None and update_feedback_weights:
            raise ValueError(
                "No feedback learning rates provided, but 'update_feedback_weights' is True.")

        if current_epoch != None:
            self.current_epoch == current_epoch
        elif self.current_epoch == None:
            # set current epoch
            self.current_epoch = 0

        if self.current_epoch == 0:
            continuing = False
        else:
            continuing = True

        if use_rand_phase_lengths:
            # generate phase lengths for all training examples
            global l_f_phase, l_t_phase
            l_f_phases = min_l_f_phase + \
                np.random.wald(2, 1, n_epochs*n_training_examples).astype(int)
            l_t_phases = min_l_t_phase + \
                np.random.wald(2, 1, n_epochs*n_training_examples).astype(int)
        else:
            l_f_phases = np.zeros(n_epochs*n_training_examples) + l_f_phase
            l_t_phases = np.zeros(n_epochs*n_training_examples) + l_t_phase

        # get array of total length of both phases for all training examples
        l_phases_tot = l_f_phases + l_t_phases

        # get current date/time and create simulation directory
        if save_simulation:
            sim_start_time = datetime.datetime.now()

            if folder_name == "":
                self.simulation_path = os.path.join(simulations_folder, "{}.{}.{}-{}.{}".format(sim_start_time.year,
                                                                                                sim_start_time.month,
                                                                                                sim_start_time.day,
                                                                                                sim_start_time.hour,
                                                                                                sim_start_time.minute))
            else:
                self.simulation_path = os.path.join(
                    simulations_folder, folder_name)

            # make simulation directory
            if not os.path.exists(self.simulation_path):
                os.makedirs(self.simulation_path)
            elif not continuing:
                if overwrite == False:
                    print("Error: Simulation directory \"{}\" already exists.".format(
                        self.simulation_path))
                    return
                else:
                    shutil.rmtree(self.simulation_path, ignore_errors=True)
                    os.makedirs(self.simulation_path)

            # copy current script to simulation directory
            filename = os.path.basename(__file__)
            if filename.endswith('pyc'):
                filename = filename[:-1]
            shutil.copyfile(filename, os.path.join(
                self.simulation_path, filename))

            params = {
                'nonspiking_mode': nonspiking_mode,
                'n_full_test': n_full_test,
                'n_quick_test': n_quick_test,
                'use_rand_phase_lengths': use_rand_phase_lengths,
                'use_rand_plateau_times': use_rand_plateau_times,
                'use_conductances': use_conductances,
                'use_broadcast': use_broadcast,
                'use_spiking_feedback': use_spiking_feedback,
                'use_spiking_feedforward': use_spiking_feedforward,
                'use_symmetric_weights': use_symmetric_weights,
                'noisy_symmetric_weights': noisy_symmetric_weights,
                'use_sparse_feedback': use_sparse_feedback,
                'update_feedback_weights': update_feedback_weights,
                'use_backprop': use_backprop,
                'use_apical_conductance': use_apical_conductance,
                'use_weight_optimization': use_weight_optimization,
                'use_feedback_bias': use_feedback_bias,
                'initial_test': initial_test,
                'record_backprop_angle': record_backprop_angle,
                'record_loss': record_loss,
                'record_voltages': record_voltages,
                'record_training_error': record_training_error,
                'record_training_labels': record_training_labels,
                'record_phase_times': record_phase_times,
                'record_plateau_times': record_plateau_times,
                'record_eigvals': record_eigvals,
                'record_matrices': record_matrices,
                'plot_eigvals': plot_eigvals,
                'dt': dt,
                'mem': mem,
                'integration_time': integration_time,
                'integration_time_test': integration_time_test,
                'l_f_phase': l_f_phase,
                'l_t_phase': l_t_phase,
                'l_f_phase_test': l_f_phase_test,
                'lambda_max': lambda_max,
                'tau_s': tau_s,
                'tau_L': tau_L,
                'g_B': g_B,
                'g_A': g_A,
                'g_L': g_L,
                'g_D': g_D,
                'k_B': k_B,
                'k_D': k_D,
                'k_I': k_I,
                'P_hidden': P_hidden,
                'P_final': P_final,
                'n': self.n,
                'f_etas': f_etas,
                'b_etas': b_etas,
                'n_training_examples': n_training_examples,
                'n_epochs': n_epochs
            }

            # save simulation params
            if not continuing:
                with open(os.path.join(self.simulation_path, 'simulation.txt'), 'w') as simulation_file:
                    print("Simulation done on {}.{}.{}-{}.{}.".format(sim_start_time.year,
                                                                      sim_start_time.month,
                                                                      sim_start_time.day,
                                                                      sim_start_time.hour,
                                                                      sim_start_time.minute), file=simulation_file)
                    if simulation_notes:
                        print(simulation_notes, file=simulation_file)
                    print("Start time: {}".format(
                        sim_start_time), file=simulation_file)
                    print("-----------------------------", file=simulation_file)
                    for key, value in sorted(params.items()):
                        line = '{}: {}'.format(key, value)
                        print(line, file=simulation_file)

                with open(os.path.join(self.simulation_path, 'simulation.json'), 'w') as simulation_file:
                    simulation_file.write(json.dumps(params))
            else:
                # load previously saved recording arrays
                self.prev_full_test_errs = np.load(os.path.join(
                    self.simulation_path, "full_test_errors.npy"))
                self.prev_quick_test_errs = np.load(os.path.join(
                    self.simulation_path, "quick_test_errors.npy"))

                if record_backprop_angle and not use_backprop:
                    self.prev_bp_angles = np.load(os.path.join(
                        self.simulation_path, "bp_angles.npy"))

                if record_loss:
                    self.prev_losses = np.load(os.path.join(
                        self.simulation_path, "final_layer_loss.npy"))

                if record_training_error:
                    self.prev_training_errors = np.load(os.path.join(
                        self.simulation_path, "training_errors.npy"))

                if record_training_labels:
                    self.prev_training_labels = np.load(os.path.join(
                        self.simulation_path, "training_labels.npy"))

                if record_plateau_times:
                    self.prev_plateau_times_full = [np.load(os.path.join(
                        self.simulation_path, "plateau_times_{}.npy".format(m))) for m in range(self.M)]

                if record_phase_times:
                    self.prev_phase_times = np.load(os.path.join(
                        self.simulation_path, "phase_times.npy"))

                if record_eigvals:
                    self.prev_max_jacobian_eigvals = np.load(os.path.join(
                        self.simulation_path, "max_jacobian_eigvals.npy"))
                    self.prev_max_weight_eigvals = np.load(os.path.join(
                        self.simulation_path, "max_weight_eigvals.npy"))
                    if record_matrices:
                        self.prev_jacobian_prod_matrices = np.load(os.path.join(
                            self.simulation_path, "jacobian_prod_matrices.npy"))
                        self.prev_weight_prod_matrices = np.load(os.path.join(
                            self.simulation_path, "weight_prod_matrices.npy"))

        # set learning rate instance variables
        self.f_etas = f_etas
        self.b_etas = b_etas

        if save_simulation and not continuing:
            # save initial weights
            self.save_weights(self.simulation_path, prefix='initial_')

        if not continuing:
            # initialize full test error recording array
            self.full_test_errs = np.zeros(n_epochs + 1)

            # initialize quick test error recording array
            self.quick_test_errs = np.zeros(
                n_epochs*int(n_training_examples/1000.0) + 1)
        else:
            self.full_test_errs = np.zeros(n_epochs)
            self.quick_test_errs = np.zeros(
                n_epochs*int(n_training_examples/1000.0))

        if record_loss:
            self.losses = np.zeros(n_epochs*n_training_examples)

        if record_training_error:
            self.training_errors = np.zeros(n_epochs)

        if record_plateau_times:
            self.plateau_times_full = [
                np.zeros((n_epochs*2*n_training_examples, self.n[m])) for m in range(self.M)]

        if record_phase_times:
            self.phase_times = np.zeros(n_epochs*n_training_examples*2)

            self.phase_times[0] = l_f_phases[0]
            for i in xrange(1, 2*n_epochs*n_training_examples):
                if i % 2 == 0:
                    self.phase_times[i] = self.phase_times[i -
                                                           1] + l_f_phases[int(i/2)]
                else:
                    self.phase_times[i] = self.phase_times[i -
                                                           1] + l_t_phases[int((i-1)/2)]

            if save_simulation:
                if not continuing:
                    phase_times = self.phase_times
                else:
                    phase_times = np.concatenate(
                        self.prev_phase_times, self.phase_times, axis=0)

                np.save(os.path.join(self.simulation_path,
                        "phase_times.npy"), phase_times)

        if record_training_labels:
            self.training_labels = np.zeros(n_epochs*n_training_examples)

        if record_eigvals:
            # initialize arrays for Jacobian testing
            self.max_jacobian_eigvals = np.zeros(n_epochs*n_training_examples)
            if record_matrices:
                self.jacobian_prod_matrices = np.zeros(
                    (n_epochs*n_training_examples, self.n[-1], self.n[-1]))

            if not continuing:
                self.max_weight_eigvals = np.zeros(
                    n_epochs*n_training_examples + 1)
                if record_matrices:
                    self.weight_prod_matrices = np.zeros(
                        (n_epochs*n_training_examples + 1, self.n[-1], self.n[-1]))
            else:
                self.max_weight_eigvals = np.zeros(
                    n_epochs*n_training_examples)
                if record_matrices:
                    self.weight_prod_matrices = np.zeros(
                        (n_epochs*n_training_examples, self.n[-1], self.n[-1]))

            # create identity matrix
            I = np.eye(self.n[-1])

            # get max eigenvalues for weights
            U = np.dot(
                self.W[-1], np.concatenate([self.Y_left[-2], self.Y_right[-2]], axis=0))

            if self.M == 1:
                U = np.dot(
                    k_D*self.W[-1], self.Y[-2])

            p = np.dot((I - U).T, I - U)
            if not continuing:
                if record_matrices:
                    self.weight_prod_matrices[0] = U
                self.max_weight_eigvals[0] = np.amax(
                    np.real(np.linalg.eigvals(p)))

            # initialize lists for storing last 100 Jacobians
            self.J_betas = []
            self.J_gammas = []

        if record_backprop_angle and not use_backprop:
            # initialize backprop angles recording array
            if self.M > 1:
                self.bp_angles = np.zeros(n_epochs*n_training_examples)

        if initial_test and not continuing:
            # do an initial weight test
            print("Start of epoch {}.".format(self.current_epoch + 1))

            # set start time
            start_time = time.time()

            test_err = self.test_weights(n_test=n_full_test)

            # get end time & elapsed time
            end_time = time.time()
            time_elapsed = end_time - start_time

            sys.stdout.write("\x1b[2K\rFE: {0:05.2f}%. T: {1:.3f}s.\n\n".format(
                test_err, time_elapsed))

            self.full_test_errs[0] = test_err

            if save_simulation:
                # save full test error
                np.save(os.path.join(self.simulation_path,
                        "full_test_errors.npy"), self.full_test_errs)

                with open(os.path.join(self.simulation_path, "full_test_errors.txt"), 'a') as test_err_file:
                    line = "%.10f" % test_err
                    print(line, file=test_err_file)

            self.quick_test_errs[0] = test_err

            if save_simulation:
                # save quick test error
                np.save(os.path.join(self.simulation_path,
                        "quick_test_errors.npy"), self.quick_test_errs)

                with open(os.path.join(self.simulation_path, "quick_test_errors.txt"), 'a') as test_err_file:
                    line = "%.10f" % test_err
                    print(line, file=test_err_file)
        else:
            # don't do an initial weight test
            print("Start of epoch {}.\n".format(self.current_epoch + 1))

        # initialize input spike history
        self.x_hist_left = np.zeros((self.n_in, mem))
        self.x_hist_right = np.zeros((self.n_in, mem))
        if self.M == 1:
            self.x_hist = np.zeros((self.n_in * 2, mem))

        # start time used for timing how long each 1000 examples take
        start_time = None

        if record_eigvals and plot_eigvals:
            plt.close("all")
            fig = plt.figure(figsize=(13, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(321)
            ax3 = fig.add_subplot(312)
            plt.show(block=False)

        if record_training_error:
            num_correct = 0

        for k in xrange(n_epochs):
            # generate arrays of forward phase plateau potential times (time until plateau potential from start of forward phase) for individual neurons
            if use_rand_plateau_times:
                self.plateau_times_f = [np.zeros((n_training_examples, self.n[m])) + l_f_phases[k*n_training_examples:(k+1)*n_training_examples, np.newaxis] - 1 - np.minimum(
                    np.abs(np.random.normal(0, 3, size=(n_training_examples, self.n[m])).astype(int)), 5) for m in range(self.M)]
            else:
                self.plateau_times_f = [np.zeros((n_training_examples, self.n[m])) + l_f_phases[k*n_training_examples:(
                    k+1)*n_training_examples, np.newaxis] - 1 for m in range(self.M)]

            # generate arrays of target phase plateau potential times (time until plateau potential from start of target phase) for individual neurons
            if use_rand_plateau_times:
                self.plateau_times_t = [np.zeros((n_training_examples, self.n[m])) + l_t_phases[k*n_training_examples:(k+1)*n_training_examples, np.newaxis] - 1 - np.minimum(
                    np.abs(np.random.normal(0, 3, size=(n_training_examples, self.n[m])).astype(int)), 5) for m in range(self.M)]
            else:
                self.plateau_times_t = [np.zeros((n_training_examples, self.n[m])) + l_t_phases[k*n_training_examples:(
                    k+1)*n_training_examples, np.newaxis] - 1 for m in range(self.M)]

            for n in xrange(n_training_examples):
                # set start time
                if start_time == None:
                    start_time = time.time()

                if use_rand_phase_lengths:
                    l_f_phase = int(l_f_phases[k*n_training_examples + n])
                    l_t_phase = int(l_t_phases[k*n_training_examples + n])

                l_phases_tot = l_f_phase + l_t_phase

                # get plateau potential times from the beginning of the simulation
                if record_plateau_times:
                    total_time_to_forward_phase = np.sum(
                        l_f_phases[:k*n_training_examples + n]) + np.sum(l_t_phases[:k*n_training_examples + n])
                    total_time_to_target_phase = np.sum(
                        l_f_phases[:k*n_training_examples + n + 1]) + np.sum(l_t_phases[:k*n_training_examples + n])
                    for m in range(self.M):
                        self.plateau_times_full[m][k*n_training_examples + 2 *
                                                   n] = total_time_to_forward_phase + self.plateau_times_f[m][n]
                        self.plateau_times_full[m][k*n_training_examples + 2*n +
                                                   1] = total_time_to_target_phase + self.plateau_times_t[m][n]

                # print every 100 examples
                if (n+1) % 100 == 0:
                    sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}.".format(
                        self.current_epoch + 1, n+1, n_training_examples))
                    sys.stdout.flush()

                # get training example data
                self.x_left = lambda_max * \
                    self.x_train_input_left[:, n][:, np.newaxis]
                self.x_right = lambda_max * \
                    self.x_train_input_right[:, n][:, np.newaxis]

                self.t = self.t_train[:, n][:, np.newaxis]

                if record_voltages:
                    # initialize voltage arrays
                    self.A_hists = [np.zeros((l_f_phase, self.l[m].size))
                                    for m in xrange(self.M-1)]
                    self.B_hists = [
                        np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M)]
                    self.C_hists = [
                        np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M)]

                # do forward & target phases
                self.f_phase(self.x_left, self.x_right, n, training=True)

                if record_training_error:
                    sel_num = np.argmax(np.mean(
                        self.l[-1].average_C_f.reshape(-1, self.n_neurons_per_category), axis=-1))

                    # get the target number from testing example data
                    target_num = np.dot(np.arange(10), self.t)

                    # increment correct classification counter if they match
                    if sel_num == target_num:
                        num_correct += 1

                self.t_phase(self.x_left, self.x_right, n)
                if record_loss:
                    self.losses[k*n_training_examples + n] = self.loss

                if record_training_labels:
                    self.training_labels[k*n_training_examples +
                                         n] = np.dot(np.arange(10), self.t)

                if record_eigvals:
                    # get max eigenvalues for jacobians
                    # U = np.dot(np.mean(np.array(self.J_betas), axis=0), np.mean(np.array(self.J_gammas), axis=0)) # product of mean of last 100 Jacobians
                    U = np.mean(np.array([np.dot(self.J_betas[i], self.J_gammas[i]) for i in range(
                        len(self.J_betas))]), axis=0)  # mean of product of last 100 Jacobians

                    p = np.dot((I - U).T, I - U)
                    if record_matrices:
                        self.jacobian_prod_matrices[k *
                                                    n_training_examples + n] = U
                    self.max_jacobian_eigvals[k*n_training_examples +
                                              n] = np.amax(np.linalg.eigvals(p))

                    # get max eigenvalues for weights
                    U = np.dot(
                        k_D*self.W[-1], np.concatenate([self.Y_left[-2], self.Y_right[-2]], axis=0))
                    if self.M == 1:
                        U = np.dot(
                            k_D*self.W[-1], self.Y[-2])

                    p = np.dot((I - U).T, I - U)
                    if not continuing:
                        if record_matrices:
                            self.weight_prod_matrices[k *
                                                      n_training_examples + n + 1] = U
                        self.max_weight_eigvals[k*n_training_examples +
                                                n + 1] = np.amax(np.linalg.eigvals(p))
                    else:
                        if record_matrices:
                            self.weight_prod_matrices[k *
                                                      n_training_examples + n] = U
                        self.max_weight_eigvals[k*n_training_examples +
                                                n] = np.amax(np.linalg.eigvals(p))

                    if plot_eigvals and k == 0 and n == 0:
                        # draw initial plots
                        if record_matrices:
                            A = self.jacobian_prod_matrices[0]
                            im_plot = ax1.imshow(
                                A, interpolation='nearest', vmin=0, vmax=1)
                            fig.colorbar(im_plot, ax=ax1)
                        if record_loss:
                            loss_plot, = ax2.plot(np.arange(1), self.losses[0])
                        max_jacobian_plot, = ax3.plot(
                            np.arange(1), self.max_jacobian_eigvals[0], '.')
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                if record_backprop_angle and not use_backprop:
                    # get backprop angle
                    if self.M > 1:
                        bp_angle = np.arccos(np.sum(self.l[0].delta_b_bp * self.l[0].delta_b_full) / (
                            np.linalg.norm(self.l[0].delta_b_bp)*np.linalg.norm(self.l[0].delta_b_full.T)))*180.0/np.pi
                        self.bp_angles[k*n_training_examples + n] = bp_angle

                if plot_eigvals and record_eigvals and (n+1) % 100 == 0:
                    max_inds = np.argsort(
                        self.max_jacobian_eigvals[k*n_training_examples + n - 99:k*n_training_examples + n + 1])
                    max_ind = np.argmax(
                        self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1])
                    min_ind = np.argmin(
                        self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1])
                    n_small = np.sum(
                        self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1] < 1)

                    # update plots
                    if record_matrices:
                        A = np.mean(np.array([self.jacobian_prod_matrices[k*n_training_examples +
                                                                          n-99:k*n_training_examples + n + 1][i] for i in max_inds][:-10]), axis=0)
                        im_plot.set_data(A)

                    if record_loss:
                        loss_plot.set_xdata(
                            np.arange(k*n_training_examples + n))
                        loss_plot.set_ydata(
                            self.losses[:k*n_training_examples + n])
                        ax2.set_xlim(0, k*n_training_examples + n)
                        ax2.set_ylim(np.amin(self.losses[:k*n_training_examples + n]) - 1e-6, np.amax(
                            self.losses[:k*n_training_examples + n]) + 1e-6)

                    max_jacobian_plot.set_xdata(
                        np.arange(k*n_training_examples + n))
                    max_jacobian_plot.set_ydata(
                        self.max_jacobian_eigvals[:k*n_training_examples + n])
                    ax3.set_xlim(0, k*n_training_examples + n)
                    ax3.set_ylim(np.amin(self.max_jacobian_eigvals[:k*n_training_examples + n]) - 1e-6, np.amax(
                        self.max_jacobian_eigvals[:k*n_training_examples + n]) + 1e-6)

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if (n+1) % 1000 == 0:
                    if n != n_training_examples - 1:
                        # we're partway through an epoch; do a quick weight test
                        test_err = self.test_weights(n_test=n_quick_test)

                        sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}. Q E: {3:05.2f}%. ".format(
                            self.current_epoch + 1, n+1, n_training_examples, test_err))

                        if not continuing:
                            self.quick_test_errs[(
                                k+1)*int(n_training_examples/1000)] = test_err
                        else:
                            self.quick_test_errs[(
                                k+1)*int(n_training_examples/1000) - 1] = test_err

                        if save_simulation:
                            with open(os.path.join(self.simulation_path, "quick_test_errors.txt"), 'a') as test_err_file:
                                line = "%.10f" % test_err
                                print(line, file=test_err_file)
                    else:
                        # we've reached the end of an epoch; do a full weight test
                        test_err = self.test_weights(n_test=n_full_test)

                        sys.stdout.write(
                            "\x1b[2K\rFE: {0:05.2f}%. ".format(test_err))

                        if not continuing:
                            self.full_test_errs[k+1] = test_err
                            self.quick_test_errs[(
                                k+1)*int(n_training_examples/1000)] = test_err
                        else:
                            self.full_test_errs[k] = test_err
                            self.quick_test_errs[(
                                k+1)*int(n_training_examples/1000) - 1] = test_err

                        if save_simulation:
                            with open(os.path.join(self.simulation_path, "full_test_errors.txt"), 'a') as test_err_file:
                                line = "%.10f" % test_err
                                print(line, file=test_err_file)

                        if record_training_error:
                            # calculate percent training error for this epoch
                            err_rate = (1.0 - float(num_correct) /
                                        n_training_examples)*100.0
                            self.training_errors[k] = err_rate

                            print("TE: {0:05.2f}%. ".format(err_rate), end="")

                            num_correct = 0

                        # save recording arrays
                        if save_simulation:
                            print("Saving...", end="")
                            if not continuing:
                                # we are running a new simulation
                                quick_test_errs = self.quick_test_errs[:(
                                    k+1)*int(n_training_examples/1000)+1]
                                full_test_errs = self.full_test_errs[:k+2]

                                if record_backprop_angle and not use_backprop:
                                    bp_angles = self.bp_angles[:(
                                        k+1)*n_training_examples]

                                if record_loss:
                                    losses = self.losses[:(
                                        k+1)*n_training_examples]

                                if record_training_labels:
                                    training_labels = self.training_labels[:(
                                        k+1)*n_training_examples]

                                if record_plateau_times:
                                    plateau_times_full = [self.plateau_times_full[m][:(
                                        k+1)*2*n_training_examples] for m in range(self.M)]

                                if record_training_error:
                                    training_errors = self.training_errors[:k+1]

                                if record_eigvals:
                                    max_jacobian_eigvals = self.max_jacobian_eigvals[:(
                                        k+1)*n_training_examples]
                                    max_weight_eigvals = self.max_weight_eigvals[:(
                                        k+1)*n_training_examples+1]
                                    if record_matrices:
                                        jacobian_prod_matrices = self.jacobian_prod_matrices[:(
                                            k+1)*n_training_examples]
                                        weight_prod_matrices = self.weight_prod_matrices[:(
                                            k+1)*n_training_examples+1]
                            else:
                                # this is a continuation of a previously-started simulation; append current recording vectors to previous ones
                                quick_test_errs = np.concatenate([self.prev_quick_test_errs, self.quick_test_errs[:(
                                    k+1)*int(n_training_examples/1000)]], axis=0)
                                if n == n_training_examples - 1:
                                    full_test_errs = np.concatenate(
                                        [self.prev_full_test_errs, self.full_test_errs[:k+1]], axis=0)

                                if record_backprop_angle and not use_backprop:
                                    bp_angles = np.concatenate(
                                        [self.prev_bp_angles, self.bp_angles[:(k+1)*n_training_examples]], axis=0)

                                if record_loss:
                                    losses = np.concatenate(
                                        [self.prev_losses, self.losses[:(k+1)*n_training_examples]], axis=0)

                                if record_training_labels:
                                    training_labels = np.concatenate(
                                        [self.prev_training_labels, self.training_labels[:(k+1)*n_training_examples]], axis=0)

                                if record_plateau_times:
                                    plateau_times_full = [np.concatenate([self.prev_plateau_times_full[m], self.plateau_times_full[m][:(
                                        k+1)*2*n_training_examples]]) for m in range(self.M)]

                                if record_training_error:
                                    training_errors = np.concatenate(
                                        [self.prev_training_errors, self.training_errors[:k+1]], axis=0)

                                if record_eigvals:
                                    max_jacobian_eigvals = np.concatenate(
                                        [self.prev_max_jacobian_eigvals, self.max_jacobian_eigvals[:(k+1)*n_training_examples]], axis=0)
                                    max_weight_eigvals = np.concatenate(
                                        [self.prev_max_weight_eigvals, self.max_weight_eigvals[:(k+1)*n_training_examples]], axis=0)
                                    if record_matrices:
                                        jacobian_prod_matrices = np.concatenate(
                                            [self.prev_jacobian_prod_matrices, self.jacobian_prod_matrices[:(k+1)*n_training_examples]], axis=0)
                                        weight_prod_matrices = np.concatenate(
                                            [self.prev_weight_prod_matrices, self.weight_prod_matrices[:(k+1)*n_training_examples]], axis=0)

                            # save quick test error
                            np.save(os.path.join(self.simulation_path,
                                    "quick_test_errors.npy"), quick_test_errs)

                            if n == n_training_examples - 1:
                                # save test error
                                np.save(os.path.join(self.simulation_path,
                                        "full_test_errors.npy"), full_test_errs)

                                # save weights
                                self.save_weights(
                                    self.simulation_path, prefix="epoch_{}_".format(self.current_epoch + 1))

                            if record_backprop_angle and not use_backprop:
                                if self.M > 1:
                                    # save backprop angles
                                    np.save(os.path.join(
                                        self.simulation_path, "bp_angles.npy"), bp_angles)

                            if record_loss:
                                np.save(os.path.join(self.simulation_path,
                                        "final_layer_loss.npy"), losses)

                            if record_training_labels:
                                np.save(os.path.join(self.simulation_path,
                                        "training_labels.npy"), training_labels)

                            if record_plateau_times:
                                for m in range(self.M):
                                    np.save(os.path.join(self.simulation_path, "plateau_times_{}.npy".format(
                                        m)), self.plateau_times_full[m])

                            if record_training_error:
                                np.save(os.path.join(self.simulation_path,
                                        "training_errors.npy"), training_errors)

                            if record_eigvals:
                                # save eigenvalues
                                np.save(os.path.join(
                                    self.simulation_path, "max_jacobian_eigvals.npy"), max_jacobian_eigvals)
                                np.save(os.path.join(
                                    self.simulation_path, "max_weight_eigvals.npy"), max_weight_eigvals)
                                if record_matrices:
                                    np.save(os.path.join(
                                        self.simulation_path, "jacobian_prod_matrices.npy"), jacobian_prod_matrices)
                                    np.save(os.path.join(
                                        self.simulation_path, "weight_prod_matrices.npy"), weight_prod_matrices)

                            print("done. ", end="")

                    if record_eigvals:
                        print("Min max Jacobian eigval: {:.4f}. ".format(np.amin(self.max_jacobian_eigvals[max(
                            0, k*n_training_examples + n - 999):k*n_training_examples + n + 1])), end="")

                        print("# max eigvals < 1: {}. ".format(np.sum(self.max_jacobian_eigvals[max(
                            0, k*n_training_examples + n - 999):k*n_training_examples + n + 1] < 1)), end="")

                    # get end time & reset start time
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    print("T: {0:.3f}s.\n".format(time_elapsed))
                    start_time = None

            # update latest epoch counter
            self.current_epoch += 1

        # record end time of training
        if save_simulation:
            with open(os.path.join(self.simulation_path, 'simulation.txt'), 'a') as simulation_file:
                sim_end_time = datetime.datetime.now()
                print("-----------------------------", file=simulation_file)
                print("End time: {}".format(sim_end_time), file=simulation_file)

    def test_weights(self, n_test=n_quick_test):
        '''
        Test the network's current weights on the test set. The network's layers are copied
        and restored to their previous state after testing.

        Arguments:
            n_test (int) : The number of test examples to use.
        '''

        global l_f_phase, integration_time
        global prediction
        global test_sample
        # save old length of forward phase
        old_l_f_phase = l_f_phase

        # set new length of forward phase
        l_f_phase = l_f_phase_test
        prediction = []
        test_sample = []
        # save old integration time
        # old_integration_time = self.integration_time
        old_integration_time = integration_time
        # integration_time = integration_time_test
        # set new integration time
        integration_time = integration_time_test
        # self.integration_time = integration_time_test

        old_x_hist_left = self.x_hist_left
        old_x_hist_right = self.x_hist_right
        if self.M == 1:
            old_x_hist = self.x_hist

        # initialize count of correct classifications
        num_correct = 0

        # shuffle testing data
        # self.x_test, self.t_test = shuffle_arrays(self.x_test, self.t_test)

        self.x_test_left = self.x_test[:self.x_test.shape[0]//2, :]
        self.x_test_right = self.x_test[self.x_test.shape[0]//2:, :]

        digits = np.arange(10)

        # create new integration recording variables
        for m in xrange(self.M):
            self.l[m].create_integration_vars()

        for n in xrange(n_test):
            # clear all layer variables
            for m in xrange(self.M):
                self.l[m].clear_vars()
            # clear input spike history
            self.x_hist_left *= 0
            self.x_hist_right *= 0
            if self.M == 1:
                self.x_hist *= 0

            # get testing example data
            self.x_left = lambda_max*self.x_test_left[:, n][:, np.newaxis]
            self.x_right = lambda_max*self.x_test_left[:, n][:, np.newaxis]
            self.t = self.t_test[:, n][:, np.newaxis]

            # do a forward phase & get the unit with maximum average somatic potential
            self.f_phase(self.x_left, self.x_right, None, training=False)
            sel_num = np.argmax(np.mean(
                self.l[-1].average_C_f.reshape(-1, self.n_neurons_per_category), axis=-1))

            # get the target number from testing example data
            target_num = np.dot(digits, self.t)
            prediction.append(target_num[0])
            test_sample.append(sel_num)

            # increment correct classification counter if they match
            if sel_num == target_num:
                num_correct += 1

            # print every 100 testing examples
            if (n + 1) % 100 == 0:
                sys.stdout.write("\x1b[2K\rTesting example {0}/{1}. E: {2:05.2f}%.".format(
                    n+1, n_test, (1.0 - float(num_correct)/(n+1))*100.0))
                sys.stdout.flush()

        # calculate percent error
        err_rate = (1.0 - float(num_correct)/n_test)*100.0

        if old_x_hist_left is not None:
            self.x_hist_left = old_x_hist_left
            self.x_hist_right = old_x_hist_right

        if self.M == 1 and old_x_hist is not None:
            self.x_hist = old_x_hist

        # self.integration_time = old_integration_time
        integration_time = old_integration_time

        l_f_phase = old_l_f_phase

        # create new integration recording variables
        for m in xrange(self.M):
            self.l[m].create_integration_vars()

        # clear all layer variables
        for m in xrange(self.M):
            self.l[m].clear_vars()

        if n_test > 100:
            sys.stdout.write("\x1b[2K\r")
            sys.stdout.flush()

        return err_rate

    def save_weights(self, path, prefix=""):
        '''
        Save the network's current weights to .npy files.

        Arguments:
            path (string)   : The path of the folder in which to save the network's weights.
            prefix (string) : A prefix to append to the filenames of the saved weights.
        '''

        for m in xrange(self.M):
            if self.M == 1:
                np.save(os.path.join(path, prefix +
                                     "W_{}.npy".format(m)), self.W[m])
                np.save(os.path.join(path, prefix +
                        "b_{}.npy".format(m)), self.b[m])
                if m != self.M - 1:
                    np.save(os.path.join(path, prefix +
                            "Y_{}.npy".format(m)), self.Y[m])
                    if use_feedback_bias:
                        np.save(os.path.join(path, prefix +
                                "c_{}.npy".format(m)), self.c[m])
            else:
                np.save(os.path.join(path, prefix +
                        "W_{}.npy".format(m)), self.W[m])
                np.save(os.path.join(path, prefix +
                        "b_{}.npy".format(m)), self.b[m])
                np.save(os.path.join(path, prefix +
                                     "W_left_{}.npy".format(m)), self.W_left[m])
                np.save(os.path.join(path, prefix +
                        "b_left_{}.npy".format(m)), self.b_left[m])
                np.save(os.path.join(path, prefix +
                        "W_right_{}.npy".format(m)), self.W_right[m])
                np.save(os.path.join(path, prefix +
                        "b_right_{}.npy".format(m)), self.b_right[m])
                if m != self.M - 1:
                    np.save(os.path.join(path, prefix +
                            "Y_{}.npy".format(m)), self.Y[m])
                    np.save(os.path.join(path, prefix +
                            "Y_left_{}.npy".format(m)), self.Y_left[m])
                    np.save(os.path.join(path, prefix +
                            "Y_right_{}.npy".format(m)), self.Y_right[m])
                    if use_feedback_bias:
                        np.save(os.path.join(path, prefix +
                                "c_{}.npy".format(m)), self.c[m])
                        np.save(os.path.join(path, prefix +
                                "c_right_{}.npy".format(m)), self.c_left[m])
                        np.save(os.path.join(path, prefix +
                                "c_left_{}.npy".format(m)), self.c_right[m])

    def load_weights(self, path, prefix=""):
        '''
        Load weights from .npy files and set them to the network's weights.

        Arguments:
            path (string)   : The path of the folder from which to load the weights.
            prefix (string) : Prefix appended to the filenames of the saved weights.
        '''

        print("Loading weights from \"{}\" with prefix \"{}\".".format(path, prefix))
        print("--------------------------------")

        for m in xrange(self.M):
            self.W[m] = np.load(os.path.join(
                path, prefix + "W_{}.npy".format(m)))
            self.b[m] = np.load(os.path.join(
                path, prefix + "b_{}.npy".format(m)))
            self.W_left[m] = np.load(os.path.join(
                path, prefix + "W_left_{}.npy".format(m)))
            self.b_left[m] = np.load(os.path.join(
                path, prefix + "b_left_{}.npy".format(m)))
            self.W_right[m] = np.load(os.path.join(
                path, prefix + "W_right_{}.npy".format(m)))
            self.b_right[m] = np.load(os.path.join(
                path, prefix + "b_right_{}.npy".format(m)))
            if m != self.M - 1:
                self.Y[m] = np.load(os.path.join(
                    path, prefix + "Y_{}.npy".format(m)))
                self.Y_left[m] = np.load(os.path.join(
                    path, prefix + "Y_left_{}.npy".format(m)))
                self.Y_right[m] = np.load(os.path.join(
                    path, prefix + "Y_right_{}.npy".format(m)))
                if use_feedback_bias:
                    self.c[m] = np.load(os.path.join(
                        path, prefix + "c_{}.npy".format(m)))
                    self.c_left[m] = np.load(os.path.join(
                        path, prefix + "c_left_{}.npy".format(m)))
                    self.c_right[m] = np.load(os.path.join(
                        path, prefix + "c_right_{}.npy".format(m)))

        # print network weights
        self.print_weights()

        print("--------------------------------")
    # ---------------------------------------------------------------
    """                 Network Helper function                   """
    # ---------------------------------------------------------------

    # --- sigmoid function --- #

    def sigma(self, x):
        return expit(x)

    def deriv_sigma(self, x):
        return expit(x)*(1.0 - expit(x))

    # --- kernel function --- #

    def kappa(self, x):
        return (np.exp(-x/tau_L) - np.exp(-x/tau_s))/(tau_L - tau_s)

    def get_kappas(self, n=mem):
        return np.array([kappa(i+1) for i in xrange(n)])


# ---------------------------------------------------------------
"""                     Helper functions                      """
# ---------------------------------------------------------------


def add_noise_to_dataset(dataset):
    # define the mean and std
    # mu, sigma = 1, 0.001

    # noise = np.random.normal(mu, sigma, [dataset.shape[0], dataset.shape[1]])

    # print(" noise ", (dataset + noise)[0], dataset[0])
    from scipy import stats

    low = 0
    high = 0.8
    mean = 0.003
    stddev = 0.002

    noise = stats.truncnorm.rvs(low, high,
                                loc=mean, scale=stddev,
                                size=(dataset.shape[0], dataset.shape[1]))
# + (noise/5)
    return dataset


def load_simulation(latest_epoch, folder_name, simulations_folder=default_simulations_folder):
    '''
        Re-load a previously saved simulation, recreating the network. This function can
        be used to continue an interrupted simulation.

        Arguments:
            latest_epoch (int)          : The latest epoch of this simulation that has been completed.
            folder_name (string)        : Name of the subfolder in the parent folder that contains data from this simulation.
            simulations_folder (string) : Name of the parent folder that contains the folder for this simulation.

        Returns:
            net (Network)             : Network object with re-loaded weights.
            f_etas (tuple)            : Learning rates for each layer's feedforward weights, eg. (0.21, 0.21).
            b_etas (tuple)            : Learning rates for each layer's feedback weights.
            n_training_examples (int) : Number of training examples per epoch.
    '''

    simulation_path = os.path.join(simulations_folder, folder_name)

    print("Loading simulation from \"{}\" @ epoch {}.\n".format(simulation_path, latest_epoch))

    if not os.path.exists(simulation_path):
        print("Error: Could not find simulation folder – path does not exist.")
        return None

    # load parameters
    with open(os.path.join(simulation_path, 'simulation.json'), 'r') as simulation_file:
        params = json.loads(simulation_file.read())

    # set global parameters
    global nonspiking_mode
    global n_full_test, n_quick_test
    global use_rand_phase_lengths, use_rand_plateau_times, use_conductances, use_broadcast, use_spiking_feedback, use_spiking_feedforward
    global use_symmetric_weights, noisy_symmetric_weights
    global use_sparse_feedback, update_feedback_weights, use_backprop, use_apical_conductance, use_weight_optimization, use_feedback_bias, initial_test
    global record_backprop_angle, record_loss, record_training_error, record_training_labels, record_phase_times, record_plateau_times, record_voltages, record_eigvals, record_matrices, plot_eigvals
    global dt, mem, integration_time, integration_time_test
    global l_f_phase, l_t_phase, l_f_phase_test
    global lambda_max
    global tau_s, tau_L
    global g_B, g_A, g_L, g_D
    global k_B, k_D, k_I
    global P_hidden, P_final
    global kappas

    nonspiking_mode = params['nonspiking_mode']
    n_full_test = params['n_full_test']
    n_quick_test = params['n_quick_test']
    use_rand_phase_lengths = params['use_rand_phase_lengths']
    use_rand_plateau_times = params['use_rand_plateau_times']
    use_conductances = params['use_conductances']
    use_broadcast = params['use_broadcast']
    use_spiking_feedback = params['use_spiking_feedback']
    use_spiking_feedforward = params['use_spiking_feedforward']
    use_symmetric_weights = params['use_symmetric_weights']
    use_sparse_feedback = params['use_sparse_feedback']
    update_feedback_weights = params['update_feedback_weights']
    use_backprop = params['use_backprop']
    use_apical_conductance = params['use_apical_conductance']
    use_weight_optimization = params['use_weight_optimization']
    use_feedback_bias = params['use_feedback_bias']
    initial_test = params['initial_test']
    record_backprop_angle = params['record_backprop_angle']
    record_loss = params['record_loss']
    record_training_error = params['record_training_error']
    record_training_labels = params['record_training_labels']
    record_phase_times = params['record_phase_times']
    record_plateau_times = params['record_plateau_times']
    record_voltages = params['record_voltages']
    record_eigvals = params['record_eigvals']
    record_matrices = params['record_matrices']
    plot_eigvals = params['plot_eigvals']
    dt = params['dt']
    mem = params['mem']
    integration_time = params['integration_time']
    integration_time_test = params['integration_time_test']
    l_f_phase = params['l_f_phase']
    l_t_phase = params['l_t_phase']
    l_f_phase_test = params['l_f_phase_test']
    lambda_max = params['lambda_max']
    tau_s = params['tau_s']
    tau_L = params['tau_L']
    g_B = params['g_B']
    g_A = params['g_A']
    g_L = params['g_L']
    g_D = params['g_D']
    k_B = params['k_B']
    k_D = params['k_D']
    k_I = params['k_I']
    P_hidden = params['P_hidden']
    P_final = params['P_final']

    n = params['n']
    f_etas = params['f_etas']
    b_etas = params['b_etas']
    n_training_examples = params['n_training_examples']

    if nonspiking_mode:
        print("* ------------ Running in non-spiking mode. ------------ *")

        # set parameters for non-spiking mode
        use_rand_phase_lengths = False
        use_rand_plateau_times = False
        use_conductances = False
        use_spiking_feedforward = False
        use_spiking_feedback = False
        record_phase_times = False
        record_plateau_times = False
        record_voltages = False

        l_f_phase = 2
        l_t_phase = 2
        l_f_phase_test = 2
        integration_time = 1
        integration_time_test = 1
        mem = 1

    # create network and load weights
    net = Network(n=n)
    net.load_weights(simulation_path, prefix="epoch_{}_".format(latest_epoch))
    net.current_epoch = latest_epoch + 1

    # re-initialize kappas array
    kappas = np.flipud(get_kappas(mem))[:, np.newaxis]

    return net, f_etas, b_etas, n_training_examples

# --- MNIST --- #


def save_MNIST(x_train, t_train, x_test, t_test, x_valid=None, t_valid=None):
    '''
        Save MNIST data arrays to .npy files. Each data array has size M x N,
        where M is the size of the inputs/targets (ie. 784 or 10 for MNIST),
        and N is the number of examples in the set.

        Arguments:
            x_train (ndarray) : Training inputs.
            t_train (ndarray) : Training targets.
            x_test (ndarray)  : Testing inputs.
            t_test (ndarray)  : Testing targets.
            x_valid (ndarray) : Validation inputs.
            t_valid (ndarray) : Validation targets.
    '''

    np.save("x_train.npy", x_train)
    np.save("x_test.npy", x_test)
    np.save("t_train.npy", t_train)
    np.save("t_test.npy", t_test)

    if x_valid != None and t_valid != None:
        np.save("x_valid.npy", x_valid)
        np.save("t_valid.npy", t_valid)


def load_MNIST(n_valid=0):
    '''
        Load MNIST data arrays from .npy files. Each data array has size M x N,
        where M is the size of the inputs/targets (ie. 784 or 10 for MNIST),
        and N is the number of examples in the set.

        Arguments:
            n_valid (int) : Number of validation examples that are saved.

        Returns:
            x_train (ndarray) : Training inputs.
            t_train (ndarray) : Training targets.
            x_test (ndarray)  : Testing inputs.
            t_test (ndarray)  : Testing targets.
            x_valid (ndarray) : Validation inputs. Returned if n_valid > 0.
            t_valid (ndarray) : Validation targets. Returned if n_valid > 0.
    '''

    try:
        x_train = np.load("x_train.npy")
        x_test = np.load("x_test.npy")
        t_train = np.load("t_train.npy")
        t_test = np.load("t_test.npy")
        if n_valid != 0:
            x_valid = np.load("x_valid.npy")
            t_valid = np.load("t_valid.npy")
    except:
        print("Could not find MNIST .npy files in the current directory.\nLooking for original binary files...")
        try:
            if n_valid != 0:
                x_train, t_train, x_test, t_test, x_valid, t_valid = get_MNIST(
                    n_valid)
                save_MNIST(x_train, t_train, x_test, t_test, x_valid, t_valid)
            else:
                x_train, t_train, x_test, t_test = get_MNIST()
                save_MNIST(x_train, t_train, x_test, t_test)
        except:
            return

        print("Saved .npy files in the current working directory.\n")

    if n_valid != 0:
        return x_train, t_train, x_test, t_test, x_valid, t_valid
    else:
        return x_train, t_train, x_test, t_test


def get_MNIST(n_valid=0):
    '''
    Open original MNIST binary files (which can be obtained from
    http://yann.lecun.com/exdb/mnist/) and generate arrays of
    training, tuning & testing input & target vectors that are
    compatible with our neural network.

    The four binary files:

        train-images.idx3-ubyte
        train-labels.idx1-ubyte
        t10k-images.idx3-ubyte
        t10k-labels.idx1-ubyte

    are expected to be in the same directory as this script.

    Each data array has size M x N, where M is the size of the
    inputs/targets (ie. 784 or 10 for MNIST), and N is the
    number of examples in the set.

    Arguments:
        n_valid (int) : Number of validation examples to use.

    Returns:
        x_train (ndarray) : Training inputs.
        t_train (ndarray) : Training targets.
        x_test (ndarray)  : Testing inputs.
        t_test (ndarray)  : Testing targets.
        x_valid (ndarray) : Validation inputs. Returned if n_valid > 0.
        t_valid (ndarray) : Validation targets. Returned if n_valid > 0.
    '''

    import MNIST

    if (os.path.isfile("train-images.idx3-ubyte") and
        os.path.isfile("train-labels.idx1-ubyte") and
        os.path.isfile("t10k-images.idx3-ubyte") and
            os.path.isfile("t10k-labels.idx1-ubyte")):
        print("Found original MNIST files. Converting to .npy files...")
        try:
            trainfeatures, trainlabels = MNIST.traindata()
            testfeatures, testlabels = MNIST.testdata()
        except:
            print("Error: Could not convert original MNIST files.")
            return
    else:
        print("Error: Could not find original MNIST files in the current directory.\nMake sure that all four binary files are in the current working directory.")
        return

    # normalize inputs
    if n_valid > 0:
        x_valid = trainfeatures[:, :n_valid]/255.0

    x_train = trainfeatures[:, n_valid:]/255.0
    x_test = testfeatures/255.0

    n_train = trainlabels.size - n_valid

    # generate target vectors
    if n_valid > 0:
        t_valid = np.zeros((10, n_valid))
        for i in range(n_valid):
            t_valid[int(trainlabels[i]), i] = 1

    t_train = np.zeros((10, n_train))
    for i in xrange(n_train):
        t_train[int(trainlabels[n_valid + i]), i] = 1

    n_test = testlabels.size
    t_test = np.zeros((10, n_test))
    for i in xrange(n_test):
        t_test[int(testlabels[i]), i] = 1

    if n_valid > 0:
        return x_train, t_train, x_test, t_test, x_valid, t_valid
    else:
        return x_train, t_train, x_test, t_test


def shuffle_arrays(*args):
    '''
    Shuffle multiple arrays using the same random permutation.

    Arguments:
        args (tuple of ndarrays) : Arrays to shuffle.

    Returns:
        results (tuple of ndarrays) : Shuffled arrays.
    '''

    p = np.random.permutation(args[0].shape[1])
    results = (a[:, p] for a in args)
    return results

# --- Misc. --- #


def plot_f_weights(self, normalize=False, save_path=None):
    plot_weights(self.W, normalize=normalize)


def plot_b_weights(self, normalize=False, save_path=None):
    plot_weights(self.Y, normalize=normalize)


def plot_weights(W_list, save_dir=None, suffix=None, normalize=False):
    '''
    Plots receptive fields given by weight matrices in W_list.

    Arguments:
        W_list (list of ndarrays) : List of weight matrices to plot.
        save_dir (string)         : Directory in which to save the plot.
        suffix (string)           : Suffix to add to the end of the filename of the plot.
        normalize (bool)          : Whether to normalize each receptive field. If True,
                                    the vmin and vmax of each receptive field subplot will
                                    be independent from the vmin and vmax of the other subplots.
    '''

    def prime_factors(n):
        # Get all prime factors of a number n.
        factors = []
        lastresult = n
        if n == 1:  # 1 is a special case
            return [1]
        while 1:
            if lastresult == 1:
                break
            c = 2
            while 1:
                if lastresult % c == 0:
                    break
                c += 1
            factors.append(c)
            lastresult /= c
        print("Factors of %d: %s" % (n, str(factors)))
        return factors

    def find_closest_divisors(n):
        # Find divisors of a number n that are closest to its square root.
        a_max = np.floor(np.sqrt(n))
        if n % a_max == 0:
            a = a_max
            b = n/a
        else:
            p_fs = prime_factors(n)
            candidates = np.array([1])
            for i in xrange(len(p_fs)):
                f = p_fs[i]
                candidates = np.union1d(candidates, f*candidates)
                candidates[candidates > a_max] = 0
            a = candidates.max()
            b = n/a
        print("Closest divisors of %d: %s" % (n, str((int(b), int(a)))))
        return (int(a), int(b))

    plt.close('all')

    fig = plt.figure(figsize=(18, 9))

    M = len(W_list)

    n = [W.shape[0] for W in W_list]
    n_in = W_list[0].shape[-1]

    print(M, n)

    grid_specs = [0]*M
    axes = [[0]*i for i in n]

    max_Ws = [np.amax(W) for W in W_list]

    min_Ws = [np.amin(W) for W in W_list]

    W_sds = [np.std(W) for W in W_list]
    W_avgs = [np.mean(W) for W in W_list]

    for m in xrange(M):
        print("Layer {0} | W_avg: {1:.6f}, W_sd: {2:.6f}.".format(
            m, np.mean(W_list[m]), np.std(W_list[m])))

    for m in xrange(M):
        if m == 0:
            img_Bims = find_closest_divisors(n_in)
        else:
            img_Bims = grid_dims

        grid_dims = find_closest_divisors(n[m])
        # tanspose grid dimensions, to better fit the space
        grid_dims = (grid_dims[1], grid_dims[0])

        grid_specs[m] = gs.GridSpec(grid_dims[0], grid_dims[1])

        for k in xrange(n[m]):
            row = k // grid_dims[1]
            col = k - row*grid_dims[1]

            axes[m][k] = fig.add_subplot(grid_specs[m][row, col])
            if normalize:
                heatmap = axes[m][k].imshow(W_list[m][k].reshape(
                    img_Bims).T, interpolation="nearest", cmap=weight_cmap)
            else:
                heatmap = axes[m][k].imshow(W_list[m][k].reshape(img_Bims).T, interpolation="nearest",
                                            vmin=W_avgs[m] - 3.465*W_sds[m], vmax=W_avgs[m] + 3.465*W_sds[m], cmap=weight_cmap)
            axes[m][k].set_xticklabels([])
            axes[m][k].set_yticklabels([])

            axes[m][k].tick_params(axis='both',  # changes apply to the x-axis
                                   which='both',  # both major and minor ticks are affected
                                   bottom='off',  # ticks along the bottom edge are off
                                   top='off',    # ticks along the top edge are off
                                   left='off',   # ticks along the left edge are off
                                   right='off')  # ticks along the right edge are off

            if m == M-1 and k == 0:
                plt.colorbar(heatmap)

        grid_specs[m].update(left=float(m)/M,
                             right=(m+1.0)/M,
                             hspace=1.0/(grid_dims[0]),
                             wspace=0.05,
                             bottom=0.02,
                             top=0.98)

    if save_dir != None:
        if suffix != None:
            plt.savefig(save_dir + 'weights' + suffix + '.png')
        else:
            plt.savefig(save_dir + 'weights.png')
    else:
        plt.show()
