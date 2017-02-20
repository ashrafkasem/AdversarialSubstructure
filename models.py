# -*- coding: utf-8 -*-

""" ...

@file:   models.py
@author: Andreas Søgaard
@date:   20 February 2017
@email:  andreas.sogaard@cern.ch
"""

# Keras include(s)
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD

# Local include(s)
from layers import *


# Definitions for both models
num_nodes_f = 32 
num_nodes_r = 32

# Create optimisers for each main model
params = {
    'lambda':  70.,
    'lr':       0.01,
    'momentum': 0.9,
}

c_optim = SGD(lr = params['lr'], momentum = params['momentum'], nesterov = True)
d_optim = SGD(lr = params['lr'], momentum = params['momentum'], nesterov = True)

# Define compiler options
opts = {

    # Classifier
    'classifier' : {
        'loss': 'binary_crossentropy',
        'optimizer': 'SGD',
    },

    # Discriminator
    'discriminator' : {
        'loss': 'binary_crossentropy',
        'optimizer': d_optim,
    },

    # Combined model
    'combined' : {
        'loss': ['binary_crossentropy', 'binary_crossentropy'],
        'optimizer': c_optim,
        'loss_weights': [1., -params['lambda']],
    }

}


def classifier_model (num_params): # ..., arch):
    """ Returns an instance of a classifier-type model (f) """
    
    # Input(s)
    input_f = Input(shape = (num_params,), name = 'input_f')

    # Layer(s)
    f_1 = Dense(num_nodes_f, activation = 'relu')(input_f)
    f_2 = Dense(num_nodes_f, activation = 'relu')(f_1)
    f_3 = Dense(num_nodes_f, activation = 'relu')(f_2)
    f_4 = Dense(num_nodes_f, activation = 'relu')(f_3)
    
    # Output(s)
    output_f = Dense(1, activation = 'sigmoid', name = 'output_f')(f_4)

    # Model
    return Model(input = input_f, output = output_f, name = 'classifier')


def discriminator_model (num_components): # ..., arch):
    """ Returns an instance of a discriminator-type model (r) """
    
    # Input(s)
    input_r      = Input(shape = (1,), name = 'input_r')
    input_masses = Input(shape = (1,), name = 'input_masses')

    # Layer(s)
    r_1 = Dense(num_nodes_r, activation = 'relu')(input_r)
    r_2 = Dense(num_nodes_r, activation = 'relu')(r_1)
    r_3 = Dense(num_nodes_r, activation = 'relu')(r_2)
    r_4 = Dense(num_nodes_r, activation = 'relu')(r_3)
    r_coeffs = Dense(num_components, activation = 'softmax')(r_4)
    r_means  = Dense(num_components)(r_4) 
    r_widths = Dense(num_components, activation = 'softplus')(r_4)

    # Output(s)
    output_r  = Posterior(1, num_components, name = 'output_r')([r_coeffs, r_means, r_widths, input_masses])

    # Model
    return Model(input = [input_r, input_masses], output = output_r, name = 'discriminator')


def combined_model (classifier, discriminator):
    """ Returns an instance of the combined model, merging the input classifier- and discriminator model. 
    Inspired by [https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py] 
    """

    combined = Model(input  = [classifier.input,  discriminator.input[1]], 
                     output = [classifier.output, discriminator([classifier.output, discriminator.input[1]])],
                     name = 'combined')

    return combined
    