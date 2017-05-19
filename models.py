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
from keras.optimizers import SGD, Adam

# Local include(s)
from layers import *

# Learning configurations
params = {
    'lambda':    1000.,
    'lr':        1E-03, # Learning rate (LR) in the classifier model
    'lr_ratio':  1E+03, # Ratio of the LR in adversarial model to that in the classifier
}

# Create optimisers for each main model
clf_optim = Adam(lr=params['lr'], decay=1E-03)
adv_optim = Adam(lr=params['lr'], decay=1E-03)

# Define compiler options
opts = {

    # Classifier
    'classifier' : {
        'loss': 'binary_crossentropy',
        'optimizer': clf_optim, #'SGD',
    },

    # Adversarial (combined)
    'adversarial' : {
        'loss': ['binary_crossentropy', 'binary_crossentropy'],
        'optimizer': adv_optim,
        'loss_weights': [1, params['lr_ratio']],
    },

}


def adversarial_model (classifier, architecture, num_posterior_components, num_posterior_dimensions):
    """ ... """

    # Classifier
    classifier.trainable = True
   
    # Adversary
    # -- Gradient reversal
    l = GradientReversalLayer(params['lambda'] / float(params['lr_ratio']))(classifier.output)

    # -- De-correlation inputs
    input_decorrelation = Input(shape=(num_posterior_dimensions,), name='input_adversary')

    # -- Intermediate layer(s)
    for ilayer, (nodes, activation) in enumerate(architecture):
        l = Dense(nodes, activation=activation, name='dense_adversary_%d' % ilayer)(l)
        pass

    # -- Posterior p.d.f. parameters
    r_coeffs = Dense(num_posterior_components, name='coeffs_adversary', activation='softmax')(l)
    r_means  = list()
    r_widths = list()
    for i in xrange(num_posterior_dimensions):
        r_means .append( Dense(num_posterior_components, name='mean_adversary_P%d'  % i)(l) )
        pass
    for i in xrange(num_posterior_dimensions):
        r_widths.append( Dense(num_posterior_components, name='width_adversary_P%d' % i, activation='softplus')(l) )
        pass

    # -- Posterior probability layer
    output_adversary = Posterior(num_posterior_components, num_posterior_dimensions, name='adversary')([r_coeffs] + r_means + r_widths + [input_decorrelation])
    
    return Model(input=[classifier.input] + [input_decorrelation], output=[classifier.output, output_adversary], name='combined')


def classifier_model (num_params, architecture):
    """ Returns an instance of a classifier-type model (f) """
    
    # Input(s)
    classifier_input = Input(shape=(num_params,), name='input_classifier')

    # Layer(s)
    l = classifier_input
    for ilayer, (nodes, activation) in enumerate(architecture):
        l = Dense(nodes, activation=activation, name='dense_classifier_%d' % ilayer)(l)
        pass

    # Output(s)
    classifier_output = Dense(1, activation='sigmoid', name='classifier')(l)

    # Model
    return Model(input=classifier_input, output=classifier_output, name='classifier')
