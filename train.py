#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script for training an adversarial neural network (ANN) for constructing an optimal combination of jet substructure variables, which is de-correlated from the jet mass.

@file    train.py
@author: Andreas Søgaard
@date:   20 February 2017
@email:  andreas.sogaard@cern.ch
"""

# Basic include(s)
import sys
import json

# Scientific include(s)
# -- Numpy
import numpy as np

# -- Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -- Keras
from keras import backend as K_
from keras.models import load_model
from keras.utils.visualize_util import plot
from keras.callbacks import Callback, LearningRateScheduler, ReduceLROnPlateau

# -- Scikit-learn
from sklearn import preprocessing

# -- Scipy
from scipy.signal import savgol_filter

# Local include(s)
from utils import *
from models import *


# Main function definition.
def main ():

    # Set pyplot style
    plt.style.use('ggplot')

    # Whether to continue training where last round left off
    resume = False

    retrain_classifier = True


    # Get data
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    substructure_vars  = ['jet_tau21', 'jet_D2', 'jet_m']
    decorrelation_vars = ['jet_m'] 
    X, Y, W, P, signal, background, names = getData(decorrelation_vars)

  
    # Split into train and test sample
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    
    # Compute number of train- and test examples
    frac_train = 0.8
    num_total = X.shape[0]
    num_train = int(num_total * frac_train)
    num_test  = num_total - num_train

    # Create training example mask
    indices = np.random.choice(num_total, num_train, replace = False)
    msk = np.zeros((num_total,), dtype = bool)
    msk[indices] = True

    # Initialise train- and test inputs, labels, and weights
    X_train, X_test = X[msk,:], X[~msk,:]
    Y_train, Y_test = Y[msk],   Y[~msk]
    W_train, W_test = W[msk],   W[~msk]
    P_train, P_test = P[msk,:], P[~msk,:]
    
    
    # Define training data sampler
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––    

    def sampler (size_batch = 1.0, num_batches = 1, replace = True):
        """ Generator to produce random, shuffled subsets of the training data. (Probably super inefficient in practice) """

        # Check(s)
        if type(size_batch) is int:
            if not replace:
                assert size_batch <= num_train, "The requested number of samples (%d) is larger than the number of unique training samples (%d)." % (size_batch, num_train)
                pass
        else:
            assert 0. < size_batch and size_batch <= 1., "The requested fraction of samples (%.2e) is not in range (0,1]." % size_batch
            size_batch = int(size_batch * num_train)
            pass

        # Get list of batch indices
        batches = np.array_split( np.random.choice(num_train, size_batch * num_batches, replace = replace), num_batches )
        
        # Loop list of batch indices
        for batch_indices in batches:
            
            # Get input arrays for mini-batch
            X_batch = X_train[batch_indices,:]
            Y_batch = Y_train[batch_indices]
            W_batch = W_train[batch_indices]
            P_batch = P_train[batch_indices]

            # Yield batch arrays
            yield X_batch, Y_batch, W_batch, P_batch

        pass


    # Train classifier model
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––    

    # Fit non-adversarial neural network
    if not retrain_classifier:
        print "\n== Loading classifier model."

        # Load existing classifier model from file
        classifier = load_model('classifier.h5')

    else:
        print "\n== Fitting non-adversarial classifier model."

        # Create new classifier model instance
        classifier = classifier_model(X.shape[1], architecture=[(128, 'tanh')] * 4)

        # Compile with optimiser configuration
        classifier.compile(**opts['classifier'])

        # Get training samples
        X_batch, Y_batch, W_batch, _ = sampler(replace=False).next()

        # Fit classifier model
        classifier.fit(X_batch, Y_batch, sample_weight=W_batch, batch_size=1024, nb_epoch=100) 

        # Save classifier model to file
        classifier.save('classifier.h5')
        pass


    # Set up combined, adversarial model
    adversarial = adversarial_model(classifier, architecture=[(64, 'tanh')] * 2, num_posterior_components=1, num_posterior_dimensions=P_train.shape[1])

    if resume: 
        load_checkpoint(adversarial)
        pass

    adversarial.compile(**opts['adversarial'])

    # Save adversarial model diagram
    plot(adversarial, to_file='adversarial.png', show_shapes=True)

    # Set fit options
    fit_opts = {
        'shuffle':          True,
        'validation_split': 0.2,
        'batch_size':       4 * 1024,
        'nb_epoch':         100,
        'sample_weight':    [W_train, np.multiply(W_train, 1. - Y_train)]
    }

    # -- Callback for storing costs at batch-level
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.lossnames = ['loss', 'classifier_loss', 'adversary_loss']
            self.losses = {name: list() for name in self.lossnames}
            return

        def on_batch_end(self, batch, logs={}):
            for name in self.lossnames:
                self.losses[name].append(float(logs.get(name)))
                pass
            return
        pass

    history = LossHistory()

    # -- Callback for updating learning rate(s)
    damp = np.power(1.0E-04, 1./float(fit_opts['nb_epoch']))
    def schedule (epoch):
        """ Update the learning rate of the two optimisers. """
        if 0 < damp and damp < 1:
            K_.set_value(adv_optim.lr, damp * K_.get_value(adv_optim.lr))
            pass
        return float(K_.eval(adv_optim.lr))

    change_lr = LearningRateScheduler(schedule)

    # -- Callback for saving model checkpoints
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath=".adversarial_checkpoint.h5", verbose=0, save_best_only=False)

    # -- Callback to reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1E-07)

    # Store callbacks in fit options
    fit_opts['callbacks'] = [history, change_lr, checkpointer]

    # Fit the combined, adversarial model
    adversarial.fit([X_train, P_train], [Y_train, np.ones_like(Y_train)], **fit_opts)
    hist = history.losses

    # Save cost log to file
    with open('cost.log', 'a' if resume else 'w') as cost_log:
        line  = "# "
        line += ", ".join(['%s' % name for name in history.lossnames])
        line += " \n"
        cost_log.write(line) 

        cost_array = np.squeeze(np.array(zip(hist.values())))
        for row in range(cost_array.shape[1]):
            costs = list(cost_array[:,row])
            line = ', '.join(['%.4e' % cost for cost in costs])
            line += " \n"
            cost_log.write(line)    
            pass
        pass

    '''
    # Plot cost log
    colors = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]

    for i, (key,l) in enumerate(hist.items()):
        name = key.replace('loss', '').replace('_', '')
        #name = name or 'combined'
        if name:
            name = r'$L_{%s}$' % name
        else:
            name = r'$L_{classifier} - \lambda L_{adversary}$'
            pass
        plt.plot(l, alpha=0.4, label=name, color=colors[i])
        plt.plot(savgol_filter(l, 101, 3), color=colors[i])
        pass
    clf_opt = hist['classifier_loss'][0]
    N = len(hist['classifier_loss'])
    plt.plot([0, N - 1], [clf_opt, clf_opt], color='gray', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid()
    plt.show()
    '''

    # Save adversarial model (?)
    save_architecture_and_weights(adversarial)
    save_checkpoint(adversarial)
    
    print "\n== Done."
    
    return


# Main function call.
if __name__ == '__main__':
   main()
   pass
