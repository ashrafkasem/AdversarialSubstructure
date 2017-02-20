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
    
# -- Scikit-learn
from sklearn import preprocessing

# Local include(s)
from utils import *
from models import *


# Main function definition.
def main ():

    # Set pyplot style
    plt.style.use('ggplot')

    # Whether to continue training where last round left off
    resume = False


    # Get data
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    input_vars = ['m', 'tau21', 'D2']
    X, Y, W, signal, background = getData(sys.argv, input_vars)


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
    
    # Get mass index
    idx_m = input_vars.index('m')
    

    # Define training data sampler
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––    

    def sampler (size_batch = 1.0, num_batches = 1, replace = True):
        """ ... """

        # Check(s)
        if type(size_batch) is int:
            if not replace:
                assert frac <= num_train, "The requested number of samples (%d) is larger than the number of unique training samples (%d)." % (frac, num_train)
                pass
        else:
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

            # Yield batch arrays
            yield X_batch, Y_batch, W_batch

        pass


    # Train classifier model
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––    

    # Fit non-adversarial neural network
    if resume:
        print "\n== Loading classifier model."

        # Load existing classifier model from file
        classifier = load_model('classifier.h5')

    else:
        print "\n== Fitting non-adversarial classifier model."

        # Create new classifier model instance
        classifier = classifier_model(len(input_vars))

        # Compile with optimiser configuration
        classifier.compile(**opts['classifier'])

        # Get training samples
        X_batch, Y_batch, W_batch = sampler(replace = False).next()

        # Fit classifier model
        classifier.fit(X_batch, Y_batch, sample_weight = W_batch, batch_size = 1024, nb_epoch = 10) # 50

        # Save classifier model to file
        classifier.save('classifier.h5')
        pass



    # Train combined, adversarial model
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––    

    print "\n== Fitting combined, adversarial model."

    # Get discriminator model
    discriminator = discriminator_model(num_components = 5)

    # Optionally, load checkpoints
    if resume:
        load_checkpoint(classifier)
        load_checkpoint(discriminator)
        pass

    # Get combined model
    combined = combined_model(classifier, discriminator)

    # Compile models
    classifier   .compile(**opts['classifier'])
    combined     .compile(**opts['combined'])
    discriminator.compile(**opts['discriminator'])
    
    # Plot models
    plot(classifier,    to_file = 'classifier.png',    show_shapes = True)
    plot(discriminator, to_file = 'discriminator.png', show_shapes = True)
    plot(combined,      to_file = 'combined.png',      show_shapes = True)

    # Initialise trainin parameters
    N  = X_train.shape[0]
    T  = 1000
    M  = 1024 * 4
    K1 =  200
    K2 =    1

    # Set damping factor
    damp = np.power(1.0E-02, 1./float(T))
    
    # Loop iterations.
    for iepoch in range(T):

        print "Iteration [%2d/%2d] " % (iepoch + 1, T)
        
        # Train discriminator model
        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––    
        
        # Train on mini-batches
        for X_batch, Y_batch, W_batch in sampler(M, num_batches = K1):

            # Get array of jet masses
            m_batch = X_batch[:,idx_m]

            # [L3]: Predict classifier without training
            classifier_prob = classifier.predict(X_batch, verbose = 0)

            # [L4]: Train discriminator separately
            cost = discriminator.train_on_batch([classifier_prob, m_batch], np.ones_like(m_batch), sample_weight = np.multiply(W_batch, 1. - Y_batch))
            
            pass
        

        # Train classifier model
        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––    
       
        # Save discriminator checkpoints
        save_checkpoint(discriminator)
        
        # Train on mini-batches
        for X_batch, Y_batch, W_batch in sampler(M, num_batches = K2):
            
            # Get array of jet masses
            m_batch = X_batch[:,idx_m]
            
            # [L7]: Train classifier separately
            costs = combined.train_on_batch([X_batch, m_batch], [Y_batch, np.ones_like(m_batch)], sample_weight = [W_batch, np.multiply(W_batch, 1. - Y_batch)])
            
            # Restore discriminator, since freezing models/layers doesn't really work in combined models
            load_checkpoint(discriminator) 

            pass
    
        # Print
        """
        if (iepoch + 1) % 10 == 0:
            #print "  Batch [%4d/%4d] d_loss : %f" % (ibatch + 1, num_batches, d_loss)
            print "  d_loss:  %-6.3f" % ( d_loss)
            print "  cc_loss: %-6.3f" % (cc_loss)
            print "  cd_loss: %-6.3f" % (cd_loss)
            print "  ct_loss: %-6.3f" % (ct_loss)
            pass
        """

        # Save checkpoints
        if iepoch % 10 == 9:
            save_checkpoint(classifier)
            save_checkpoint(discriminator)
            pass

        # Change learning rate
        if damp < 1.:
            K_.set_value(c_optim.lr, damp * K_.get_value(c_optim.lr))
            K_.set_value(d_optim.lr, damp * K_.get_value(d_optim.lr))
            pass

        pass

    print "\n== Done."
    
    # ...

    return


# Main function call.
if __name__ == '__main__':
   main()
   pass
