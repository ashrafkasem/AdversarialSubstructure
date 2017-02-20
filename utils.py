# -*- coding: utf-8 -*-

""" ...

@file:   utils.py
@author: Andreas Søgaard
@date:   20 February 2017
@email:  andreas.sogaard@cern.ch
"""

# Numpy include(s)
import numpy as np

# Scikit-learn include(s)
from sklearn import preprocessing

# Local include(s)
from snippets.functions import *


def save_checkpoint (model):
    """ ... """
    model.save_weights('.%s_checkpoint.h5' % model.name)
    return


def load_checkpoint (model):
    """ ... """
    model.load_weights('.%s_checkpoint.h5' % model.name)
    return


def roc (sig, bkg, sig_weight = None, bkg_weight = None):
    """ Return the signal and background efficiencies for successive cuts """
    
    # Check(s).
    if sig_weight is None:
        sig_weight = np.ones_like(sig)
        pass

    if bkg_weight is None:
        bkg_weight = np.ones_like(bkg)
        pass

    # Store and sort 2D array
    sig2 = np.vstack((sig.ravel(), sig_weight.ravel(), np.zeros_like(sig.ravel()))).T
    bkg2 = np.vstack((bkg.ravel(), np.zeros_like(bkg.ravel()), bkg_weight.ravel())).T
    sig_bkg      = np.vstack((sig2, bkg2))
    sig_bkg_sort = sig_bkg[sig_bkg[:,0].argsort()]

    # Accumulated (weighted) counts
    eff_sig = np.cumsum(sig_bkg_sort[:,1]) / np.sum(sig_weight)
    eff_bkg = np.cumsum(sig_bkg_sort[:,2]) / np.sum(bkg_weight)

    # Make sure that cut direction is correct
    if np.sum(eff_sig < eff_bkg) > len(eff_sig) / 2:
        eff_sig = 1. - eff_sig
        eff_bkg = 1. - eff_bkg
        pass

    return eff_sig, eff_bkg


def getData(args, input_vars = ['m', 'tau21', 'D2']):
    """ ... """ 

    # Read input files
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Validate input arguments
    validateArguments(args)

    # Load cross sections files.
    xsec = loadXsec('./sampleInfo.csv')

    # Get list of file paths to plot from commandline arguments.
    paths = [arg for arg in args[1:] if not arg.startswith('-')]

    # Specify which variables to get.
    treename = 'BoostedJet+ISRgamma/EventSelection/Pass/NumFatjetsAfterDphi/Postcut'
    prefix   = 'leadingfatjet_'

    getvars  = ['pt'] + input_vars

    # Load data.
    print "\n== Reading background (gamma + jets)",
    background = loadDataFast(paths, treename, getvars, prefix, xsec,
                              keepOnly = (lambda DSID: 361039 <= DSID <= 361062 )
                              )

    print "\n== Reading signal (gamma + W)",
    signal     = loadDataFast(paths, treename, getvars, prefix, xsec,
                              keepOnly = (lambda DSID: 305435 <= DSID <= 305439 )
                              )


    # Check output.
    if (not background) or (not signal):
        print "WARNING: No values were loaded."
        return

    # Discard unphysical jets.
    for values in [background, signal]:
        msk_good = reduce(np.intersect1d, (np.where(values['pt']    > 0),
                                           np.where(values['m']     > 0),
                                           np.where(values['tau21'] > 0),
                                           np.where(values['D2']    > 0),
                                           ))

        for var, arr in values.items():
            values[var] = arr[msk_good]
            pass

        pass

    # Get number of signal and background examples
    num_signal     = len(signal    [getvars[0]])
    num_background = len(background[getvars[0]])
        

    # Turn dictionaries into numpy arrays
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Weights
    W_signal     = signal    ['weight'] / np.sum(signal    ['weight']) * float(background['weight'].size) 
    W_background = background['weight'] / np.sum(background['weight']) * float(background['weight'].size) 
    W = np.hstack((W_signal, W_background))
    
    # Labels
    Y = np.hstack((np.ones(num_signal, dtype = int), np.zeros(num_background, dtype = int)))

    # Inputs
    num_params = len(input_vars)

    X_signal     = np.vstack(tuple(signal    [var] for var in input_vars)).T
    X_background = np.vstack(tuple(background[var] for var in input_vars)).T

    input_scaler = preprocessing.StandardScaler().fit(X_background)
    X_signal     = input_scaler.transform(X_signal)
    X_background = input_scaler.transform(X_background)

    # @TODO: Make recorded array? Is that compatible with Keras, etc.?
    X = np.vstack((X_signal, X_background))

    return X, Y, W, signal, background
