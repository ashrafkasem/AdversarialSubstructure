# -*- coding: utf-8 -*-

""" ...

@file:   utils.py
@author: Andreas Søgaard
@date:   20 February 2017
@email:  andreas.sogaard@cern.ch
"""

# Basic include(s)
import json
import os

# Numpy include(s)
import numpy as np

# Keras include(s)
from keras.models import model_from_json

# Scikit-learn include(s)
from sklearn import preprocessing

# Local include(s)
from snippets.functions import *

def save_architecture_and_weights (model, name=None):
    """ Save arbitrary model architecture and -weights to file(s). """

    # Check(s)
    name = name or model.name
    assert type(name) == str and len(name) > 0 and name[0] != '/', "Model name '{}' not accepted.".format(name)

    # Save model
    json_string = model.to_json()
    with open('{}_architecture.json'.format(name), 'wb') as f:
        json.dump(json_string, f)
        pass

    # Save weights
    model.save_weights('{}_weights.h5'.format(name))
    return


def load_architecture_and_weights (name):
    """ Load arbitrary model architecture and -weights from file(s). """

    # Check(s)
    assert type(name) == str and len(name) > 0 and name[0] != '/', "Model name '{}' not accepted.".format(name)

    # Load model
    json_string = open('{}_architecture.json'.format(name), 'r').read()
    model = model_from_json(json_string)
    
    # Save weights
    model.load_weights('{}_weights.h5'.format(name))
    Return


def save_checkpoint (model):
    """ ... """
    model.save_weights('.%s_checkpoint.h5' % model.name)
    return


def load_checkpoint (model):
    """ ... """
    model.load_weights('.%s_checkpoint.h5' % model.name)
    return


def roc (sig, bkg, sig_weight=None, bkg_weight=None):
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


def getData(decorrelation_vars=[]):
    """ ... """ 

    # Read input files
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Get list of file paths to plot from commandline arguments.
    path = 'csv_data/'
    paths = [path + f for f in ['signal.csv', 'background.csv']]

    assert len(paths) > 0, "Must have at least one input file."

    with open(paths[0], 'r') as f:
        names = f.readline().strip('#\n').replace(' ', '').replace('\t', '').split(',')
        pass
    dtype = {'names': tuple(names), 'formats': tuple(np.float for _ in names)}
    
    arrays = list()
    for path in paths:
        path_npz = path.replace('.csv', '.npz')
        if os.path.isfile(path_npz):
            print "NPZ file '{}' exists. Reading from there.".format(path_npz)
            f = np.load(path_npz)
            arrays.append(f['arr_0']) # @TODO: Better names?
            names = f['arr_1']
        else:
            print "NPZ file '{}' doesn't exist. Reading from '{}' and saving to NPZ.".format(path_npz, path)
            arrays.append(np.loadtxt(path, delimiter=',', skiprows=1, dtype=dtype))
            np.savez(path_npz, arrays[-1], names)
            pass
        pass

    # Dicti-fy array
    signal     = {key: arrays[0][key] for key in names}
    background = {key: arrays[1][key] for key in names}

    # Check output.
    if (not background) or (not signal):
        print "WARNING: No values were loaded."
        return

    # Discard unphysical jets.
    for values in [background, signal]:
        msk_good = reduce(np.intersect1d, (np.where(values['jet_pt']    > 0),
                                           np.where(values['jet_m']     > 0),
                                           np.where(values['jet_tau21'] > 0),
                                           np.where(values['jet_D2']    > 0),
                                           ))

        for var, arr in values.items():
            values[var] = arr[msk_good]
            pass

        # New variable(s)
        values['jet_rho'] = np.log(np.square(values['jet_m']) / (np.square(values['jet_pt']) + 1.0E-09))

        pass

    # Get number of signal and background examples
    num_signal     = len(signal    ['weight'])
    num_background = len(background['weight'])
        

    # Turn dictionaries into numpy arrays
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Remove 'weight' from input array
    names = [name for name in names if name != 'weight']

    # Weights
    W_signal     = signal    ['weight'] / np.sum(signal    ['weight']) * float(background['weight'].size) 
    W_background = background['weight'] / np.sum(background['weight']) * float(background['weight'].size) 
    W = np.hstack((W_signal, W_background))
    
    # Labels
    Y = np.hstack((np.ones(num_signal, dtype=int), np.zeros(num_background, dtype=int)))

    # Input(s)
    X_signal     = np.vstack(tuple(signal    [var] for var in names)).T
    X_background = np.vstack(tuple(background[var] for var in names)).T

    # De-correlation variable(s)
    P_signal     = np.vstack(tuple(signal    [var] for var in decorrelation_vars)).T
    P_background = np.vstack(tuple(background[var] for var in decorrelation_vars)).T
    
    # Data pre-processing
    substructure_scaler = preprocessing.StandardScaler().fit(X_background)
    X_signal     = substructure_scaler.transform(X_signal)
    X_background = substructure_scaler.transform(X_background)

    decorrelation_scaler = preprocessing.StandardScaler().fit(P_background)
    P_signal     = decorrelation_scaler.transform(P_signal)
    P_background = decorrelation_scaler.transform(P_background)

    # @TODO: Make recorded array? Is that compatible with Keras, etc.?
    X = np.vstack((X_signal, X_background))
    P = np.vstack((P_signal, P_background))

    return X, Y, W, P, signal, background, names



def wpercentile (values, eff, weights=None):
    """ Get the weighted percentile of array. """

    # Check(s)
    if weights is None:
        weights = np.ones_like(values)
        pass

    # Concatenate values and (relative) weights in single matrix
    matrix = np.column_stack((values, weights / np.sum(weights)))

    # Sort by ascending value
    matrix_sort = matrix[matrix[:,0].argsort()]

    # Get cumulative sum of weights
    matrix_sort[:,1] = np.cumsum(matrix_sort[:,1])

    # Check that efficiency is reasonable
    if eff <=  matrix_sort[0,1]:
        return matrix_sort[0,0]
    if eff >=  matrix_sort[-1,1]:
        return matrix_sort[-1,0]

    # Get indices closest to 'eff'
    idx1 = np.where(matrix_sort[:,1] < eff)[0]

    # If no counts are below efficiency, break early
    if len(idx1) == 0: return None
    idx1 = idx1[-1]
    idx2 = idx1 + 1

    # Get closest values and weights
    v1, v2 = matrix_sort[[idx1, idx2],0]
    w1, w2 = matrix_sort[[idx1, idx2],1]

    # Perform linear interpolation
    percentile = v1 + (eff - w1) / (w2 - w1) * (v2 - v1)

    return percentile


def weighted_avg_and_std(values, weights):
    """ From [http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy]
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average  = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))
