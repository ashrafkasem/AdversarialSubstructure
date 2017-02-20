#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script for plotting the results of ...

@file    plotting.py
@author: Andreas Søgaard
@date:   20 February 2017
@email:  andreas.sogaard@cern.ch
"""

# Basic include(s)
import sys

# Scientific include(s)
# -- ROOT
from ROOT import *
from root_numpy import fill_profile, hist2array

# -- Numpy
import numpy as np

# -- Matplotlib
import matplotlib.pyplot as plt

# -- Keras
from keras.models import load_model
    
# -- Scikit-learn
from sklearn import preprocessing

# Local include(s)
from utils import *
from models import *


# Main function definition.
def main ():

    # Set pyplot style
    plt.style.use('ggplot')

    # Whether to save plots
    save = False


    # Get data
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    input_vars = ['m', 'tau21', 'D2']
    X, Y, W, signal, background = getData(sys.argv, input_vars)


    # Load pre-trained classifier
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Load existing classifier model from file
    classifier = load_model('classifier.h5')

    # Add neural network classifier output, without adversarial training
    msk_sig = (Y == 1.)
    signal    ['NN'] = classifier.predict(X[ msk_sig], batch_size = 1024)
    background['NN'] = classifier.predict(X[~msk_sig], batch_size = 1024)

    # Scale to mean 0.5 and sensible range
    scaler = preprocessing.StandardScaler().fit(background['NN'].reshape(-1,1))
    signal    ['NN'] = (scaler.transform(signal    ['NN'].reshape(-1,1)) / 4. + 0.5).reshape(signal    ['m'].shape)
    background['NN'] = (scaler.transform(background['NN'].reshape(-1,1)) / 4. + 0.5).reshape(background['m'].shape)

    # Remember to use 'NN' in comparisons later
    input_vars += ['NN']


    # Load adversarially trained models
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Classifier
    load_checkpoint(classifier)

    # Discriminator
    discriminator = discriminator_model(5)
    load_checkpoint(discriminator)

    # Add neural network classifier output, without adversarial training
    msk_sig = (Y == 1.)
    signal    ['ANN'] = classifier.predict(X[ msk_sig], batch_size = 1024)
    background['ANN'] = classifier.predict(X[~msk_sig], batch_size = 1024)

    # Scale to mean 0.5 and sensible range
    scaler = preprocessing.StandardScaler().fit(background['ANN'].reshape(-1,1))
    signal    ['ANN'] = (scaler.transform(signal    ['ANN'].reshape(-1,1)) / 4. + 0.5).reshape(signal    ['m'].shape)
    background['ANN'] = (scaler.transform(background['ANN'].reshape(-1,1)) / 4. + 0.5).reshape(background['m'].shape)

    # Remember to use 'ANN' in comparisons later
    input_vars += ['ANN']


    # Plot 1D distribution(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    print "\n1D distributions:"

    h_sig = dict()
    h_bkg = dict()
    for var in input_vars:
        print "-- %s" % var
        bins = np.linspace(0, 4.0 if var == 'D2' else (300. if var == 'm' else 1.0), 100 + 1, True)
        h_bkg[var] = plt.hist(background[var], bins, weights = background['weight'],      alpha = 0.6, label = 'Background')
        h_sig[var] = plt.hist(signal    [var], bins, weights = signal    ['weight'] * 20, alpha = 0.6, label = 'Signal (x 20)')
        plt.xlim([bins[0], bins[-1]])
        plt.xlabel(r'%s' % displayNameUnit(var, latex = True))
        plt.ylabel(r'Events [fb]')
        plt.legend()
        if save: plt.savefig('distrib_%s.pdf' % var)
        plt.show()
        pass


    # Plot ROC curve(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    print "\nROC curves:"

    eff_sig, eff_bkg = dict(), dict()
    for var in input_vars:
        eff_sig[var], eff_bkg[var] = roc(signal[var], background[var], signal['weight'], background['weight'])
        pass

    plt.figure(figsize=(6,6))
    
    plt.plot(np.linspace(0, 1, 100 + 1, True), np.linspace(0, 1, 100 +1, True), color = 'gray', linestyle = '--')
    plt.fill_between(np.linspace(0, 1, 100 + 1, True), 
                     np.linspace(0, 1, 100 + 1, True), 
                     np.ones(100 + 1), color = 'black', alpha = 0.1)

    for var in input_vars:
        plt.plot(eff_sig[var], eff_bkg[var], label = r'%s' % displayName(var, latex = True))
        pass

    plt.xlabel(r'$\epsilon_{sig.}$')
    plt.ylabel(r'$\epsilon_{bkg.}$')
    plt.legend()
    if save: plt.savefig('ROC.pdf')
    plt.show()


    # Plot substructure profile(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    print "\nSubstructure profiles:"

    for var in input_vars:
        print "-- %s" % var
        bins  = np.linspace(0, 300, 50)
        bins += (bins[1] - bins[0]) / 2.

        for r in [(150, 200), (300, 400), (400, 500), (500, 700), (700, 1000), (1000, 2000), (0, 10000)]:
            profile = TProfile("profile_%s_%d_%d" % (var, r[0], r[1]), "", len(bins), 0, 300)
            msk = (background['pt'] >= r[0]) & (background['pt'] < r[1])
            fill_profile(profile, np.vstack((background['m'][msk], background[var][msk])).T, weights = background['weight'][msk])
            

            prof = np.zeros(len(bins))
            for ibin in range(len(bins)):
                prof[ibin] = profile.GetBinContent(ibin + 1)
                pass
            prof = np.ma.masked_array(prof, mask = (prof == 0))
            if r[0] == 0:
                plt.plot(bins, prof, color = 'black', alpha = 0.7, label = r'Incl. $p_{T}$')
            else:
                plt.scatter(bins, prof, label = r'$p_{T} \in [%d, %d]$ GeV' % (r[0], r[1]))
                pass
            pass

        plt.xlim([0, 300])
        plt.ylim([0, 4 if var == 'D2' else (1 if var == 'tau21' else (300. if var == 'm' else 1.))])
        plt.xlabel(displayNameUnit('m', latex = True))
        plt.ylabel(r'$\langle %s \rangle$' % displayName(var, latex = True).replace('$', ''))
        plt.legend()
        if save: plt.savefig('profile_%s.pdf' % var)
        plt.show()
        pass

    # ...

    return


# Main function call.
if __name__ == '__main__':
   main()
   pass
