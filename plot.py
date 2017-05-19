#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script for plotting the results of ...

@file    plot.py
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

# -- Scipy
from scipy.signal import savgol_filter

# Local include(s)
from utils import *
from models import *


# Main function definition.
def main ():

    # Set pyplot style
    plt.style.use('ggplot')

    # Whether to save plots
    save = True


    # Get data
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    substructure_vars  = ['jet_tau21', 'jet_D2', 'jet_m']
    decorrelation_vars = ['jet_m'] 
    X, Y, W, P, signal, background, names = getData(decorrelation_vars)

    msk_sig = (Y == 1.)


    # Load pre-trained classifier
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Load existing classifier model from file
    classifier = load_model('classifier.h5')

    # Add neural network classifier output, without adversarial training
    signal    ['NN'] = classifier.predict(X[ msk_sig], batch_size=1024)
    background['NN'] = classifier.predict(X[~msk_sig], batch_size=1024)

    # Scale to mean 0.5 and sensible range
    #scaler = preprocessing.StandardScaler().fit(background['NN'].reshape(-1,1))
    #signal    ['NN'] = (scaler.transform(signal    ['NN'].reshape(-1,1)) / 4. + 0.5).reshape(signal    ['jet_m'].shape)
    #background['NN'] = (scaler.transform(background['NN'].reshape(-1,1)) / 4. + 0.5).reshape(background['jet_m'].shape)

    wmean, wstd = weighted_avg_and_std(background['NN'].ravel(), background['weight'].ravel())
    signal    ['NN'] = ((signal    ['NN'] - wmean) / wstd / 8. + 0.5).reshape(signal    ['jet_m'].shape)
    background['NN'] = ((background['NN'] - wmean) / wstd / 8. + 0.5).reshape(background['jet_m'].shape)

    # Remember to use 'NN' in comparisons later
    substructure_vars += ['NN']


    # Load adversarially trained models
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Combined
    adversarial = adversarial_model(classifier, [(64, 'tanh')] * 2, 1, P.shape[1])
    load_checkpoint(adversarial)

    # Add neural network classifier output, without adversarial training
    signal    ['ANN'] = classifier.predict(X[ msk_sig], batch_size=1024)
    background['ANN'] = classifier.predict(X[~msk_sig], batch_size=1024)

    # Scale to mean 0.5 and sensible range
    #scaler = preprocessing.StandardScaler().fit(background['ANN'].reshape(-1,1))
    #signal    ['ANN'] = (scaler.transform(signal    ['ANN'].reshape(-1,1)) / 4. + 0.5).reshape(signal    ['jet_m'].shape)
    #background['ANN'] = (scaler.transform(background['ANN'].reshape(-1,1)) / 4. + 0.5).reshape(background['jet_m'].shape)

    wmean, wstd = weighted_avg_and_std(background['ANN'].ravel(), background['weight'].ravel())
    signal    ['ANN'] = ((signal    ['ANN'] - wmean) / wstd / 8. + 0.5).reshape(signal    ['jet_m'].shape)
    background['ANN'] = ((background['ANN'] - wmean) / wstd / 8. + 0.5).reshape(background['jet_m'].shape)

    # Remember to use 'ANN' in comparisons later
    substructure_vars += ['ANN']



    # Weights sparsity
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    if False:

        print "\nWeights sparsity:"

        bins = np.linspace(0, 1, 100.)

        for ilayer, layer in enumerate(classifier.layers):

            # If layer doesn't have any weights (e.g. input or output layer), continue
            if len(layer.get_weights()) == 0: continue


            weights  = np.sort(np.abs(layer.get_weights()[0]).ravel())
            weights /= weights[-1]
            bins     = np.linspace(0, 1, weights.size, endpoint=True)

            plt.plot(bins, weights, alpha=0.4, label='Layer %d' % (ilayer + 1))
            pass

        plt.grid()
        plt.legend()
        plt.show()

        pass


    # Percentile contours
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    if False:

        print "\nPercentile contours:"

        profile_var = 'jet_m' # Variable against which to compute and show profile

        for var in substructure_vars:
            print "-- %s" % var

            binsx = np.logspace(1, 2, 50 + 1, endpoint=True) * 3. #np.linspace( 0., 300.,  60 + 1, endpoint = True)
            binsy = np.linspace(-50.,  500., 10000 + 1, endpoint=True)

            H, _, _ = np.histogram2d(background[profile_var], background[var], [binsx, binsy], weights=background['weight'])
            H = np.array(H).T
            
            num_contours = 15

            binsx = (binsx[:-1] + binsx[1:]) * 0.5
            binsy = (binsy[:-1] + binsy[1:]) * 0.5

            contours = np.zeros((len(binsx), num_contours))

            for bin in range(len(binsx)):
                for c in range(num_contours):
                    eff = (c + 0.5) / float(num_contours)
                    value = wpercentile(binsy, eff, weights=H[:,bin])
                    if value is None: value = np.nan
                    contours[bin,c] = value
                    pass
                pass


            if num_contours % 2: # odd
                linewidths = [1] * (num_contours//2) + [3] + [1] * (num_contours//2)
            else:
                linewidths = [1] *  num_contours
                pass

            for c in range(num_contours):
                plt.plot(binsx, contours[:,c], linewidth=linewidths[c], color='red')
                pass
            plt.xlabel(r'%s' % displayNameUnit(profile_var, latex=True))
            plt.ylabel(r'%s' % displayNameUnit(var,         latex=True))
            plt.xlim([0, 300])
            if var.endswith('NN'):
                plt.ylim([0, 1])
                pass
            if save: plt.savefig('percentile_countours_%s.pdf' % var)
            plt.show()


            pass

        pass

    

    # Cost log(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    if True:

        print "\nCost log:"


        # Plot cost log
        colors = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]

        costlog = np.loadtxt('cost.log', delimiter=',')
        names = ['loss']

        for i, (key,l) in enumerate(zip(names,costlog.tolist())):
            name = key.replace('loss', '').replace('_', '')
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
        c_log, d_log = list(), list()
        with open('cost.log', 'r') as f:
            for line in f:
                fields = line.split(',')
                d_log.append(float(fields[0]))
                c_log.append(float(fields[1]))
                pass
            pass

        plt.plot(c_log, label='Classifier',    alpha=0.4)
        plt.plot(d_log, label='Discriminator', alpha=0.4)
        plt.plot(savgol_filter(c_log,201,3), label='Classifier (smooth)',)
        plt.plot(savgol_filter(d_log,201,3), label='Discriminator (smooth)',)
        plt.legend()
        plt.show()
        '''

        pass



    # Plot 1D distribution(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    if True:

        print "\n1D distributions:"

        h_sig = dict()
        h_bkg = dict()
        for var in substructure_vars:
            print "-- %s" % var
            bins = np.linspace(0, 4.0 if var == 'jet_D2' else (300. if var == 'jet_m' else 1.0), 100 + 1, True)
            h_bkg[var] = plt.hist(background[var], bins, weights=background['weight'],      alpha=0.6, label='Background')
            h_sig[var] = plt.hist(signal    [var], bins, weights=signal    ['weight'] * 20, alpha=0.6, label='Signal (x 20)')
            plt.xlim([bins[0], bins[-1]])
            plt.xlabel(r'%s' % displayNameUnit(var, latex=True))
            plt.ylabel(r'Events [fb]')
            plt.legend()
            if save: plt.savefig('distrib_%s.pdf' % var)
            plt.show()
            pass

        pass


    # Plot ROC curve(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    if True:

        print "\nROC curves:"

        eff_sig, eff_bkg = dict(), dict()
        for var in substructure_vars:
            eff_sig[var], eff_bkg[var] = roc(signal[var], background[var], signal['weight'], background['weight'])
            pass

        plt.figure(figsize=(6,6))
        
        plt.plot(np.linspace(0, 1, 100 + 1, True), np.linspace(0, 1, 100 +1, True), color='gray', linestyle='--')
        plt.fill_between(np.linspace(0, 1, 100 + 1, True), 
                         np.linspace(0, 1, 100 + 1, True), 
                         np.ones(100 + 1), color='black', alpha=0.1)

        for var in substructure_vars:
            plt.plot(eff_sig[var], eff_bkg[var], label=r'%s' % displayName(var, latex=True))
            pass

        plt.xlabel(r'$\epsilon_{sig.}$')
        plt.ylabel(r'$\epsilon_{bkg.}$')
        plt.legend()
        if save: plt.savefig('ROC.pdf')
        plt.show()

        pass


    # Plot substructure profile(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    if True:

        print "\nSubstructure profiles:"

        profile_var = 'jet_m'
        for var in substructure_vars:
            print "-- %s" % var
            bins  = np.linspace(0, 300, 50)
            bins += (bins[1] - bins[0]) / 2.

            #for r in [(150, 200), (300, 400), (400, 500), (500, 700), (700, 1000), (1000, 2000), (0, 10000)]:
            for r in [(150, 10000), (200, 10000), (250, 10000), (300, 10000), (0, 10000)]:
                if profile_var == 'jet_m':
                    profile = TProfile("profile_%s_%d_%d" % (var, r[0], r[1]), "", len(bins), 0, 300)
                else:
                    profile = TProfile("profile_%s_%d_%d" % (var, r[0], r[1]), "", len(bins), -5, -1)
                    pass

                msk = (background['jet_pt'] >= r[0]) & (background['jet_pt'] < r[1])
                fill_profile(profile, np.vstack((background[profile_var][msk], background[var][msk])).T, weights=background['weight'][msk])
                

                prof = np.zeros(len(bins))
                for ibin in range(len(bins)):
                    prof[ibin] = profile.GetBinContent(ibin + 1)
                    pass
                prof = np.ma.masked_array(prof, mask=(prof == 0))
                if r[0] == 0:
                    plt.plot(bins, prof, color='black', alpha=0.7, label=r'Incl. $p_{T}$')
                elif r[1] >= 10000:
                    plt.scatter(bins, prof, label=r'$p_{T} > %d$ GeV' % r[0])
                else:
                    plt.scatter(bins, prof, label=r'$p_{T} \in [%d, %d]$ GeV' % (r[0], r[1]))
                    pass
                pass

            plt.xlim([profile.GetXaxis().GetXmin(), profile.GetXaxis().GetXmax()])
            plt.ylim([0, 4 if var == 'jet_D2' else (1 if var == 'jet_tau21' else (300. if var == 'jet_m' else 1.))])
            plt.xlabel(displayNameUnit(profile_var, latex=True))
            plt.ylabel(r'$\langle %s \rangle$' % displayName(var, latex=True).replace('$', ''))
            plt.legend()
            if save: plt.savefig('profile_%s.pdf' % var)
            plt.show()
            pass

            pass


    # Plot reverse substructure profile(s)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    if True:

        print "\nReverse substructure profiles:"

        profile_var = 'jet_m'
        for var in substructure_vars:
            print "-- %s" % var
            if   var == 'jet_m':
                bins  = np.linspace(0, 300, 50)
            elif var == 'jet_D2':
                bins  = np.linspace(0, 5, 50)
            else:
                bins  = np.linspace(0, 1, 50)
                pass

            bins += (bins[1] - bins[0]) / 2.

            for r in [(150, 200), (300, 400), (400, 500), (500, 700), (700, 1000), (1000, 2000), (0, 10000)]:
                if profile_var == 'jet_m':
                    profile = TProfile("profile_%s_%d_%d" % (var, r[0], r[1]), "", len(bins), bins[0], bins[-1])
                else:
                    profile = TProfile("profile_%s_%d_%d" % (var, r[0], r[1]), "", len(bins), bins[0], bins[-1])
                    pass

                msk = (background['jet_pt'] >= r[0]) & (background['jet_pt'] < r[1])
                fill_profile(profile, np.vstack((background[var][msk], background[profile_var][msk])).T, weights=background['weight'][msk])
                

                prof = np.zeros(len(bins))
                for ibin in range(len(bins)):
                    prof[ibin] = profile.GetBinContent(ibin + 1)
                    pass
                prof = np.ma.masked_array(prof, mask=(prof == 0))
                if r[0] == 0:
                    plt.plot(bins, prof, color='black', alpha=0.7, label=r'Incl. $p_{T}$')
                else:
                    plt.scatter(bins, prof, label=r'$p_{T} \in [%d, %d]$ GeV' % (r[0], r[1]))
                    pass
                pass

            plt.xlim([profile.GetXaxis().GetXmin(), profile.GetXaxis().GetXmax()])
            #plt.ylim([0, 4 if var == 'jet_D2' else (1 if var == 'jet_tau21' else (300. if var == 'jet_m' else 1.))])
            plt.ylim([0, 300.])
            plt.xlabel(displayName(var, latex=True))
            plt.ylabel(r'$\langle %s \rangle$' % displayName(profile_var, latex=True).replace('$', ''))
            plt.legend()
            if save: plt.savefig('reverse_profile_%s.pdf' % var)
            plt.show()
            pass

        pass

    # ...

    return


# Main function call.
if __name__ == '__main__':
   main()
   pass
