# -*- coding: utf-8 -*-

""" ...

@file:   layers.py
@author: Andreas Søgaard
@date:   20 February 2017
@email:  andreas.sogaard@cern.ch
"""

# Numpy include(s)
import numpy as np

# Keras include(s)
from keras import backend as K_
from keras.engine.topology import Layer


class Posterior(Layer):
    """ Custom layer, modelling the posterior probability distribution for the jet mass using a gaussian mixture model (GMM) """

    def __init__(self, output_dim, num_components, **kwargs):
        self.output_dim = output_dim
        self.num_components = num_components
        super(Posterior, self).__init__(**kwargs)

    def call(self, x, mask=None):
        """ Main call-method of the layer. 

        The GMM needs to be implemented (1) within this method and (2) using Keras backend functions in order for the error back-propagation to work properly 
        """

        # Unpack list of inputs
        coeffs, means, widths, masses = x

        # Compute the pdf from the GMM
        pdf = coeffs[:,0] * K_.exp( - K_.square(masses[:,0] - means[:,0]) / 2. / K_.square(widths[:,0])) / K_.sqrt( 2. * K_.square(widths[:,0]) * np.pi)
        for c in range(1, self.num_components):
            pdf += coeffs[:,c] * K_.exp( - K_.square(masses[:,0] - means[:,c]) / 2. / K_.square(widths[:,c])) / K_.sqrt( 2. * K_.square(widths[:,c]) * np.pi)
            pass

        return K_.reshape(pdf, (pdf.shape[0], 1))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    pass  
