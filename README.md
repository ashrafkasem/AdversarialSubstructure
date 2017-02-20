# AdversarialSubstructure

This package allows for the combination of several jet substructure variables into a single tagger which is designed to be pivotal with respect to mass of the jet, i.e. independent hereof, using an adversarial neural network (ANN). This approach is inspired by [1].

The ANN consists of a classifier and a discriminator:

* __classifier__: trained to perform optimal separation of signal and background, based on the available high-level jet substructure features;
* __discriminator__: trained to construct a posterior probability distribution for the jet mass, based on the classifier output.

In the adversarial training, this means that the discriminator tries to "guess" the jet mass from the classifier output, and the classifier tries to makes this task as hard as possible, while retaining as much separating power as possible.


## Contents
This package contains the following files:

* [models.py](models.py), containing the definition of the classifier-, discriminator-, and the combined adverarial neural network model;
* [layers.py](layers.py), containing the custom class for the posterior probability layer of the discriminator model;
* [train.py](train.py), which performs the training of the ANN and stores the weights of the trained classifier and discriminator models to file. Use as
```
python train.py path/to/data/*.root
```
* [plotting.py](plotting.py), which produces a few plots based on the output of [train.py](train.py). Use as
```
python plotting.py path/to/data/*.root
```
* [utils.py](utils.py), which contains a few common utility functions, including `getData` which reads training data from ROOT files. Consult for details about the assumed `TTree` structure.


## Dependencies

This packages relies on `Keras` for implementing the neural networks; `numpy` for handling and manipulating data; `scikit-learn` for data preprocessing; `matplotlib` for plotting; and `ROOT` for storing the data to be read and trained on. Finally the scripts in [snippets](https://github.com/asogaard/snippets) are used for reading in data.


## References

[1] G. Louppe, M. Kagan, and K.Cranmer. _Learning to Pivot with Adversarial Networks_. [arXiv:1611.01046](https://arxiv.org/abs/1611.01046)