CONTENTS
--------

1. Introduction
2. Prerequisites
3. Compiling and Running
4. Documentation
5. License
6. Contact


1. Introduction
--------------------

This is an implementation of single hidden layer neural network (NN) models for multi-label learning.

Multilabel learning is an extension of standard binary classification where the goal is to predict a set of labels (we call an individual label a tag) for each input example. The recent probabilistic classifier chain (PCC) method learns a series of probabilistic models that capture tag correlations. Using a neural network with a hidden layer, instead of connections between output nodes, brings several advantages that include tractable test-time inference and removing the need to select a fixed tag ordering. Moreover, the hidden units capture nonlinear latent structure, which improves classification accuracy, and allows correlations between tags to be visualized explicitly. 

Compared to previous neural network methods for multilabel learning, this implementatation includes several design decisions that lead to a notable decrease in training time and an increase in accuracy. Empirical results show that the new method outperforms existing MLL methods on benchmark datasets. 
Further details of the NN models can be found [here](http://is.gd/NNForMLL).


2. Prerequisites
---------------------
This implementation uses L-BFGS in concert with backpropagation for training the model parameters. 
The library liblbfgs must be installed prior to compilation. This can be obtained from [liblbfgs](http://www.chokkan.org/software/liblbfgs/).

Moreover, in order to statically link the liblbfgs library to the neural network code, the tool libtool (provided as part of the libfgs download) must be available at 

     ../liblbfgs-1.10/libtool

relative to the current directory (containing this README file).


3. Compiling and Running
-----------------------------------
To compile the code, execute the compile.sh bash script

     $ ./compile.sh

Upon completion, the compiled binary will be available in

     ./bin/mll

In order to run the file from the command line, enter:

     $ ./mll <method> <train file> <test file> <p> <h> <k> [singleLayerC]

<dl>
<dt>method:     </dt>
<dd>Either 1,2 or 3 for BN-MLL, SLN-MLL and BR-NN respectively (as described in the paper, http://is.gd/NNForMLL).</dd>

<dt>train file: </dt> 
<dd>Enter the full path to the training file. The file should be a csv file with only numbers, such that each line represents a document, the first p columns indicate values for the p features, the next k columns indicate values (0 or 1) for the k labels.</th></tr>

<dt>test file:  </dt> 
<dd>Enter the full path to the test file. The format should be similar to the training file format.</dd>

<dt>p:          </dt> 
<dd>The dimensionality of the input space, or the number of features.</dd>

<dt>d:          </dt> 
<dd>The number of hidden layers to use in the neural network model.</dd>

<dt>k:          </dt> 
<dd>The dimensionality of the output space, or the number of labels.</dd>

<dt>singleLayerC:</dt>
<dd>If method #2 is being used, then the regularization weight for the component of the model corresponding to method #1 can be specified separately.</dd>
</dl>

4. Documentation
----------------
The details of the models and optimizations done are provided in this [paper](http://is.gd/NNForMLL).


5. License
----------
This code is available under the [Apache License, version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).
Please let me know if you find this work useful and are using it in any of your projects.


6. Contact
----------

Email:

     abhishek [at] ucsd [dot] edu

[Website](http://cseweb.ucsd.edu/~abk004/):

     http://cseweb.ucsd.edu/~abk004/
