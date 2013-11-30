CONTENTS
--------

1. Introduction
2. How to Install and Compile
3. How to Run
4. Documentation
5. License
6. Contact


1. Introduction
--------------------

This is an implementation of single hidden layer neural network (NN) models for multi-label learning.

Multi-label learning is an extension of standard binary classification where the goal is to predict a set of labels for each input example. This project implements three models (i.e. BN-MLL, SLN-MLL and BR-MLL) that learn to predict labels for unseen documents, using a neural network with a hidden layer. These hidden units capture nonlinear latent structure, which improves classification accuracy, and allows correlations between tags to be visualized explicitly.

Compared to previous neural network methods for multi-label learning, this implementatation includes several design decisions that lead to a notable decrease in training time and an increase in accuracy. Empirical results show that the new method outperforms existing methods on benchmark datasets.
Further details of the models can be found in a draft manuscript [here](http://is.gd/NNForMLL).


2. How to Install and Compile
---------------------
This implementation uses L-BFGS in concert with backpropagation for training the model parameters.
To install, checkout this repository and run `make`:


     $ git clone https://github.com/abhishek-kumar/NNForMLL
     $ cd NNForMLL/
     $ make

If lib-lbfgs is not installed on your system, the make file will try to install it automatically (you might need to provide an administrator password).
Upon completion, the compiled binary will be available in

     ./bin/mll

Notes:
  * If the make command fails, you can try installing explicitly using the `install.sh` script.
  * The make file assumes that your system library paths are set to the default `'/usr/local/lib'`. If not, the libraries may be installed elsewhere. If this happens, the output of `install.sh` should tell you where the libraries are installed. This path should then be added to the Makefile in place of `'/usr/local/lib'`.
  * The library liblbfgs has been obtained from here: [liblbfgs](http://www.chokkan.org/software/liblbfgs/).
  * If you don't have `make` installed, then install it on Ubuntu using `sudo apt-get install make`. On a mac, open `XCode` (create a new empty project), go to the XCode menu, click Preferences. Go to the downloads tab and then to components. Under `Command line tools`, click Install.
  * The commands in this file have been tested on a PC running Ubuntu. They should work correctly on a mac, on other linux platforms and on windows with Cygwin. Please let me know if something doesn't work right on your platform.


3. How to Run
-----------------------------------
To compile the code, run `make` from the `NNForMLL` directory.

     $ make

To run the `BN-MLL` model on the provided benchmark datasets, simply run the `run.sh` script file.

     $ ./run.sh

In order to run program on your own datasets, with your own parameters, run using the following format:

     $ ./bin/mll <method> <train file> <test file> <p> <h> <k> [singleLayerC]

<dl>
<dt>method:     </dt>
<dd>Either 1,2 or 3 for BN-MLL, SLN-MLL and BR-NN respectively (as described in the paper, http://is.gd/NNForMLL).</dd>

<dt>train file: </dt>
<dd>Enter the full path to the training file. The file should be a csv file with only numbers, such that each line represents a document, the first p columns indicate values for the p features, the next k columns indicate values (0 or 1) for the k labels. More details on the data format here: https://github.com/abhishek-kumar/NNForMLL/tree/master/data. </th></tr>

<dt>test file:  </dt>
<dd>Enter the full path to the test file. The format should be similar to the training file format.</dd>

<dt>p:          </dt>
<dd>The dimensionality of the input space, or the number of features.</dd>

<dt>h:          </dt>
<dd>The number of hidden layers to use in the neural network model.</dd>

<dt>k:          </dt>
<dd>The dimensionality of the output space, or the number of labels.</dd>

<dt>singleLayerC:</dt>
<dd>Regularization weight for the bottom edges (edges between input layer and hidden layer). This argument is optional. If not provided, the program will learn using cross validation (this will take ~15x longer to run)</dd>
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

     abhishek.kumar.ak [at] gmail [dot] com

[Website](http://abhishek-kumar.com):

     http://abhishek-kumar.com
