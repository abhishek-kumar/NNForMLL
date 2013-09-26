#ifndef BR_SINGLELAYERNN_H
#define BR_SINGLELAYERNN_H

#include "singleLayerNN.h"

// A neural network with 1 hidden layer, and no direct connections between
// Input and Output layer (this model corresponds to the BN-MLL model in the paper)
class BRSingleLayerNN
{
public:

  // ctors. fileio contains data read from a dataset.
  BRSingleLayerNN(io & fileio, int numFeatures, int numHiddenUnits, int numTags);

  // Trains the model including finding the best regularization strength.
  // Cross validation is done with cvFolds folds to select the best C
  // We search for all values of C between
  // 2^(lowerLimit) <= C < 2^(UpperLimit)
  // in steps of stepSize
  void train(int lowerLimit = -12, int upperLimit = 13, int stepSize = 1, int cvFolds = 5);

  // Given a test instance (xtest), compare our prediction with ground truth (ytest) and
  // compute evaluation metric values
  error_t test(data_t xtest, data_t ytest);

  ~BRSingleLayerNN();

private:
  // Note: Declaration order is important for initialization
  int p,d,k;             // Dimensions of the parameters and gradient
  //floatnumber C;           // Regularization, lower => stronger regularization
  //floatnumber linearity;       // 1.0e-5;  // To speed up convergence
  //parameters * wopt;        // Parameters of the model
  //int counter;          // Keeps a count of LBFGS iterations done
  data_t &xtr, &ytr;        // training set, training labels
  int m;              // Number of training examples
  vector<data_t> ytr_tag;      // The label for a single tag
  vector<singleLayerNN *> baseModels;// The k base models
};

#endif
