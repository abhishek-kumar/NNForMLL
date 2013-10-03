#ifndef BR_MLL_H
#define BR_MLL_H

#include "BN_MLL.h"
#include "types.h"
#include <vector>

using namespace std;

// A neural network with no hidden layer and direct connections between
// input and output layer. This model is a binary relevance model and is
// implemented here as a baseline for comparison with BN_MLL and SLN_MLL.
class BR_MLL {
 public:

  // ctors.
  // fileio contains data read from a dataset.
  // dim contains dimensions of the network
  BR_MLL(io & fileio, dimensions dim);

  // Trains the model including finding the best regularization strength.
  // Cross validation is done with cvFolds folds to select the best C
  // We search for all values of C between
  // 2^(lowerLimit) <= C < 2^(UpperLimit)
  // in steps of stepSize
  void Train(cv_params cv);

  // Given a test instance (xtest), compare our prediction with ground truth (ytest) and
  // compute evaluation metric values
  error_t Test(data_t xtest, data_t ytest);

  ~BR_MLL();

private:
  // Note: Declaration order is important for initialization
  int p,d,k;                   // Dimensions of the network and gradient
  data_t &xtr, &ytr;           // training set, training labels
  int m;                       // Number of training examples
  vector<data_t> ytr_tag;      // The label for a single tag
  vector<BN_MLL*> baseModels;  // The k base models
};

#endif
