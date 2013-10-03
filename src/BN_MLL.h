#ifndef BN_MLL_H
#define BN_MLL_H

#include "types.h"
#include "compatibility.h"

class io;
struct nn_dimensions;
typedef nn_dimensions dimensions;
struct sparameters;
typedef sparameters parameters;
struct error_t_struct;
typedef error_t_struct error_t;

// A neural network with 1 hidden layer, and no direct connections between 
// Input and Output layer (this model corresponds to the BN-MLL model in the paper) 
class BN_MLL {
 public:
  // ctors. 
  // fileio: contains data read from a dataset.
  // dim: dimensions of the neural network.
  // C: Regularization parameter.
  BN_MLL(io& fileio, dimensions dim);
  BN_MLL(data_t& xtrain, data_t& ytrain, dimensions dim, floatnumber C_);

  // Trains the model given input data (xtrain) with p features, 
  // input labels (ytrain) with k tags. The neural network has d hidden units.
  // This function does NOT train the regularization strength, and 
  // instead expects it to be set in the member variable "C"
  void Train();

  // similar to train(), but this method additionally calculates an optimal
  // value for C. Cross validation is done with cvFolds folds to select the
  // best C. We search for all values of C between 
  //     2^(lowerLimit) <= C < 2^(UpperLimit)
  // in steps of stepSize
  void Train(cv_params cv);

  // Given a test instance (xtest), compare our prediction with ground truth (ytest) and
  // compute evaluation metric values
  error_t Test(data_t xtest, data_t ytest);

  ~BN_MLL();

  floatnumber  getRegularizationStrength() { return C; }
  parameters*  getParameters()       { return wopt; }

 protected:
  // Calculate the Jacobian (derivative) of the NN model. 
  // Useful for LBFGS training. 
  // Function parameters are the same as for fit().
  void calculateJacobian(
    record_t const & x, record_t const & y, const floatnumber *y_hata, 
    const floatnumber *y_hat, const floatnumber *ha,
    const floatnumber *h, parameters const & w, parameters & jacobian);

  // Given a new data instance (x), forward propagate the NN. 
  // The objective is to compute the outputs:
  //   y_hata_ (activation vector for y_hat_),
  //   y_hat_
  //   ha = activation vector for the hidden layer,
  //   h = hidden layer
  // TODO(abhishek): Clean up the arguments and wrap activation vectors
  // into a struct to be passed around.
  void forwardPropagate(
    record_t const& x, parameters const& w, floatnumber* y_hata_, 
    floatnumber* y_hat_, floatnumber* ha_, floatnumber* h_);

  // After predictions have been obtained for a test example (y_hata, y_hat), compare with 
  // ground truth (y) and determine loss values (nll, hl = Negative log likelihood, Hamming loss).
  // This is used by the LBFGS calculations and so is separate from the test methods above.
  void calculateLosses(const floatnumber* y_hata, const floatnumber* y_hat, 
    record_t const & y, parameters const & w, floatnumber& nll, floatnumber& hl);

  // LBFGS called routines need full access to this class.
  friend lbfgsfloatval_t Compatibility::evaluate(void *instance,
                  const lbfgsfloatval_t *wv,
                  lbfgsfloatval_t *g,
                  const int n,
                  const lbfgsfloatval_t step);

  friend int Compatibility::progress(void *instance,
              const lbfgsfloatval_t *x, 
              const lbfgsfloatval_t *g, 
              const lbfgsfloatval_t fx, 
              const lbfgsfloatval_t xnorm,
              const lbfgsfloatval_t gnorm,
              const lbfgsfloatval_t step,
              int n,
              int k,
              int ls);

  friend class BR_MLL;

private:
  // Note: Declaration order is important for initialization
  int m;                  // Number of training examples
  int p,d,k;              // Dimensions of the parameters and gradient
  data_t &xtr, &ytr;      // training set, training labels
  parameters * wopt;      // Parameters of the model
  floatnumber C;          // Regularization, lower => stronger regularization
  floatnumber linearity;  // 1.0e-5;  // To speed up convergence
  int counter;            // Keeps a count of LBFGS iterations done
};

#endif

