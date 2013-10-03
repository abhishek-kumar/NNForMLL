#ifndef SLN_MLL_H
#define SLN_MLL_H

#include "compatibility.h"
#include "types.h"

class io;
struct nn_dimensions;
typedef nn_dimensions dimensions;
struct sparameters;
typedef sparameters parameters;
struct error_t_struct;
typedef error_t_struct error_t;

// A neural network with 1 hidden layer, and direct connections between input
// and output layer, in addition to connections between input and hidden layer,
// and hidden and output layer. This model corresponds to the SLN_MLL model in
// the paper.
class SLN_MLL {
 public:
  // ctors.
  // fileio: data read from dataset file.
  // dim: dimensions of the neural network.
  // C: regularization weight for edges (input to hidden) and (hidden to output)
  // C2: regularization weight for edges (input to output).
  SLN_MLL(io& fileio, dimensions dim);
  SLN_MLL(
    data_t& xtrain, data_t& ytrain, dimensions dim,
    floatnumber C, floatnumber C2);

  // Trains the model given input data (xtrain) with p features, 
  // input labels (ytrain) with k tags. The neural network has d hidden units.
  // This function does NOT train the regularization strength, and 
  // instead expects it to be set in the member variable "C"
  void train();

  // similar to train(), but this method additionally calculates an optimal
  // value for C2. "C" is expected to be set correctly prior to call.
  // Cross validation is done with cvFolds folds to select the
  // best C. We search for all values of C between 
  //     2^(lowerLimit) <= C < 2^(UpperLimit)
  // in steps of stepSize. Train time is expected to be ~ 250 times the time
  // taken for train() if the number of folds is 10, and C2 is searched in the
  // range {2^-12, 2^-11, ..., 2^12}.
  void train(cv_params params_for_C2);

  // Similar to train(cv_params params_for_C), but this method trains
  // both C and C2. Train time is expected to be 
  // ~ 500 times the time taken for train() if the number of folds is 10;
  // and C and C2 are searched in the range {2^-12, 2^-11, ..., 2^12}
  void train(cv_params params_for_C, cv_params params_for_C2);

  // Given a test set (xtest), compare our predictions with the ground truth
  // (ytest) and return the calculated evaluation metric values.
  error_t test(data_t xtest, data_t ytest);

  ~SLN_MLL();  // Since we don't do inheritance, it is not virtual (yet).

  floatnumber getC()          { return C; }
  floatnumber getC2()         { return C2;}
  floatnumber getParameters() { return wopt;}


 protected:

  // Given an instance (x), forward propagate the neural network to obtain:
  //   y_hata_: activation vector for output layer.
  //   y_hat:   output (label estimates).
  //   ha:      activation vector for the hidden layer.
  //   h:       values computed at the hidden layer.
  void forwardPropagate(
    record_t const & x, parameters const & w,
    floatnumber *y_hata_, floatnumber *y_hat_,
    floatnumber *ha_, floatnumber *h_);

  // Compare predictions (y_hat) with ground truth (y) and determine loss values
  // negative log likelihood (nll) and hamming loss (hl).
  void calculateLosses(
    const floatnumber *y_hata, const floatnumber *y_hat, record_t const & y,
    parameters const & w, floatnumber & nll, floatnumber & hl);

  // Calculate the jacobian of the parameters of the neural network model.
  // Used by LBFGS for optimization.
  void calculateJacobian(
    record_t const & x, record_t const & y,
    const floatnumber *y_hata, const floatnumber *y_hat,
    const floatnumber *ha, const floatnumber *h,
    parameters const& w, parameters& jacobian);

  friend lbfgsfloatval_t Compatibility::evaluate(
    void* instance, const lbfgsfloatval_t* wv,
    lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step);


  friend int Compatibility::progress(
    void *instance, const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls);

 private:
  int m;                           // Number of training examples
  int p,d,k;                       // Dimensions of the parameters and gradient
  data_t xtr, ytr, xte, yte;       // training and test datasets
  parameters * wopt = 0;           // Parameters of the model
  floatnumber C;                   // Regularization, lower => stronger regularization
  floatnumber C2;                  // Regularization, lower => stronger regularization
  floatnumber linearity = 0.0;     // 1.0e-5;  // To speed up convergence
  int counter=0;                   // Keeps a count of LBFGS iterations done
}

#endif
