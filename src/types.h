#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <vector>
// types used by lbfgs library
#include "lbfgs.h"

// float datatype used in the project
typedef lbfgsfloatval_t floatnumber;

// A list of float values is a record
typedef std::vector<floatnumber> record_t;

// A dataset is a vector of records
typedef std::vector<record_t> data_t;

// Parameters for cross validation, objective is to find optimal C
typedef struct cross_validation_params_t {
  int lower_bound_power_of_2;  // lower bound for C: 2^lower_bound_power_or_2
  int upper_bound_power_of_2;  // upper bound for C: 2^upper_bound_power_or_2
  int step_size_power_of_2;    // C_hat incremented by 2^step_size_power_of_2
  int num_folds;               // Number of folds to use for cross validation
  
  cross_validation_params_t(int lower, int upper, int step, int folds) :
     lower_bound_power_of_2(lower), upper_bound_power_of_2(upper),
     step_size_power_of_2(step), num_folds(folds) { }
} cv_params;

// struct for holding neural network dimensions
typedef struct nn_dimensions {
  int p;  // Number of input nodes (#Features).
  int h;  // Number of hidden nodes.
  int k;  // Number of output nodes.
  nn_dimensions(int input_nodes, int hidden_nodes, int output_nodes) {
    p = input_nodes;
    h = hidden_nodes;
    k = output_nodes;
  }
} dimensions;

// struct for holding loss values
typedef struct error_t_struct {
  floatnumber nll;
  floatnumber hl;
  floatnumber sl;
  floatnumber rl;
  floatnumber nrl;
  floatnumber oe;
  floatnumber avprec;
  error_t_struct() : nll(0.0), hl(0.0), sl(0.0), rl(0.0),
      nrl(0.0), oe(0.0), avprec(0.0) { };
} error_t;

#endif
