#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <functional>
#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <string>
#include <time.h>
#include "types.h"

using namespace std;

// log file.
extern FILE * log_file;

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
  error_t_struct();
  void print();
  void print_stdout();
} error_t;

// Random number generator
int rand_int(int n);

// Randomly reshuffle the dataset for cross validation
void randomShuffle(int *array, int n);

// Calculate mean and standard deviation.
void mean_and_stdev(record_t numbers, floatnumber* mean, floatnumber* stdev);

// Logging functions
void log(const string message, ...);
void log_stdout(const string message, ...);

#endif
