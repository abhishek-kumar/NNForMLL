#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <functional>
#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <sstream>
#include <string>
#include <time.h>
#include "lbfgs.h"

using namespace std;

// float datatype used in the project
typedef lbfgsfloatval_t floatnumber;

// A list of float values is a record
typedef vector<floatnumber> record_t;

// A dataset is a vector of records
typedef vector<record_t> data_t;

// log file.
extern FILE * log_file;

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
