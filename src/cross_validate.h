#ifndef CROSS_VALIDATE_H
#define CROSS_VALIDATE_H

#include "io.h"
#include "types.h"

typedef error_t (*callback_fn)(io&, dimensions, floatnumber);

// Random number generator
int rand_int(int n);

// Randomly reshuffle the dataset for cross validation
void randomShuffle(int *array, int n);

// Calculate mean and standard deviation.
void mean_and_stdev(record_t numbers, floatnumber* mean, floatnumber* stdev);

// Cross validates and finds the best C for the given dataset.
// xtrain: training set features
// ytrain: training set labels
// error_t (*evaluate)(...) Function pointer to an evaluator that calculates 
// loss over a given subset of the dataset.
floatnumber FindBestC(
    cv_params cv,
    data_t& xtrain, data_t& ytrain, dimensions dim,
    callback_fn evaluate_fn,
    bool silent_mode);

#endif
