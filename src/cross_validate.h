#ifndef CROSS_VALIDATE_H
#define CROSS_VALIDATE_H

#include "io.h"
#include "types.h"

// Functions that takes a dataset, dimensions of network,
// regularization parameter and return the computed loss.
typedef error_t (*callback_fn)(io&, dimensions, floatnumber);
typedef error_t (*callback_fn2)(io&, dimensions, floatnumber, floatnumber);

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

// Similar to FindBestC above, but this cross validates to find the best C2
// when C has already been found.
// Note that C refers to the regularization parameter for weights of
// edges input --> hidden --> output; whereas C2 refers to the regularization
// for weights of edges input --> output.
floatnumber FindBestC2(
    cv_params cv,
    data_t& xtrain, data_t& ytrain, dimensions dim, floatnumber C,
    callback_fn2 evaluate_fn,
    bool silent_mode);
#endif
