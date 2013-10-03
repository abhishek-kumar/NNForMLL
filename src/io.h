#ifndef IO_H_
#define IO_H_

#include <string>
#include "types.h"

// Class that holds a dataset, including training set, test set,
// training labels, and test labels.
class io {
 public:
  data_t xtr;  // Training set features.
  data_t xte;  // Test set features.
  data_t ytr;  // Training set labels.
  data_t yte;  // Test set labels.

  floatnumber * xtrmean;  // Mean value of each feature.

  std::string trainFileName;  // Full path to the training file.
  std::string testFileName;  // Full path to the test file.

  // Read training data with p features and k tags in its label, from filename.
  void readTrainingData(char *filename, int p, int k);

  // Read test data with p features and k tags in its label, from filename.
  void readTestData(char *filename, int p, int k); 

  // Normalize the dataset to have 0-mean and 1-variance.
  void normalize(int p);
};

#endif
