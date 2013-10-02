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


#endif
