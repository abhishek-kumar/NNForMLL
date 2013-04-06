#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <functional>
#include <vector>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <time.h>
#include "lbfgs.h"

using namespace std;
typedef lbfgsfloatval_t floatnumber;
typedef vector<floatnumber> record_t;
typedef vector<record_t> data_t;

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
} error_t;

int rand_int(int n);

void randomShuffle(int *array, int n);

void meanAndStdev(record_t numbers, floatnumber& mean, floatnumber& stdev);

#endif