#include "nn.h"

int rand_int(int n) {
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;

  do {
    rnd = rand();
  } while (rnd >= limit);
  return rnd % n;
}

void randomShuffle(int *array, int n) {
  int i, j, tmp;

  for (i = n - 1; i > 0; i--) {
    j = rand_int(i + 1);
    tmp = array[j];
    array[j] = array[i];
    array[i] = tmp;
  }
}

void meanAndStdev(record_t numbers, floatnumber& mean, floatnumber& stdev) {
	floatnumber sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
	mean = sum / float(numbers.size());
	floatnumber sqSum = 0.0;
	for (record_t::iterator itr = numbers.begin(); itr != numbers.end(); ++itr) {
		sqSum += (*itr - mean)*(*itr - mean);
	}
	stdev = sqrt(sqSum / numbers.size()); 
}

error_t_struct::error_t_struct() :  nll(0.0), hl(0.0), sl(0.0), rl(0.0), 
  nrl(0.0), oe(0.0), avprec(0.0) { }


void error_t_struct::print() { 
  cout << "Evaluation error values:" << endl;
  cout << "\tNegative Log Likelihood: " << nll << endl;
  cout << "\t0/1 Subset Loss: " << sl << endl;
  cout << "\tHamming Loss: " << hl << endl;
  cout << "\tRanking Loss: " << rl << endl;
  cout << "\tNormalized Ranking Loss: " << nrl << endl;
  cout << "\tOne Error: " << oe << endl;
  cout << "\tAverage Precision: " << avprec << endl;
}