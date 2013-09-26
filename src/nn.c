#include "nn.h"

// log file.
FILE * log_file;

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

void mean_and_stdev(record_t numbers, floatnumber* mean, floatnumber* stdev) {
  floatnumber sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
  *mean = sum / float(numbers.size());
  floatnumber sqSum = 0.0;
  for (record_t::iterator itr = numbers.begin(); itr != numbers.end(); ++itr) {
    sqSum += (*itr - *mean)*(*itr - *mean);
  }
  *stdev = sqrt(sqSum / numbers.size());
}

error_t_struct::error_t_struct() :  nll(0.0), hl(0.0), sl(0.0), rl(0.0),
  nrl(0.0), oe(0.0), avprec(0.0) { }


void error_t_struct::print() {
  log("  Negative Log Likelihood: %0.4f", nll);
  log("  0/1 Subset Loss: %0.4f", sl);
  log("  Hamming Loss: %0.4f", hl);
  log("  Ranking Loss: %0.4f", rl);
  log("  Normalized Ranking Loss: %0.4f", nrl);
  log("  One Error: %0.4f", oe);
  log("  Average Precision: %0.4f", avprec);
}

void error_t_struct::print_stdout() {
  log_stdout("    Negative Log Likelihood: %0.4f", nll);
  log_stdout("    0/1 Subset Loss: %0.4f", sl);
  log_stdout("    Hamming Loss: %0.4f", hl);
  log_stdout("    Ranking Loss: %0.4f", rl);
  log_stdout("    Normalized Ranking Loss: %0.4f", nrl);
  log_stdout("    One Error: %0.4f", oe);
  log_stdout("    Average Precision: %0.4f", avprec);
}

void log(const string message, ...) {
  va_list args;
  va_start(args, message);
  fprintf(log_file, "INFO: ");
  vfprintf(log_file, message.c_str(), args);
  fprintf(log_file, "\n");
  va_end(args);
}

void log_stdout(const string message, ...) {
  va_list args;
  va_start(args, message);
  vprintf(message.c_str(), args);
  printf("\n");
  va_end(args);
}
