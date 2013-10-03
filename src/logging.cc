#include "logging.h"

#include <fstream>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string>

using namespace std;

// log file.
FILE* log_file;

void Log(const string message, ...) {
  va_list args;
  va_start(args, message);
  fprintf(log_file, "INFO: ");
  vfprintf(log_file, message.c_str(), args);
  fprintf(log_file, "\n");
  va_end(args);
  fflush(log_file);
}

void Log_stdout(const string message, ...) {
  va_list args;
  va_start(args, message);
  vprintf(message.c_str(), args);
  printf("\n");
  va_end(args);
}

void Log(error_t& losses) {
  Log("  Negative Log Likelihood: %0.4f", losses.nll);
  Log("  0/1 Subset Loss: %0.4f", losses.sl);
  Log("  Hamming Loss: %0.4f", losses.hl);
  Log("  Ranking Loss: %0.4f", losses.rl);
  Log("  Normalized Ranking Loss: %0.4f", losses.nrl);
  Log("  One Error: %0.4f", losses.oe);
  Log("  Average Precision: %0.4f", losses.avprec);
}

void Log_stdout(error_t& losses) {
  Log_stdout("    Negative Log Likelihood: %0.4f", losses.nll);
  Log_stdout("    0/1 Subset Loss: %0.4f", losses.sl);
  Log_stdout("    Hamming Loss: %0.4f", losses.hl);
  Log_stdout("    Ranking Loss: %0.4f", losses.rl);
  Log_stdout("    Normalized Ranking Loss: %0.4f", losses.nrl);
  Log_stdout("    One Error: %0.4f", losses.oe);
  Log_stdout("    Average Precision: %0.4f", losses.avprec);
}

const string get_current_datetime() {
  time_t now = time(0);
  struct tm  tstruct;
  char current_time[80];
  tstruct = *localtime(&now);
  strftime(current_time, sizeof(current_time), "%Y-%m-%d  %X", &tstruct);
  return current_time;
}

void start_log() {
  log_file = fopen("mll.log", "a");
  setvbuf(log_file, NULL, _IOLBF, 1024);
  Log("======================================================");
  Log("NNForMLL: A program for multi-label learning");
  Log("(c) Abhishek Kumar");
  Log("Details: https://github.com/abhishek-kumar/NNForMLL");
  Log("");
  Log("Program started at %s", get_current_datetime().c_str());
  Log("======================================================");
  Log("");
}

void finish_log() {
  Log("Program finished at %s", get_current_datetime().c_str());
  fclose(log_file);
}
