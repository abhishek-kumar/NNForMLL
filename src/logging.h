#ifndef LOGGING_H_
#define LOGGING_H_

#include "types.h"
#include <string>

// log file.
extern FILE * log_file;

// Logging functions
void Log(const std::string message, ...);
void Log(error_t& losses);
void Log_stdout(const std::string message, ...);
void Log_stdout(error_t& losses);

// Logging setup
const std::string get_current_datetime();
void start_log();
void finish_log();

#endif
