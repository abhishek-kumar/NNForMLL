#include "nn.h"
#include "io.h"
#include <time.h>
#include "parameters.h"
#include "singleLayerNN.h"
#include "fullyconnectedNN.h"
#include "BRSingleLayerNN.h"

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
  log("======================================================");
  log("NNForMLL: A program for multi-label learning");
  log("(c) Abhishek Kumar");
  log("Details: https://github.com/abhishek-kumar/NNForMLL");
  log("");
  log("Program started at %s", get_current_datetime().c_str());
  log("======================================================");
  log("");
}

void finish_log() {
  log("Program finished at %s", get_current_datetime().c_str());
  fclose(log_file);
}

int main(int argc, char **argv)
{
  if(argc<7) {
    cerr << "Usage: "
         << "$ mll <1|2|3> <train file> <test file> <p> <h> <k> [singleLayerC]"
         << endl;
    return 1;
  }

  // Initialization
  srand ( time(NULL) );
  floatnumber singleLayerC = -1.0;
  error_t loss;

  // Read data
  char *trainfile = argv[2]; char *testfile = argv[3];
  int p = atoi(argv[4]), d=atoi(argv[5]), k=atoi(argv[6]);
  io fileio; fileio.readTrainingData(trainfile,p,k); fileio.readTestData(testfile,p,k);
  fileio.normalize(p);

  // Parse flags
  if(atoi(argv[1]) == 1) {
    start_log();
    singleLayerNN* BNMLL;

    // Check if we need to learn regularization weights
    if (argc > 7) {
      singleLayerC = atof(argv[7]);
      BNMLL = new singleLayerNN(fileio.xtr, fileio.ytr, p, d, k, singleLayerC);
      log("Training model with regularization weight C = %f", singleLayerC);
      BNMLL->fit();
    } else {
      // We need to learn hyperparams
      BNMLL = new singleLayerNN(fileio, p, d, k);
      log("Training model. Regularization weight will be learnt via "
          "cross validation. This might take some time since training "
          "will be done ~15 times on the training set.");
      BNMLL->train(-12, 13, 2);
      singleLayerC = BNMLL->getRegularizationStrength();
    }
    loss = BNMLL->test(fileio.xte, fileio.yte);

    // Print out some interesting tag-correlations for first 6 hidden units
    printTagCorrelations(*(BNMLL->getParameters()), 6);
  } else if(atoi(argv[1]) == 2) {
    if(argc < 8) {
      cerr << "You must provide 7 arguments, the last one being C"
           << endl;
      return 1;
    }
    singleLayerC = atof(argv[7]);
    //bestC = trainFullyConnectedNN(fileio, p, d, k, trainfile, singleLayerC);
    cerr << "SLN-MLL is temporarily unsupported while I refactor the code "
         << "Please use BN-MLL (first argument = 1) "
         << "or BR-MLL (first argument = 3) instead"
         << endl;
    return 1;
  } else if(atoi(argv[1]) == 3) {
    BRSingleLayerNN model(fileio, p, d, k);
    model.train(-12, 13, 2);
    loss = model.test(fileio.xte, fileio.yte);
  } else {
    cerr << "Error, unknown first argument provided. "
         << "Must be either 1 (BN-MLL), 2 (SLN-MLL) or 3 (BR-BNMLL)."
         << endl;
    return 1;
  }

  // Print results
  log("Performance on Test Set '%s':", testfile);
  loss.print();
  log_stdout("  Finished training and testing. Performance on Test Set:");
  loss.print_stdout();
  finish_log();
}
