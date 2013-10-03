#include "BN_MLL.h"
#include "BR_MLL.h"
#include "io.h"
#include "logging.h"
#include "parameters.h"
#include <iostream>
#include <stdarg.h>
#include <stdlib.h>
// #include "fullyconnectedNN.h"


int main(int argc, char **argv) {
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

    // Check if we need to learn regularization weights
    if (argc > 7) {
      singleLayerC = atof(argv[7]);
      BN_MLL bn_mll(fileio.xtr, fileio.ytr, dimensions(p, d, k), singleLayerC);
      Log("Training model with regularization weight C = %f", singleLayerC);
      bn_mll.Train();
      loss = bn_mll.Test(fileio.xte, fileio.yte);
      // Print out some interesting tag-correlations - top 6 for each hidden unit
      LogTagCorrelations(*(bn_mll.getParameters()), 6);
    } else {
      // We need to learn hyperparams
      BN_MLL bn_mll(fileio, dimensions(p, d, k));
      Log("Training model. Regularization weight will be learnt via "
          "cross validation. This might take some time since training "
          "will be done ~15 times on the training set.");
      bn_mll.Train(cv_params(-12, 12, 1, 8));
      loss = bn_mll.Test(fileio.xte, fileio.yte);
      // Print out some interesting tag-correlations - top 6 for each hidden unit
      LogTagCorrelations(*(bn_mll.getParameters()), 6);
    }
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
    BR_MLL model(fileio, dimensions(p, d, k));
    model.Train(cv_params(-12, 13, 2, 5));
    loss = model.Test(fileio.xte, fileio.yte);
  } else {
    cerr << "Error, unknown first argument provided. "
         << "Must be either 1 (BN-MLL), 2 (SLN-MLL) or 3 (BR-BNMLL)."
         << endl;
    return 1;
  }

  // Print results
  Log("Performance on Test Set '%s':", testfile);
  Log(loss);
  Log_stdout("  Finished training and testing. Performance on Test Set:");
  Log_stdout(loss);
  finish_log();
}
