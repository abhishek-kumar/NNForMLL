/*
 * cross_validate.cc
 *
 *  Created on: Oct 2, 2013
 *      Author: abhishekkr
 */

#include "cross_validate.h"
#include "io.h"
#include "logging.h"
#include "types.h"
#include <math.h>
#include <numeric>
#include <stdlib.h>

floatnumber FindBestC(
    cv_params cv,
    data_t& xtrain, data_t& ytrain, dimensions dim,
    callback_fn evaluate_fn,
    bool silent_mode) {

  // Create a shuffled ordering of training data
  int m = xtrain.size();
  int indexes[m];
  for(int i=0; i<m; ++i) {
    indexes[i] = i;
  }
  randomShuffle(indexes, m);

  // cross validate to find regularization strength
  floatnumber bestC=0.0, lastNLL=1e10, lasttolastNLL=1e10;
  floatnumber bestNllMean=1e10;
  for(int cc = cv.lower_bound_power_of_2;
      cc <= cv.upper_bound_power_of_2;
      cc += cv.step_size_power_of_2) {
    floatnumber C = pow(2,cc);
    record_t cvLossesNll, cvLossesHl, cvLossesSl, cvLossesRl, cvLossesNrl;
    record_t cvLossesOe, cvLossesAvprec;

    if (!silent_mode) {
      Log("Regularization Parameter Training: Evaluating with C = 2^%d = %f.", cc, C);
    }

    // For each fold
    for(int fold = 0; fold < cv.num_folds; ++fold) {
      int testsetstart = fold*float(m)/cv.num_folds;
      int testsetend = testsetstart+float(m)/cv.num_folds;
      io cv_data;

      for(int i=0; i<testsetstart; ++i)
      {
        cv_data.xtr.push_back(xtrain[indexes[i]]);
        cv_data.ytr.push_back(ytrain[indexes[i]]);
      }
      for(int i=testsetstart; i<testsetend; ++i)
      {
        cv_data.xte.push_back(xtrain[indexes[i]]);
        cv_data.yte.push_back(ytrain[indexes[i]]);
      }
      for(int i=testsetend; i<m; ++i)
      {
        cv_data.xtr.push_back(xtrain[indexes[i]]);
        cv_data.ytr.push_back(ytrain[indexes[i]]);
      }

      if (!silent_mode) {
        Log("  Cross validation: training model on fold #%d", fold + 1);
      }
      error_t loss = evaluate_fn(cv_data, dim, C);

      cvLossesNll.push_back(loss.nll); cvLossesHl.push_back(loss.hl); cvLossesSl.push_back(loss.sl);
      cvLossesRl.push_back(loss.rl); cvLossesNrl.push_back(loss.nrl); cvLossesOe.push_back(loss.oe); cvLossesAvprec.push_back(loss.avprec);
    }

    floatnumber nllMean, nllStdev, hlMean, hlStdev, slMean, slStdev, rlMean, rlStdev, nrlMean,
        nrlStdev, oeMean, oeStdev, avprecMean, avprecStdev;
    mean_and_stdev(cvLossesNll, &nllMean, &nllStdev);
    mean_and_stdev(cvLossesHl, &hlMean, &hlStdev);
    mean_and_stdev(cvLossesSl, &slMean, &slStdev);
    mean_and_stdev(cvLossesRl, &rlMean, &rlStdev);
    mean_and_stdev(cvLossesNrl, &nrlMean, &nrlStdev);
    mean_and_stdev(cvLossesOe, &oeMean, &oeStdev);
    mean_and_stdev(cvLossesAvprec, &avprecMean, &avprecStdev);

    if (!silent_mode) {
      Log("  Cross validation complete. Results:");
      Log("  NLL: %0.3f+-%0.3f, HL: %0.3f+-%0.3f, SL: %0.3f+-%0.3f, "
          "RL: %0.3f+-%0.3f, NRL: %0.3f+-%0.3f, OE: %0.3f+-%0.3f, "
          "AVPREC: %0.3f+-%0.3f, ", nllMean, nllStdev, \
          hlMean, hlStdev, slMean, slStdev, rlMean, rlStdev, \
          nrlMean, nrlStdev, oeMean, oeStdev, avprecMean, avprecStdev);
    }

    if(nllMean<bestNllMean) {
      bestC = C;
      bestNllMean = nllMean;
    }
    if( (lasttolastNLL+0.01) < lastNLL && (lastNLL+0.01) < nllMean) {
      // we're doing worse down this path. stop.
      if (!silent_mode) {
        Log("  Regularization Parameter Training: "
            "Further parameter checking is futile, losses seem to increase. "
            "Halting at bestC: %f (currently evaluated C: %f )\n", bestC, C);
      }
      break;
    }
    lasttolastNLL = lastNLL; lastNLL = nllMean;
  }
  if (!silent_mode) {
    Log("Regularization parameter training "
        "complete. Best C: %f; and Loss (NLL) = %f", bestC, bestNllMean);
  }
  return bestC;
}

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


