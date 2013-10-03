#include "BR_MLL.h"
#include "io.h"
#include "logging.h"
#include "parameters.h"
#include "types.h"


BR_MLL::BR_MLL(io & fileio, dimensions dim) :
   p(dim.p), d(dim.h), k(dim.k), xtr(fileio.xtr), ytr(fileio.ytr),
   m(xtr.size()), ytr_tag(k, data_t()), baseModels() {
    // Create k separate label-vectors because we are going to train k models

    // For each example
    for(data_t::iterator itr = ytr.begin(); itr != ytr.end(); ++itr ) {

      // For each tag
      for(int i=0; i<k; ++i) {
        // Note: data_t contains record_t which is a vector<floatnumber>; and ytr_tag is a vector<data_t>
        ytr_tag[i].push_back(record_t(1, (*itr)[i]));
      }
    }

    // Initialize the k base models
    for(int i=0; i<k; ++i) {
      // A shallow copy and destruct is OK because wopt is not allocated until fit() is called.
      baseModels.push_back(new BN_MLL(xtr, ytr_tag[i], dimensions(dim.p, dim.h, 1), 0.0));
    }

}

void BR_MLL::Train(cv_params cv) {

  // Train the k base models
  for(int i=0; i<k; ++i) {
    baseModels[i]->Train(cv);
  }
}

error_t BR_MLL::Test(data_t xtest, data_t ytest) {
  int sz = xtest.size();
  floatnumber y_hat[k], curloss = 0.0, wrongtags = 0.0;
  error_t loss = error_t();
  record_t xrecord, yrecord;

  // label vector for each base model
  vector<data_t> ytest_pertag(k, data_t());
  for(int tag=0; tag < k; ++tag) {
    for(int docno=0; docno < sz; ++docno)
      ytest_pertag[tag].push_back(record_t(1, ytest[docno][tag]));
  }

  // Calculate loss metrics per example
  for(int i = 0; i < sz; ++i)
  {
    xrecord = xtest[i]; yrecord = ytest[i];
    wrongtags = 0.0;

    for(int j=0; j<k; ++j) {
      floatnumber y_hata_tag, y_hat_tag;
      BN_MLL& tagModel = *(baseModels[j]);
      parameters& weights = *(tagModel.getParameters());
      floatnumber temp = 0.0, junk[d], junk2[d];
      tagModel.forwardPropagate(xrecord, weights, &y_hata_tag, &y_hat_tag, junk, junk2);
      tagModel.calculateLosses(&y_hata_tag, &y_hat_tag, ytest_pertag[j][i], weights, curloss, temp);
      y_hat[j]  = y_hat_tag;
      loss.nll += curloss;
      wrongtags += temp;
    }
    loss.hl += wrongtags; if(wrongtags>0) ++(loss.sl);

    // Number of relevant tags (for normalized RL)
    floatnumber r=0; for(int temp=0; temp<k; ++temp) r+=yrecord[temp];

    // Ranking Loss
    floatnumber rl=0.0, rl2 =0.0;
    for(int pid=0; pid<k; ++pid)
      for(int nid=0; nid<k; ++nid)
        if(yrecord[pid] > 0.5 && yrecord[nid] < 0.5)
        {
          if(y_hat[pid]<y_hat[nid]) ++rl, ++rl2;
          if(y_hat[pid]==y_hat[nid]) rl+=0.5, ++rl2;
        }
    loss.rl += rl;
    loss.nrl += rl2 / float(r*(k-r));

    // One Error
    floatnumber ymax=0.0; int argmaxy = -1;
    for(int ii=0; ii<k; ++ii)
      if(y_hat[ii] > ymax)
      {
        ymax = y_hat[ii];
        argmaxy = ii;
      }
    if(yrecord[argmaxy] < 0.5) ++(loss.oe);

    // Average Precision
    floatnumber oe=0.0;
    for(int ii=0; ii<k; ++ii)
    {
      floatnumber rank=0.0; for(int iii=0; iii<k; ++iii) if(y_hat[iii] >= y_hat[ii]) ++rank;
      floatnumber count = 0.0; for(int iii=0; iii<k; ++iii) if(y_hat[iii] >= y_hat[ii]) count += yrecord[iii];
      oe += (yrecord[ii])*(count / rank);
    }
    loss.avprec += oe/float(r);
  }
  loss.hl /= sz*k; loss.sl /= sz, loss.rl /= sz; loss.nrl /= sz; loss.oe /= sz; loss.avprec /= sz;
  return loss;
}

BR_MLL::~BR_MLL() {
  // Destroy the base models
  for(vector<BN_MLL *>::iterator itr = baseModels.begin(); itr != baseModels.end(); ++itr) {
    delete (*itr);
  }
}
