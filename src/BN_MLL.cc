#include <math.h>

#include "BN_MLL.h"
#include "cross_validate.h"
#include "io.h"
#include "logging.h"
#include "parameters.h"

using namespace std;

// Constructors
BN_MLL::BN_MLL(io & fileio, dimensions dim) :
    m(fileio.xtr.size()), p(dim.p), d(dim.h), k(dim.k),
    xtr(fileio.xtr), ytr(fileio.ytr),
    wopt(0), C(0.0), linearity(0.0), counter(0) { }

BN_MLL::BN_MLL(
  data_t& xtrain, data_t& ytrain, dimensions dim,
  floatnumber C_) :
    m(xtrain.size()), p(dim.p), d(dim.h), k(dim.k),
    xtr(xtrain), ytr(ytrain),
   wopt(0), C(C_), linearity(0.0), counter(0)  { }

void BN_MLL::Train() {
  // Initialize parameters for LBFGS
  if(wopt) delete wopt;
  wopt = new parameters(p,d,k,true);
  floatnumber bestloss = 1e+5;
  lbfgs_parameter_t param; int ret=0; lbfgs_parameter_init(&param);
  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;

  // Run the LBFGS training process
  counter = 0; // count of iterations done so far
  Compatibility::model = this;
  ret = lbfgs(wopt->N, wopt->getvector(), &bestloss, Compatibility::evaluate, Compatibility::progress, &xtr, &param);

  // Report the optimization result.
  Log("    L-BFGS optimization terminated with status code = %d", ret);
  Log("    Loss (Regularized NLL) after gradient descent = %f", bestloss);
}

// Given a small subset dataset, train on the training subset and test on
// the test subset and return loss values.
error_t TrainAndTest(io& dataset, dimensions dim, floatnumber C) {
  // Train
  BN_MLL bn_mll(dataset.xtr, dataset.ytr, dim, C);
  bn_mll.Train();

  // Test and return loss
  return bn_mll.Test(dataset.xte, dataset.yte);
}

void BN_MLL::Train(cv_params cv) {
  // cross validate to find regularization strength
  this->C = FindBestC(cv, xtr, ytr, dimensions(p, d, k),
      TrainAndTest, false);

  // Train with the new C value.
  Train();
}

error_t BN_MLL::Test(data_t xtest, data_t ytest) {
  int sz = xtest.size();
  floatnumber y_hata[k], y_hat[k], ha[d], h[d], curloss, hl;
  error_t loss = error_t();
  record_t xrecord, yrecord;

  for(int i = 0; i < sz; ++i) {
    xrecord = xtest[i]; yrecord = ytest[i];
    forwardPropagate(xrecord, *wopt, y_hata, y_hat, ha, h);
    calculateLosses(y_hata, y_hat, yrecord, *wopt, curloss, hl);
    loss.nll += curloss; loss.hl += hl; if(hl>0) ++(loss.sl);

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

inline void BN_MLL::calculateJacobian(
    record_t const & x, record_t const & y,
    const floatnumber *y_hata, const floatnumber *y_hat,
    const floatnumber *ha, const floatnumber *h,
    parameters const & w, parameters & jacobian) {

  // Calculate deltaLoss / deltaActivation for both levels
  floatnumber dy[k], dh[d];
  for(int j=0; j<k; ++j)
  {
    floatnumber yi = 2*y[j] - 1; // convert to {-1,+1}
    floatnumber minusyi = -yi;
    dy[j] = minusyi*exp(minusyi*y_hata[j]) / (1.0+exp(minusyi*y_hata[j]));
    dy[j] -= linearity*yi;
  }

  for(int j=0; j<d; ++j)
  {
    floatnumber delta=0.0;
    for(int kk=0; kk<k; ++kk)
      delta += dy[kk]*w.val(1,j,kk);
    dh[j] = delta*(1.0 + linearity - pow(tanh(ha[j]),2) );
  }

  // Calculate gradients (w/o regularization)
  for(int i=0; i<p; ++i)
    for(int j=0; j<d; ++j)
      jacobian.val(0,i,j) += dh[j]*x[i];
  for(int j=0; j<d; ++j)
    jacobian.val(0,p,j) += dh[j];

  for(int i=0; i<d; ++i)
    for(int j=0;j<k; ++j)
      jacobian.val(1,i,j) += dy[j]*h[i];
  for(int j=0; j<k; ++j)
    jacobian.val(1,d,j) += dy[j];
};

void BN_MLL::forwardPropagate(  record_t const & x,
                                parameters const & w,
                    floatnumber *y_hata_,
                    floatnumber *y_hat_,
                    floatnumber *ha_,
                    floatnumber *h_) {
  // Layer 0
  for(int j=0; j<d; ++j)
    ha_[j] = w.val(0,p,j);
  for(int i=0; i<p; ++i)
    for(int j=0; j<d; ++j)
      ha_[j] += x[i]*w.val(0,i,j);
  for(int j=0; j<d; ++j)
    h_[j] = tanh(ha_[j]) + linearity*ha_[j];

  // Layer 1
  for(int j=0; j<k; ++j)
    y_hata_[j] = w.val(1,d,j);
  for(int i=0; i<d; ++i)
    for(int j=0; j<k; ++j)
      y_hata_[j] += h_[i]*w.val(1,i,j);
  for(int j=0; j<k; ++j)
    y_hat_[j] = 1.0/(1.0+exp((-1)*y_hata_[j]));
}

void BN_MLL::calculateLosses(
  const floatnumber *y_hata, const floatnumber *y_hat, record_t const & y,
  parameters const & w, floatnumber & nll, floatnumber & hl) {

  // All losses without regularization
  nll = 0; hl = 0;

  for(int i=0; i<k; ++i) {
    floatnumber yi = 2*y[i] - 1;  // convert to {-1,+1}
    floatnumber minusyi = (-1)*yi;
    nll += log(1.0+exp(minusyi*y_hata[i])) - linearity*yi*y_hata[i];
    if(  (y_hat[i]<0.5 && y[i]>=0.5) || (y_hat[i]>=0.5 && y[i]<0.5) )
      ++hl;
  }
}

// This section is created because lbfgs needs function pointers
// for functions that calculate things like losses (e.g. evaluate() ).
// These function pointers cannot point to class member functions because
// there is a type mismatch.
// Given that our program is single threaded, we can let the calling class
// set the vars that the evaluate() fn needs, as static vars temporarily.
// We can avoid this approach by wrapping functions and passing the wrapped
// type, but this is simple and avoids debugging nightmares.

// Internal function to be used by the LBFGS library.
lbfgsfloatval_t Compatibility::evaluate(
    void *instance,
    const lbfgsfloatval_t *wv,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step) {

  // Copy over some needed vars locally
  int p = model->p, d = model->d, k = model->k, m = model->m;
  data_t& xtr = model->xtr;
  data_t& ytr = model->ytr;
  floatnumber C = model->C;

  floatnumber curloss=0.0, loss=0.0, hl=0.0; // NLL and HammingLoss
  floatnumber wrongtags=0,wronglabels=0;
  const parameters w = parameters(p, d, k, wv); //w.init(p,d,k,wv);
  parameters jacobian(p, d, k, false);


  for(int i = 0; i < m; ++i) {
    record_t x = xtr[i]; record_t y = ytr[i];

    floatnumber y_hata[k], y_hat[k], ha[d], h[d];
    model->forwardPropagate(x,w,y_hata, y_hat,ha,h);

    model->calculateJacobian(x,y,y_hata,y_hat,ha,h,w,jacobian);
    model->calculateLosses(y_hata,y_hat,y,w, curloss, hl);

    loss += curloss;
    wrongtags += hl;
    if(hl>0) ++wronglabels;
  }

  // Add regularization
  floatnumber wnorm = 0;
  for(int i=0; i<n; ++i)
  {
    g[i] = C*jacobian[i] + 2*m*wv[i];
    wnorm += pow(wv[i],2);
  }
  jacobian.destroy();
  loss = C*loss + m*wnorm;
  return loss;
}

// Anything on the heap is destroyed here
BN_MLL::~BN_MLL() {
   if (wopt) {
     wopt->destroy();
     delete wopt;
   }
}

// Internal function to be used by the LBFGS library.
// The return value of this function outputs the progress made so far at periodic intervals.
int Compatibility::progress(void *instance,
            const lbfgsfloatval_t *x,
            const lbfgsfloatval_t *g,
            const lbfgsfloatval_t fx,
            const lbfgsfloatval_t xnorm,
            const lbfgsfloatval_t gnorm,
            const lbfgsfloatval_t step,
            int n,
            int k,
            int ls) {
  int & counter = model->counter;
  ++counter;
  if(counter % 1000 != 0)
    return 0;
    Log("    Iteration %d | "
        "loss: %f; w-norm = %f, jacobian-norm = %f, step = %f", \
        counter, fx, xnorm, gnorm, step);
    return 0;
}

BN_MLL * Compatibility::model;
