
namespace fullyConnectedNN
{
int m;                           // Number of training examples
int p,d,k;                       //dimensions of the parameters and gradient
floatnumber C;                           //Regularization, lower => stronger regularization
floatnumber C2;                           //Regularization, lower => stronger regularization
floatnumber linearity = 0.0;    //1.0e-5;  // To speed up convergence
parameters * wopt = 0;
data_t xtr, ytr, xte, yte;
int counter=0;                  // Keeps a count of LBFGS iterations done

void forwardPropagate(record_t const & x,
                      parameters const & w,
                      floatnumber *y_hata_,
                      floatnumber *y_hat_,
                      floatnumber *ha_,
                      floatnumber *h_)
{
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

  // Layer 2
  for(int j=0; j<k; ++j)
    y_hata_[j] += w.val(2,p,j);
  for(int i=0; i<p; ++i)
    for(int j=0; j<k; ++j)
      y_hata_[j] += x[i]*w.val(2,i,j);

  for(int j=0; j<k; ++j)
    y_hat_[j] = 1.0/(1.0+exp((-1)*y_hata_[j]));
}


inline void calculateLosses(const floatnumber *y_hata,
                     const floatnumber *y_hat,
                     record_t const & y,
                     parameters const & w,
                     floatnumber & nll,
                     floatnumber & hl)
{
  // All losses without regularization
  nll = 0; hl = 0;

  for(int i=0; i<k; ++i)
  {
    floatnumber yi = 2*y[i] - 1; // convert to {-1,+1}
    floatnumber minusyi = (-1)*yi;
    nll += log(1.0+exp(minusyi*y_hata[i])) - linearity*yi*y_hata[i];
    if(  (y_hat[i]<0.5 && y[i]>=0.5) || (y_hat[i]>=0.5 && y[i]<0.5) )
      ++hl;
  }
}

inline void calculateJacobian(record_t const & x,
                              record_t const & y,
                              const floatnumber *y_hata, const floatnumber *y_hat,
                              const floatnumber *ha, const floatnumber *h,
                              parameters const & w,
                              parameters & jacobian)
{
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

  for(int i=0; i<p; ++i)
    for(int j=0; j<k; ++j)
      jacobian.val(2,i,j) += dy[j]*x[i];
  for(int j=0; j<k; ++j)
    jacobian.val(2,p,j) += dy[j];

}

static lbfgsfloatval_t evaluate(
  void *instance,
  const lbfgsfloatval_t *wv,
  lbfgsfloatval_t *g,
  const int n,
  const lbfgsfloatval_t step
  )
{

  floatnumber curloss=0.0, loss=0.0, hl=0.0; // NLL and HammingLoss
  floatnumber wrongtags=0,wronglabels=0;
  const parameters w = parameters(p,d,k,wv); //w.init(p,d,k,wv);
  parameters jacobian(p,d,k,false);

  for(int i=0; i<m; ++i)
  {
    record_t x = xtr[i]; record_t y = ytr[i];

    floatnumber y_hata[k], y_hat[k], ha[d], h[d];
    forwardPropagate(x,w,y_hata, y_hat,ha,h);

    calculateJacobian(x,y,y_hata,y_hat,ha,h,w,jacobian);
    calculateLosses(y_hata,y_hat,y,w, curloss, hl);

    loss += curloss;
    wrongtags += hl;
    if(hl>0) ++wronglabels;
  }

  // Add regularization
  floatnumber wnorm = 0;
  for(int i=0; i<jacobian.layer1N + jacobian.layer2N; ++i)
  {
    g[i] = C*jacobian[i] + 2*m*wv[i];
    wnorm += pow(wv[i],2);
  }
  // Extra regularization for level 2 parameters
  for(int i=jacobian.layer1N + jacobian.layer2N; i<n; ++i)
  {
    g[i] = C*jacobian[i] + 2*m*C2*wv[i];
    wnorm += C2*pow(wv[i],2);
  }
  jacobian.destroy();
  loss = C*loss + m*wnorm;
  return loss;
}


static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    ++counter;
    if(counter % 1000 != 0)
      return 0;
    printf("\t\tIteration %d | ", counter);
    printf("loss: %f; ", fx);
    printf("w-norm = %f, jacobian-norm = %f, step = %f\n", xnorm, gnorm, step);
    return 0;
}

void fit(data_t xtrain, data_t ytrain, int pp, int dd, int kk, floatnumber CC, parameters * slW, floatnumber Clevel2)
{
  p = pp; d = dd; k = kk; C = CC; C2 = Clevel2;
  xtr = xtrain; ytr = ytrain;
  m = xtr.size();

  // Initialize parameters for LBFGS
  if(wopt) delete wopt; // In case this model has been trained earlier too
  wopt = new parameters(p,d,k,false);

  floatnumber *wvto = wopt->getvector(); floatnumber *wvfrom = slW->getvector();
  for(int i=0; i < slW->N; ++i, ++wvto, ++wvfrom)
      *wvto = *wvfrom;
  /*
  cout << "FullyConnectedNN: Printing initialized weights before running LBFGS\n";
  for(int i=0; i < wopt->N; ++i)
    cout << "(" << i+1 << ", " << ((*wopt)[i]) << "), ";
  cout << endl;
  */
  floatnumber bestloss = 1e+5;
  lbfgs_parameter_t param; int ret=0; lbfgs_parameter_init(&param);
  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;

  // Run the LBFGS training process
  counter = 0; // count of iterations done so far
  ret = lbfgs(wopt->N, wopt->getvector(), &bestloss, evaluate, progress, &xtr, &param);

  /*
  cout << "FullyConnectedNN: Printing Learned weights\n";
  for(int i=0; i < wopt->N; ++i)
    cout << "(" << i+1 << ", " << ((*wopt)[i]) << "), ";
  cout << endl;
  */

  /* Report the result. */
  printf("\tL-BFGS optimization terminated with status code = %d", ret);
  printf("; bestLoss = %f\n", bestloss);

}

void test(data_t xtest,
          data_t ytest,
          floatnumber & nll_,
          floatnumber & sl_,
          floatnumber & hl_,
          floatnumber & rl_,
          floatnumber & nrl_,
          floatnumber & oneerror_,
          floatnumber & avprec_)
{
  int sz = xtest.size();
  floatnumber y_hata[k], y_hat[k], ha[d], h[d], curloss, hl;
  floatnumber NLL = 0.0, HL=0.0, SL=0.0, RL = 0.0, normalizedRL = 0.0, oneError=0.0, avPrec=0.0;
  record_t xrecord, yrecord;

  for(int i=0; i<sz;++i)
  {
    xrecord = xtest[i]; yrecord = ytest[i];
    forwardPropagate(xrecord, *wopt, y_hata, y_hat, ha, h);
    calculateLosses(y_hata, y_hat, yrecord, *wopt, curloss, hl);
    NLL += curloss; HL += hl; if(hl>0) ++SL;

    // Number of relevant tags (for normalized RL)
    int r=0; for(int temp=0; temp<k; ++temp) r+=yrecord[temp];

    // Ranking Loss
    floatnumber rl=0.0, rl2 = 0.0;
    for(int pid=0; pid<k; ++pid)
      for(int nid=0; nid<k; ++nid)
        if(yrecord[pid] > 0.5 && yrecord[nid] < 0.5)
        {
          if(y_hat[pid]<y_hat[nid]) ++rl, ++rl2;
          if(y_hat[pid]==y_hat[nid]) rl+=0.5, ++rl2;
        }
    RL += rl;
    normalizedRL += rl2 / (r*(k-r));

    //One Error
    floatnumber ymax=0.0; int argmaxy = -1;
    for(int ii=0; ii<k; ++ii)
      if(y_hat[ii] > ymax) {
        ymax = y_hat[ii];
        argmaxy = ii;
      }
    if (argmaxy == -1) {
      fprintf(stderr, "True label has all zeroes. Cannot calculate one-error.\n");
    } else {
      if(yrecord[argmaxy] < 0.5) {
        ++oneError;
      }
    }

    // Average Precision
    floatnumber oe=0.0;
    for(int ii=0; ii<k; ++ii)
    {
      floatnumber rank=0.0; for(int iii=0; iii<k; ++iii) if(y_hat[iii] >= y_hat[ii]) ++rank;
      floatnumber count = 0.0; for(int iii=0; iii<k; ++iii) if(y_hat[iii] >= y_hat[ii]) count += yrecord[iii];
      oe += (yrecord[ii])*(count / rank);
    }
    avPrec += oe/r;
  }
  HL /= sz*k; SL /= sz, RL /= sz; normalizedRL /= sz; oneError /= sz; avPrec /= sz;
  nll_ = NLL; sl_ = SL; hl_ = HL; rl_ = RL; nrl_ = normalizedRL; oneerror_ = oneError; avprec_ = avPrec;
}

}
